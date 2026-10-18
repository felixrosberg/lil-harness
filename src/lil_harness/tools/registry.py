"""Turn plain Python functions into tools a model can call.

The JSON schema the model sees is generated from the function's type hints, and the
same pydantic model validates the model's arguments. That way the schema and the
implementation can never drift apart.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel, ValidationError, create_model

from lil_harness.permissions import PermissionPolicy, Risk
from lil_harness.types import ToolCall, ToolResult


class ToolError(Exception):
    """Raise this from a tool to send the model a plain, actionable error message.

    Other exceptions are reported too, but with their type name, which reads like a
    crash. A ToolError reads like advice: "old_text appears 3 times; include more
    context".
    """


@dataclass
class ToolSpec:
    """Everything the harness knows about one tool."""

    name: str
    description: str
    fn: Callable[..., Any]
    args_model: type[BaseModel]
    # How much care a call needs; see permissions.py. Unlabelled tools get the most.
    risk: Risk = "execute"

    @property
    def parameters(self) -> dict[str, Any]:
        """JSON schema for the arguments, as sent to the model."""
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return schema


def _args_model_from_signature(fn: Callable[..., Any]) -> type[BaseModel]:
    # get_type_hints turns string annotations (from `from __future__ import annotations`
    # in the tool's module) back into real types, using that module's imports.
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for param in inspect.signature(fn).parameters.values():
        if param.name not in hints:
            raise TypeError(f"Tool {fn.__name__!r}: parameter {param.name!r} needs a type hint")
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (hints[param.name], default)
    return create_model(f"{fn.__name__}_args", **fields)


class ToolRegistry:
    """Holds the tools available to the model and runs them.

    With a `policy`, every call is checked before it runs (see permissions.py).
    Without one, every call runs, which is what the examples and evals want.
    """

    def __init__(self, policy: PermissionPolicy | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.policy = policy

    def tool(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        risk: Risk = "execute",
    ) -> Any:
        """Decorator: register a function as a tool. Its docstring becomes the description.

        Usable as `@registry.tool` or `@registry.tool(name="other_name")`. Pass
        `description` to override the docstring, for text that is only known at runtime,
        and `risk` to say what the tool can do: "read", "write" or "execute".
        """

        def register(tool_function: Callable[..., Any]) -> Callable[..., Any]:
            spec = ToolSpec(
                name=name or tool_function.__name__,
                description=description or inspect.cleandoc(tool_function.__doc__ or ""),
                fn=tool_function,
                args_model=_args_model_from_signature(tool_function),
                risk=risk,
            )
            if spec.name in self._tools:
                raise ValueError(f"Tool {spec.name!r} is already registered")
            self._tools[spec.name] = spec
            return tool_function

        return register(fn) if fn is not None else register

    def specs(self) -> list[ToolSpec]:
        """The tools to offer the model. Tools the policy denies outright are left out."""
        return [
            s for s in self._tools.values() if self.policy is None or self.policy.offers(s.risk)
        ]

    async def execute(self, call: ToolCall) -> ToolResult:
        """Run one tool call. Never raises: every failure becomes an error result.

        Errors go back to the model as tool results so it can correct itself,
        instead of crashing the loop.
        """
        spec = self._tools.get(call.name)
        if spec is None:
            available = ", ".join(sorted(self._tools)) or "none"
            return self._error(call, f"Unknown tool {call.name!r}. Available tools: {available}.")

        if call.arguments_error is not None:
            return self._error(
                call,
                f"Arguments for {call.name!r} were not valid JSON ({call.arguments_error}). "
                f"You sent: {call.raw_arguments!r}",
            )

        try:
            args = spec.args_model.model_validate(call.arguments)
        except ValidationError as e:
            problems = "; ".join(
                f"{'.'.join(map(str, err['loc'])) or 'arguments'}: {err['msg']}"
                for err in e.errors()
            )
            return self._error(call, f"Invalid arguments for {call.name!r}: {problems}")

        if self.policy is not None:
            decision = await self.policy.check(spec.name, spec.risk, args.model_dump())
            if not decision.allowed:
                return self._error(call, f"Permission denied. {decision.message}")

        try:
            output = spec.fn(**args.model_dump())
            if inspect.isawaitable(output):
                output = await output
        except ToolError as e:
            return self._error(call, str(e))
        except Exception as e:  # noqa: BLE001 - any tool failure is reported to the model
            return self._error(call, f"Tool {call.name!r} failed: {type(e).__name__}: {e}")

        content = output if isinstance(output, str) else json.dumps(output, default=str)
        return ToolResult(tool_call_id=call.id, name=call.name, content=content)

    @staticmethod
    def _error(call: ToolCall, message: str) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id, name=call.name, content=f"Error: {message}", is_error=True
        )
