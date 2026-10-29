"""Turn plain Python functions into tools a model can call.

The JSON schema the model sees is generated from the function's type hints, and the
same pydantic model validates the model's arguments. That way the schema and the
implementation can never drift apart.

A tool's docstring is split in two: the text before the first section (`Args:`,
`Returns:`, ...) becomes the tool's description, and each entry under `Args:` becomes
that argument's description in the schema. `Returns:` and other sections are for
people reading the code; the model sees the actual result instead.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias, get_type_hints

from pydantic import BaseModel, Field, ValidationError, create_model

from lil_harness.permissions import PermissionPolicy, Risk
from lil_harness.types import Image, ToolCall, ToolName, ToolOutput, ToolResult

JSONSchema: TypeAlias = dict[str, Any]
"""A JSON schema as a dict, like {"type": "object", "properties": {...}}."""
ArgDocs: TypeAlias = dict[str, str]
"""Argument name -> its description, from a docstring's Args: section."""

# Google-style docstring section headers, on a line of their own.
_SECTION = re.compile(r"^(Args|Arguments|Returns|Yields|Raises|Examples?|Notes?):\s*$")


class ToolError(Exception):
    """Raise this from a tool to send the model a plain, actionable error message.

    Other exceptions are reported too, but with their type name, which reads like a
    crash. A ToolError reads like advice: "old_text appears 3 times; include more
    context".
    """


@dataclass
class ToolSpec:
    """Everything the harness knows about one tool.

    Attributes:
        name: The name the model calls the tool by.
        description: What the tool does, as the model sees it.
        fn: The Python function (or bound method) that runs the tool.
        args_model: Pydantic model built from `fn`'s signature. It produces the JSON
            schema and validates the model's arguments.
        risk: How much care a call needs ("read", "write" or "execute"); see
            permissions.py. Unlabelled tools get the most.
    """

    name: ToolName
    description: str
    fn: Callable[..., Any]
    args_model: type[BaseModel]
    risk: Risk = "execute"

    @property
    def parameters(self) -> JSONSchema:
        """JSON schema for the arguments, as sent to the model.

        Returns:
            The schema without pydantic's auto-generated "title" fields, which only
            repeat the names and cost the model tokens.
        """
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return schema


def parse_docstring(doc: str | None) -> tuple[str, ArgDocs]:
    """Split a Google-style docstring into the summary and per-argument descriptions.

    Args:
        doc: The raw docstring, or None.

    Returns:
        A pair: the text before the first section header, and a dict mapping each
        argument named under `Args:` to its description (continuation lines joined).
    """
    lines = inspect.cleandoc(doc or "").splitlines()
    summary: list[str] = []
    args: ArgDocs = {}
    section: str | None = None
    current: str | None = None
    for line in lines:
        header = _SECTION.match(line)
        if header:
            section, current = header.group(1), None
            continue
        if section is None:
            summary.append(line)
        elif section in ("Args", "Arguments") and line.strip():
            entry = re.match(r"^\s{0,4}(\w+)(?:\s*\([^)]*\))?:\s*(.*)$", line)
            if entry and not line.startswith(" " * 5):
                current = entry.group(1)
                args[current] = entry.group(2).strip()
            elif current:
                args[current] = f"{args[current]} {line.strip()}".strip()
    return "\n".join(summary).strip(), args


def _args_model_from_signature(fn: Callable[..., Any], arg_docs: ArgDocs) -> type[BaseModel]:
    """Build a pydantic model whose fields are `fn`'s parameters.

    Args:
        fn: The tool function. Every parameter needs a type hint.
        arg_docs: Descriptions per parameter name, from the docstring's `Args:`.

    Returns:
        A pydantic model class named `<fn>_args`.

    Raises:
        TypeError: If a parameter has no type hint, since the schema can't be built.
    """
    # get_type_hints turns string annotations (from `from __future__ import annotations`
    # in the tool's module) back into real types, using that module's imports.
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for param in inspect.signature(fn).parameters.values():
        if param.name not in hints:
            raise TypeError(f"Tool {fn.__name__!r}: parameter {param.name!r} needs a type hint")
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (
            hints[param.name],
            Field(default, description=arg_docs.get(param.name)),
        )
    return create_model(f"{fn.__name__}_args", **fields)


class ToolRegistry:
    """Holds the tools available to the model and runs them.

    With a `policy`, every call is checked before it runs (see permissions.py).
    Without one, every call runs, which is what the examples and evals want.
    """

    def __init__(self, policy: PermissionPolicy | None = None) -> None:
        """Create an empty registry.

        Args:
            policy: Decides which calls may run. None means every call runs.
        """
        self._tools: dict[ToolName, ToolSpec] = {}
        self.policy = policy

    def tool(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        risk: Risk = "execute",
    ) -> Any:
        """Decorator: register a function as a tool.

        Usable as `@registry.tool`, as `@registry.tool(name="other_name")`, or as a
        plain call, `registry.tool(fn, risk="read")`.

        Args:
            fn: The function to register. Left out when used as `@registry.tool(...)`.
            name: The tool's name for the model. Defaults to the function's name.
            description: Replaces the docstring's summary, for text only known at
                runtime. Argument descriptions still come from the docstring.
            risk: What the tool can do: "read", "write" or "execute".

        Returns:
            The function unchanged (so it can still be called directly), or, without
            `fn`, a decorator that registers and returns it.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """

        def register(tool_function: Callable[..., Any]) -> Callable[..., Any]:
            """Build the ToolSpec for `tool_function` and store it."""
            summary, arg_docs = parse_docstring(tool_function.__doc__)
            spec = ToolSpec(
                name=name or tool_function.__name__,
                description=description or summary,
                fn=tool_function,
                args_model=_args_model_from_signature(tool_function, arg_docs),
                risk=risk,
            )
            if spec.name in self._tools:
                raise ValueError(f"Tool {spec.name!r} is already registered")
            self._tools[spec.name] = spec
            return tool_function

        return register(fn) if fn is not None else register

    def specs(self) -> list[ToolSpec]:
        """The tools to offer the model.

        Returns:
            Every registered tool, in registration order, except those whose risk
            level the policy denies outright.
        """
        return [
            spec
            for spec in self._tools.values()
            if self.policy is None or self.policy.offers(spec.risk)
        ]

    async def execute(self, call: ToolCall) -> ToolResult:
        """Run one tool call. Never raises: every failure becomes an error result.

        Errors go back to the model as tool results so it can correct itself,
        instead of crashing the loop. The checks run in order: known tool, valid
        JSON, valid arguments, permission, and finally the tool itself.

        Args:
            call: The tool call as parsed by the backend.

        Returns:
            The tool's output as a string (non-string outputs are JSON-encoded, and an
            `Image` or `ToolOutput`'s images go on the result's `images`), or an
            error result whose content starts with "Error:".
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

        images: list[Image] = []
        if isinstance(output, Image):
            output, images = "", [output]
        elif isinstance(output, ToolOutput):
            output, images = output.text, list(output.images)
        content = output if isinstance(output, str) else json.dumps(output, default=str)
        return ToolResult(tool_call_id=call.id, name=call.name, content=content, images=images)

    @staticmethod
    def _error(call: ToolCall, message: str) -> ToolResult:
        """Wrap `message` as an error result for `call`."""
        return ToolResult(
            tool_call_id=call.id, name=call.name, content=f"Error: {message}", is_error=True
        )
