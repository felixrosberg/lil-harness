"""Step 7: images in user messages and tool results, and how they reach the server."""

import pytest

from lil_harness.backends.openai_compat import DROPPED_IMAGE, messages_to_openai
from lil_harness.tools.registry import ToolRegistry
from lil_harness.types import Image, Message, ToolCall, ToolOutput, ToolResult

PNG = Image(b"\x89PNG fake", "image/png")


def call(name, **arguments):
    return ToolCall(id=f"call_{name}", name=name, arguments=arguments)


# ---- messages and the wire format --------------------------------------------


def test_image_from_path_guesses_the_format_and_makes_a_data_url(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"jpeg bytes")
    image = Image.from_path(tmp_path / "a.jpg")

    assert image.mime == "image/jpeg"
    assert image.to_data_url() == "data:image/jpeg;base64,anBlZyBieXRlcw=="
    with pytest.raises(ValueError):
        Image.from_path(tmp_path / "notes.txt")


def test_a_user_message_with_an_image_becomes_content_parts():
    [out] = messages_to_openai([Message.user("What is this?", images=[PNG])])

    assert out["role"] == "user"
    assert out["content"] == [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": PNG.to_data_url()}},
    ]


def test_messages_without_images_stay_plain_strings():
    [out] = messages_to_openai([Message.user("Hi")])
    assert out == {"role": "user", "content": "Hi"}


def test_tool_images_follow_the_run_of_tool_messages_in_one_user_message():
    first = call("view_image")
    second = ToolCall(id="call_2", name="read_file", arguments={})
    history = [
        Message.user("Look"),
        Message(role="assistant", tool_calls=[first, second]),
        Message.tool(ToolResult(first.id, "view_image", "a.png: 10x10", images=[PNG])),
        Message.tool(ToolResult(second.id, "read_file", "text")),
    ]

    out = messages_to_openai(history)

    # Both tool messages still directly follow the assistant turn; the image comes after.
    assert [message["role"] for message in out] == ["user", "assistant", "tool", "tool", "user"]
    assert out[2]["content"] == "a.png: 10x10\n(The image is in the next message.)"
    assert out[3]["content"] == "text"
    assert out[4]["content"] == [
        {"type": "text", "text": "Image from view_image (call call_view_image):"},
        {"type": "image_url", "image_url": {"url": PNG.to_data_url()}},
    ]


def test_max_images_keeps_the_newest_and_leaves_a_note_for_the_rest():
    history = [Message.user(f"image {number}", images=[PNG]) for number in range(3)]

    out = messages_to_openai(history, max_images=1)

    assert out[0]["content"][1] == {"type": "text", "text": DROPPED_IMAGE}
    assert out[1]["content"][1] == {"type": "text", "text": DROPPED_IMAGE}
    assert out[2]["content"][1]["type"] == "image_url"


async def test_a_tool_returning_images_puts_them_on_the_result():
    registry = ToolRegistry()

    @registry.tool(risk="read")
    def snapshot() -> ToolOutput:
        """Take a picture."""
        return ToolOutput("one picture", [PNG])

    @registry.tool(risk="read")
    def bare() -> Image:
        """Return just an image."""
        return PNG

    result = await registry.execute(call("snapshot"))
    assert (result.content, result.images) == ("one picture", [PNG])
    result = await registry.execute(call("bare"))
    assert (result.content, result.images) == ("", [PNG])
