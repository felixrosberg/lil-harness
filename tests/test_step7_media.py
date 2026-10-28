"""Step 7: the image and video tools, and the step 7 example. Needs the media extra."""

import importlib.util
import io
import json
from pathlib import Path

import pytest

pil = pytest.importorskip("PIL.Image")
pytest.importorskip("av")

from lil_harness.agent.loop import run_agent  # noqa: E402
from lil_harness.backends.fake import FakeBackend  # noqa: E402
from lil_harness.backends.openai_compat import messages_to_openai  # noqa: E402
from lil_harness.tools.media import MediaTools, clock, load_image  # noqa: E402
from lil_harness.types import AssistantTurn, Message, ToolCall  # noqa: E402

EXAMPLE = Path(__file__).parent.parent / "examples" / "media.py"


def call(name, **arguments):
    return ToolCall(id=f"call_{name}", name=name, arguments=arguments)


spec = importlib.util.spec_from_file_location("media_example", EXAMPLE)
step7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step7)


@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    return step7.make_samples(tmp_path_factory.mktemp("media"))


def decoded(image):
    return pil.open(io.BytesIO(image.data))


def test_clock_formats_video_times():
    assert (clock(4.95), clock(62.5), clock(3723)) == ("0:05.0", "1:02.5", "1:02:03.0")


def test_view_image_shrinks_big_images(samples):
    output = MediaTools(samples, max_side=800).view_image("01.png")

    assert decoded(output.images[0]).size == (800, 600)
    assert output.text == "01.png: 1600x1200 pixels, shown at 800x600."
    assert decoded(load_image(samples / "01.png", 400)).size == (400, 300)


def test_zoom_enlarges_the_region(samples):
    output = MediaTools(samples, max_side=800).zoom_image("01.png", 0.9, 0.9, 1.0, 1.0)

    assert decoded(output.images[0]).size == (800, 600)
    assert "pixels x 1440-1600, y 1080-1200" in output.text


async def test_bad_regions_and_paths_go_back_to_the_model(samples):
    registry = MediaTools(samples).registry()

    flipped = await registry.execute(call("zoom_image", path="01.png", left=0.5, top=0,
                                          right=0.2, bottom=1))  # fmt: skip
    outside = await registry.execute(call("view_image", path="../secret.png"))
    not_image = await registry.execute(call("view_image", path="clip.mp4"))

    assert flipped.is_error and "left < right" in flipped.content
    assert outside.is_error and "outside the workspace" in outside.content
    assert not_image.is_error and "isn't an image" in not_image.content


def test_video_info(samples):
    info = MediaTools(samples).video_info("clip.mp4")
    assert info.startswith("clip.mp4: 0:08.0 (8.0 s), 480x270, 15 fps, about 120 frames")


def test_video_overview_spreads_frames_over_the_stretch(samples):
    output = MediaTools(samples).video_overview("clip.mp4", frames=4, start=4, end=6)

    assert "4 frames from 0:04.0 to 0:06.0" in output.text
    assert "1 at 0:04.3, 2 at 0:04.7, 3 at 0:05.3, 4 at 0:05.7" in output.text
    assert decoded(output.images[0]).width == 1024


def test_video_frame_shows_that_moment(samples):
    media = MediaTools(samples)
    # The sample's background turns from blue to green at 5 seconds.
    before = decoded(media.video_frame("clip.mp4", 4.8).images[0]).getpixel((5, 5))
    after = decoded(media.video_frame("clip.mp4", 5.2).images[0]).getpixel((5, 5))

    assert before[2] > before[1]  # more blue than green
    assert after[1] > after[2]
    with pytest.raises(Exception, match="between 0 and 8.0"):
        media.video_frame("clip.mp4", 30)


async def test_the_loop_shows_the_model_what_the_tool_returned(samples):
    backend = FakeBackend([
        AssistantTurn(tool_calls=[call("view_image", path="02.png")]),
        AssistantTurn(content="A blue square."),
    ])  # fmt: skip

    result = await run_agent(backend, MediaTools(samples).registry(), [Message.user("Look")])

    assert result.answer == "A blue square."
    [tool_message] = [message for message in backend.calls[1].messages if message.role == "tool"]
    assert len(tool_message.images) == 1
    sent = messages_to_openai(backend.calls[1].messages)
    assert sent[-1]["content"][1]["type"] == "image_url"


# ---- the step 7 example -----------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        '{"shape": "circle", "color": "red"}',
        'Here it is:\n```json\n{"shape": "circle", "color": "red"}\n```',
        'Not this {nope}, but {"shape": "circle", "color": "red"}',
    ],
)
def test_parse_label_finds_the_json(reply):
    assert step7.parse_label(reply) == {"shape": "circle", "color": "red"}


def test_parse_label_without_json():
    assert step7.parse_label("I can't tell.") is None


async def test_annotate_attaches_the_image_and_parses_the_label(samples):
    label = {"shape": "triangle", "color": "green", "tag": "P9"}
    backend = FakeBackend([
        AssistantTurn(tool_calls=[call("zoom_image", path="03.png", left=0.9, top=0.9,
                                       right=1, bottom=1)]),
        AssistantTurn(content=json.dumps(label)),
    ])  # fmt: skip

    got, result = await step7.annotate_file(
        backend, MediaTools(samples), samples / "03.png", step7.ANNOTATE_PROMPT
    )

    assert got == label and result.steps == 2
    first_user = backend.calls[0].messages[1]
    assert "03.png" in first_user.content and len(first_user.images) == 1


def test_score_label_counts_matching_fields():
    expected = ("circle", "red", "K7")
    assert step7.score_label({"shape": "Circle ", "color": "red", "tag": "K1"}, expected) == 2
    assert step7.score_label(None, expected) == 0
