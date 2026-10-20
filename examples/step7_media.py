"""Step 7 demo: a model that can see, labelling images and answering questions about video.

Needs the media extra and a vision model. Check yours can see with `ollama show
qwen3.5:4b`: "vision" should be under Capabilities.
    uv sync --extra media

Label every image in a folder, one fresh conversation per image, and save the labels
to labels.jsonl. Without --folder it makes sample images whose answers it knows, and
scores the model:
    uv run python examples/step7_media.py annotate ollama --model qwen3.5:4b

Ask about images or videos in a folder (default: the samples, with a short clip):
    uv run python examples/step7_media.py ask ollama --model qwen3.5:4b
    uv run python examples/step7_media.py ask ollama --folder D:\\clips -q "What happens?"

For your own labels, give --folder and --prompt saying which JSON fields you want.
"""

import asyncio
import json
import re
from pathlib import Path

import click
from openai import APIConnectionError, BadRequestError

from lil_harness.agent.loop import RunResult, run_agent
from lil_harness.backends.base import ChatBackend
from lil_harness.backends.ollama import OllamaBackend
from lil_harness.backends.vllm import VLLMBackend
from lil_harness.tools.media import MediaTools, load_image
from lil_harness.types import Message

SAMPLES = Path(__file__).resolve().parent.parent / ".lil_media"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# (file, shape, color, tag): what make_samples draws, and the answers it scores against.
SAMPLE_IMAGES = [
    ("01.png", "circle", "red", "K7"),
    ("02.png", "square", "blue", "M2"),
    ("03.png", "triangle", "green", "P9"),
    ("04.png", "square", "yellow", "R4"),
    ("05.png", "circle", "blue", "T1"),
    ("06.png", "triangle", "red", "W8"),
]
SAMPLE_VIDEO = "clip.mp4"
SAMPLE_QUESTION = (
    "In clip.mp4, which way does the ball move, and at what time does the background "
    "change color? What are the colors before and after?"
)

ANNOTATE_PROMPT = """Label the image {file}.
Reply with only a JSON object with these fields:
- "shape": the large shape: "circle", "square" or "triangle"
- "color": its color, one word
- "tag": the small code printed in the bottom-right corner (use zoom_image to read it)"""

ANNOTATE_SYSTEM = """You label images for a dataset, with no human to ask.
- The image is attached. If a detail is too small to make out, use zoom_image on it.
- When you're sure, reply with only the JSON object asked for: no other text."""

ASK_SYSTEM = """You answer questions about the image and video files in a folder.
- Look before you answer: view_image for images; for videos, video_info and
  video_overview first, then video_frame or a shorter overview to check details.
- Give times as they're labelled on the frames, like 0:04.5.
- Answer only from what you saw. If you couldn't tell, say so."""


def make_samples(folder: Path) -> Path:
    """Draw the sample images and video into `folder`, if they aren't there yet.

    Args:
        folder: Where to put them. Created if needed.

    Returns:
        The folder.
    """
    from PIL import Image, ImageDraw, ImageFont

    folder.mkdir(parents=True, exist_ok=True)
    colors = {"red": "#d62728", "blue": "#1f5fd6", "green": "#2ca02c", "yellow": "#f2c80f"}
    for file, shape, color, tag in SAMPLE_IMAGES:
        if (folder / file).exists():
            continue
        # Big, like a photo, with the tag small enough that seeing it whole isn't enough.
        picture = Image.new("RGB", (1600, 1200), "#f4f1ea")
        draw = ImageDraw.Draw(picture)
        box = (500, 300, 1100, 900)
        if shape == "circle":
            draw.ellipse(box, fill=colors[color])
        elif shape == "square":
            draw.rectangle(box, fill=colors[color])
        else:
            draw.polygon([(800, 300), (1100, 900), (500, 900)], fill=colors[color])
        draw.text((1520, 1160), tag, fill="#555555", font=ImageFont.load_default(size=16))
        picture.save(folder / file)
    if not (folder / SAMPLE_VIDEO).exists():
        write_sample_video(folder / SAMPLE_VIDEO)
    return folder


def write_sample_video(path: Path, seconds: float = 8.0, fps: int = 15) -> None:
    """Write an 8-second clip: a ball crossing left to right, blue turning green at 5 s.

    Args:
        path: The .mp4 file to write.
        seconds: How long the clip is.
        fps: Frames per second.
    """
    import av
    from PIL import Image, ImageDraw

    width, height = 480, 270
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for index in range(round(seconds * fps)):
            time = index / fps
            picture = Image.new("RGB", (width, height), "#1f5fd6" if time < 5 else "#2ca02c")
            x = 30 + (width - 60) * time / seconds
            ImageDraw.Draw(picture).ellipse((x - 25, 110, x + 25, 160), fill="#d62728")
            frame = av.VideoFrame.from_image(picture)
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


def parse_label(text: str | None) -> dict | None:
    """Find the JSON object in a model's reply.

    Small models wrap JSON in ```json fences or add a sentence, so this takes the
    first {...} it can decode rather than insisting the whole reply is JSON.

    Args:
        text: The reply.

    Returns:
        The object, or None if there isn't one.
    """
    for match in re.finditer(r"\{.*?\}", text or "", re.DOTALL):
        try:
            label = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        if isinstance(label, dict):
            return label
    return None


def score_label(label: dict | None, expected: tuple[str, str, str]) -> int:
    """Count how many of a sample's fields the model got right.

    Args:
        label: The parsed label, or None if the reply had none.
        expected: The sample's shape, color and tag.

    Returns:
        0 to 3. Case and surrounding spaces don't matter.
    """
    if label is None:
        return 0
    return sum(
        str(label.get(field, "")).strip().lower() == answer.lower()
        for field, answer in zip(("shape", "color", "tag"), expected, strict=True)
    )


async def annotate_file(
    backend: ChatBackend, media: MediaTools, file: Path, prompt: str, max_steps: int = 6
) -> tuple[dict | None, RunResult]:
    """Label one image in a conversation of its own.

    The image goes straight into the user message, so the model sees it without
    spending a step; the tools are there to zoom in.

    Args:
        backend: The vision model.
        media: The media tools, bound to the image's folder.
        file: The image.
        prompt: The labelling instructions; "{file}" is replaced by the file's name.
        max_steps: The step budget.

    Returns:
        The parsed label (None if the reply had no JSON object) and the whole run.
    """
    name = file.resolve().relative_to(media.root).as_posix()
    messages = [
        Message.system(ANNOTATE_SYSTEM),
        Message.user(prompt.replace("{file}", name), images=[load_image(file, media.max_side)]),
    ]
    result = await run_agent(backend, media.registry(), messages, max_steps=max_steps)
    return parse_label(result.answer), result


def model_options(command):
    """Add the options every subcommand shares: the engine, model and image settings."""
    options = [
        click.argument("engine", type=click.Choice(["ollama", "vllm"]), default="ollama"),
        click.option("--model", help="Defaults to qwen3:4b for Ollama; vLLM asks the server."),
        click.option(
            "--max-images", default=3, show_default=True,
            help="Images kept in the context; older ones are replaced by a note.",
        ),
        click.option(
            "--max-side", default=1024, show_default=True,
            help="Images are shrunk to at most this many pixels on their longer side.",
        ),
        click.option(
            "--folder", type=click.Path(exists=True, file_okay=False, path_type=Path),
            help="The folder with your files. Default: sample files, made in .lil_media/.",
        ),
    ]  # fmt: skip
    for option in reversed(options):
        command = option(command)
    return command


def make_settings(engine: str, model: str | None, max_images: int, max_side: int) -> dict:
    """Build the backend, and bundle it with the settings the runs need.

    Args:
        engine: "ollama" or "vllm".
        model: The model name, or None for the engine's default.
        max_images: How many images to keep in the context.
        max_side: The longest side, in pixels, of any image the model sees.

    Returns:
        {"backend", "engine", "max_side"}.
    """
    backend_cls = OllamaBackend if engine == "ollama" else VLLMBackend
    backend = backend_cls(max_images=max_images, **({"model": model} if model else {}))
    return {"backend": backend, "engine": engine, "max_side": max_side}


@click.group()
def main() -> None:
    """Label images, or ask about images and video, with a local vision model."""


@main.command()
@model_options
@click.option("--prompt", help="Labelling instructions; {file} becomes the file's name.")
@click.option("--max-steps", default=6, show_default=True)
def annotate(
    engine: str,
    model: str | None,
    max_images: int,
    max_side: int,
    folder: Path | None,
    prompt: str | None,
    max_steps: int,
) -> None:
    """Label every image in a folder and save the labels to labels.jsonl there.

    \f
    Args:
        engine: "ollama" or "vllm".
        model: The model name, or None for the engine's default.
        max_images: How many images to keep in the context.
        max_side: The longest side, in pixels, of any image the model sees.
        folder: The images' folder, or None for the samples.
        prompt: The labelling instructions, or None for the samples' prompt.
        max_steps: The step budget per image.
    """
    if folder is None:
        folder = make_samples(SAMPLES)
    elif prompt is None:
        raise click.UsageError("With your own --folder, say what to label with --prompt.")
    settings = make_settings(engine, model, max_images, max_side)
    asyncio.run(run_annotate(settings, folder, prompt or ANNOTATE_PROMPT, max_steps))


async def run_annotate(settings: dict, folder: Path, prompt: str, max_steps: int) -> None:
    """Label the folder's images one by one, and score them if they're the samples.

    Args:
        settings: What `make_settings` returned.
        folder: The images' folder.
        prompt: The labelling instructions.
        max_steps: The step budget per image.
    """
    media = MediaTools(folder, max_side=settings["max_side"])
    files = sorted(file for file in folder.iterdir() if file.suffix.lower() in IMAGE_SUFFIXES)
    truth = {file: (shape, color, tag) for file, shape, color, tag in SAMPLE_IMAGES}
    scored = correct = 0
    with (folder / "labels.jsonl").open("w", encoding="utf-8") as out:
        for file in files:
            label, result = await guard(settings, annotate_file(
                settings["backend"], media, file, prompt, max_steps
            ))  # fmt: skip
            zooms = sum(
                tool_call.name == "zoom_image"
                for message in result.messages
                for tool_call in message.tool_calls
            )
            record = {"file": file.name, "label": label, "raw": result.answer,
                      "steps": result.steps, "zooms": zooms}  # fmt: skip
            out.write(json.dumps(record) + "\n")
            verdict = ""
            if folder.resolve() == SAMPLES and file.name in truth:
                hits = score_label(label, truth[file.name])
                scored, correct = scored + 3, correct + hits
                verdict = f"  {hits}/3 right"
            click.echo(f"{file.name}: {json.dumps(label) if label else result.answer!r}"
                       f"  (steps={result.steps}, zooms={zooms}){verdict}")  # fmt: skip
    click.echo(f"\nLabels saved to {folder / 'labels.jsonl'}")
    if scored:
        click.echo(f"Score: {correct}/{scored} fields right ({correct / scored:.0%})")


@main.command()
@model_options
@click.option("--question", "-q", help="What to ask. Default: a question about the sample clip.")
@click.option("--max-steps", default=10, show_default=True)
def ask(
    engine: str,
    model: str | None,
    max_images: int,
    max_side: int,
    folder: Path | None,
    question: str | None,
    max_steps: int,
) -> None:
    """Ask a question about the images and videos in a folder.

    \f
    Args:
        engine: "ollama" or "vllm".
        model: The model name, or None for the engine's default.
        max_images: How many images to keep in the context.
        max_side: The longest side, in pixels, of any image the model sees.
        folder: The files' folder, or None for the samples.
        question: What to ask, or None for the question about the sample clip.
        max_steps: The step budget.
    """
    folder = folder or make_samples(SAMPLES)
    settings = make_settings(engine, model, max_images, max_side)
    asyncio.run(run_ask(settings, folder, question or SAMPLE_QUESTION, max_steps))


async def run_ask(settings: dict, folder: Path, question: str, max_steps: int) -> None:
    """Run the agent on the question, and print every call and result.

    Args:
        settings: What `make_settings` returned.
        folder: The files' folder.
        question: What to ask.
        max_steps: The step budget.
    """
    media = MediaTools(folder, max_side=settings["max_side"])
    files = ", ".join(sorted(path.name for path in folder.iterdir() if path.is_file()))
    messages = [
        Message.system(ASK_SYSTEM),
        Message.user(f"Files in the folder: {files}\n\n{question}"),
    ]
    result = await guard(settings, run_agent(
        settings["backend"], media.registry(), messages, max_steps=max_steps
    ))  # fmt: skip
    for message in result.messages[2:]:
        for tool_call in message.tool_calls:
            click.secho(f"[call] {tool_call.name}({tool_call.raw_arguments})", fg="cyan")
        if message.role == "tool":
            seen = f" + {len(message.images)} image" if message.images else ""
            click.secho(f"[tool] {message.content}{seen}", dim=True)
        elif message.content:
            click.echo(f"[{message.role}] {message.content}")
    click.echo(f"\nstop_reason={result.stop_reason} steps={result.steps}")


async def guard(settings: dict, work):
    """Await `work`, turning the usual server problems into readable errors.

    Args:
        settings: What `make_settings` returned.
        work: The coroutine to run.

    Returns:
        What `work` returns.

    Raises:
        click.ClickException: If the server is down or the model can't take images.
    """
    try:
        return await work
    except APIConnectionError:
        raise click.ClickException(
            f"Can't reach {settings['backend'].client.base_url}. "
            f"Is the {settings['engine']} server running?"
        ) from None
    except BadRequestError as e:
        raise click.ClickException(
            f"The server refused the request: {e.message}\n"
            "If it's about images, this model probably can't see. Check with "
            "`ollama show <model>` that it lists vision."
        ) from None


if __name__ == "__main__":
    main()
