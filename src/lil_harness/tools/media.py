"""Image and video tools, for models that can see.

    media = MediaTools("path/to/files")
    registry = media.registry()                     # just these tools, read-only
    media.register(CodingTools(root).registry(...)) # or add them to other tools

They need Pillow and PyAV: `uv sync --extra media`.

The model can't open a file itself, so every tool here turns a file into an `Image`
the model sees in its next turn (see types.py and openai_compat.py for how it gets
there). Two things shape them:

- **Pixels cost tokens.** A vision model cuts an image into small patches and each
  patch becomes tokens, so a 12-megapixel photo would fill a small model's whole
  context. Every image is shrunk so its longer side is at most `max_side` pixels
  (1024 by default, around a thousand tokens). For detail, `zoom_image` crops a
  region and enlarges it instead of sending the whole image bigger.
- **Video is a stack of images.** Ollama's OpenAI API takes images but not video, so
  the tools pick frames: `video_overview` puts evenly spaced frames into one grid,
  each labelled with its time, and `video_frame` shows one moment in full. That's
  the same thing models that "watch" video do inside: sample a few frames per second.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import TYPE_CHECKING

from lil_harness.permissions import PermissionPolicy
from lil_harness.tools.coding import resolve_in
from lil_harness.tools.registry import ToolError, ToolRegistry
from lil_harness.types import Image, ToolOutput

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

MAX_FRAMES = 16
"""The most frames one overview may hold. More would make each too small to read."""


def _pil():
    """Import Pillow, with a hint on how to install it.

    Returns:
        The PIL.Image module.

    Raises:
        ImportError: If Pillow isn't installed.
    """
    try:
        from PIL import Image as PILImageModule
    except ImportError:
        raise ImportError("The media tools need Pillow and PyAV: uv sync --extra media") from None
    return PILImageModule


def _av():
    """Import PyAV, with a hint on how to install it.

    Returns:
        The av module.

    Raises:
        ImportError: If PyAV isn't installed.
    """
    try:
        import av
    except ImportError:
        raise ImportError("The media tools need Pillow and PyAV: uv sync --extra media") from None
    return av


def encode(picture: PILImage, max_side: int = 1024, *, enlarge: bool = False) -> Image:
    """Scale a picture to fit `max_side` and encode it as JPEG for the model.

    Args:
        picture: A Pillow image.
        max_side: The most pixels its longer side may have.
        enlarge: Also scale small pictures up to `max_side`, for zoomed-in crops.

    Returns:
        The encoded image.
    """
    scale = max_side / max(picture.size)
    if scale < 1 or (enlarge and scale > 1):
        size = (max(1, round(picture.width * scale)), max(1, round(picture.height * scale)))
        picture = picture.resize(size, _pil().Resampling.LANCZOS)
    buffer = io.BytesIO()
    # JPEG has no transparency, so flatten anything else to RGB first.
    picture.convert("RGB").save(buffer, format="JPEG", quality=90)
    return Image(buffer.getvalue(), "image/jpeg")


def load_image(path: str | Path, max_side: int = 1024) -> Image:
    """Load an image file, shrunk for the model. Use this for images in user messages.

    Args:
        path: Any format Pillow reads.
        max_side: The most pixels its longer side may have.

    Returns:
        The image as JPEG, turned upright if the camera stored it rotated.
    """
    from PIL import ImageOps

    with _pil().open(path) as picture:
        return encode(ImageOps.exif_transpose(picture), max_side)


def clock(seconds: float) -> str:
    """Format a time in a video, like 0:07.5 or 1:02:03.0.

    Args:
        seconds: The time in seconds.

    Returns:
        Minutes and seconds to a tenth, with hours in front if there are any.
    """
    tenths = round(seconds * 10)
    hours, rest = divmod(tenths, 36000)
    minutes, rest = divmod(rest, 600)
    text = f"{minutes}:{rest // 10:02d}.{rest % 10}"
    return f"{hours}:{minutes:02d}:{rest // 10:02d}.{rest % 10}" if hours else text


class MediaTools:
    """The image and video tools, bound to one workspace folder."""

    def __init__(self, root: str | Path, *, max_side: int = 1024) -> None:
        """Set up the tools for one folder.

        Args:
            root: The workspace folder. Every path the model gives is relative to it.
            max_side: The most pixels the longer side of any image the model sees may
                have. Lower it if images crowd the context out; raise it if your
                model and context have room and details are being missed.

        Raises:
            ValueError: If `root` isn't an existing folder.
            ImportError: If Pillow or PyAV isn't installed.
        """
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Workspace {self.root} is not a folder")
        _pil(), _av()
        self.max_side = max_side

    def register(self, registry: ToolRegistry) -> ToolRegistry:
        """Add the media tools to a registry. They only read, so they're risk "read".

        Args:
            registry: The registry to add them to.

        Returns:
            The same registry, for chaining.
        """
        for tool in (
            self.image_info,
            self.view_image,
            self.zoom_image,
            self.video_info,
            self.video_overview,
            self.video_frame,
        ):
            registry.tool(tool, risk="read")
        return registry

    def registry(self, policy: PermissionPolicy | None = None) -> ToolRegistry:
        """A new registry with just the media tools.

        Args:
            policy: Decides which calls run. Default: the "readonly" preset.

        Returns:
            The registry.
        """
        return self.register(ToolRegistry(policy or PermissionPolicy.preset("readonly")))

    # ---- helpers -------------------------------------------------------------

    def _file(self, path: str) -> Path:
        """Resolve `path` in the workspace and check it's a file.

        Raises:
            ToolError: If it's outside the workspace, missing or a folder.
        """
        target = resolve_in(self.root, path)
        if not target.is_file():
            raise ToolError(f"{path!r} isn't a file. Use list_files to find the right path.")
        return target

    def _open_image(self, path: str) -> tuple[Path, PILImage]:
        """Open an image file, upright.

        Returns:
            The resolved path and the loaded picture.

        Raises:
            ToolError: If the file isn't an image Pillow can read.
        """
        from PIL import ImageOps, UnidentifiedImageError

        target = self._file(path)
        try:
            with _pil().open(target) as picture:
                picture.load()
                return target, ImageOps.exif_transpose(picture)
        except UnidentifiedImageError:
            raise ToolError(f"{path!r} isn't an image file this tool can read.") from None

    def _rel(self, path: Path) -> str:
        """Returns: `path` relative to the workspace, with forward slashes."""
        return path.relative_to(self.root).as_posix()

    # ---- image tools ---------------------------------------------------------

    def image_info(self, path: str) -> str:
        """Get an image file's size and format, without looking at it.

        Args:
            path: The image file, relative to the workspace.

        Returns:
            One line: path, pixel size, format, color mode and file size.
        """
        target, picture = self._open_image(path)
        kilobytes = target.stat().st_size / 1024
        return (
            f"{self._rel(target)}: {picture.width}x{picture.height} pixels, "
            f"{picture.format or target.suffix.lstrip('.').upper()}, {picture.mode}, "
            f"{kilobytes:,.0f} KB"
        )

    def view_image(self, path: str) -> ToolOutput:
        """Look at an image file. Large images are shrunk; use zoom_image for details.

        Args:
            path: The image file, relative to the workspace, e.g. "photos/cat.jpg".

        Returns:
            The image, and a line saying its size and the size it's shown at.
        """
        target, picture = self._open_image(path)
        image = encode(picture, self.max_side)
        shown = _size_of(image)
        return ToolOutput(
            f"{self._rel(target)}: {picture.width}x{picture.height} pixels, "
            f"shown at {shown[0]}x{shown[1]}.",
            [image],
        )

    def zoom_image(
        self, path: str, left: float, top: float, right: float, bottom: float
    ) -> ToolOutput:
        """Look closely at part of an image: crop a region and show it enlarged.

        The region is given in fractions of the image, so (0, 0, 0.5, 0.5) is the
        top-left quarter and (0.25, 0.25, 0.75, 0.75) the middle.

        Args:
            path: The image file, relative to the workspace.
            left: The region's left edge: 0 is the image's left side, 1 its right.
            top: The region's top edge: 0 is the image's top, 1 its bottom.
            right: The region's right edge, more than left.
            bottom: The region's bottom edge, more than top.

        Returns:
            The region, enlarged, and a line saying which pixels it covers.
        """
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise ToolError(
                "The region must have 0 <= left < right <= 1 and 0 <= top < bottom <= 1, "
                "as fractions of the image's width and height."
            )
        target, picture = self._open_image(path)
        box = (
            round(left * picture.width),
            round(top * picture.height),
            max(round(right * picture.width), round(left * picture.width) + 1),
            max(round(bottom * picture.height), round(top * picture.height) + 1),
        )
        image = encode(picture.crop(box), self.max_side, enlarge=True)
        shown = _size_of(image)
        return ToolOutput(
            f"{self._rel(target)}: pixels x {box[0]}-{box[2]}, y {box[1]}-{box[3]} of "
            f"{picture.width}x{picture.height}, shown at {shown[0]}x{shown[1]}.",
            [image],
        )

    # ---- video tools ---------------------------------------------------------

    def video_info(self, path: str) -> str:
        """Get a video file's length, frame size, frame rate and codec.

        Args:
            path: The video file, relative to the workspace.

        Returns:
            One line describing the video.
        """
        target = self._file(path)
        with self._open_video(target, path) as container:
            stream = container.streams.video[0]
            duration = _duration(container, stream)
            fps = float(stream.average_rate or 0)
            frames = stream.frames or round(duration * fps)
            audio = "yes" if container.streams.audio else "no"
            return (
                f"{self._rel(target)}: {clock(duration)} ({duration:.1f} s), "
                f"{stream.width}x{stream.height}, {fps:.3g} fps, about {frames} frames, "
                f"codec {stream.codec_context.name}, audio: {audio}"
            )

    def video_overview(
        self, path: str, frames: int = 9, start: float = 0.0, end: float | None = None
    ) -> ToolOutput:
        """See a video, or a stretch of it, at a glance: evenly spaced frames in one grid.

        Each frame is labelled with its number and time. Look at the overview first,
        then use video_frame on the moments that matter, or a shorter overview of a
        stretch with start and end.

        Args:
            path: The video file, relative to the workspace.
            frames: How many frames, 1 to 16. More shows more moments, each smaller.
            start: Where the stretch starts, in seconds from the beginning.
            end: Where it ends, in seconds. Default: the end of the video.

        Returns:
            The grid, and a line listing each frame's time.
        """
        if not 1 <= frames <= MAX_FRAMES:
            raise ToolError(f"frames must be between 1 and {MAX_FRAMES}.")
        target = self._file(path)
        with self._open_video(target, path) as container:
            duration = _duration(container, container.streams.video[0])
            stop = duration if end is None else min(end, duration)
            if not 0 <= start < stop:
                raise ToolError(f"Need 0 <= start < end; the video is {duration:.1f} seconds long.")
            # The middle of each of `frames` equal slices, which skips the black first
            # frame many videos have, and never asks for the very end.
            step = (stop - start) / frames
            times = [start + (index + 0.5) * step for index in range(frames)]
            grabbed = _grab_frames(container, times)
        sheet = _contact_sheet(grabbed, self.max_side)
        listing = ", ".join(
            f"{number} at {clock(time)}" for number, (time, _) in enumerate(grabbed, start=1)
        )
        return ToolOutput(
            f"{self._rel(target)}: {len(grabbed)} frames from {clock(start)} to "
            f"{clock(stop)}, left to right, top to bottom: {listing}.",
            [encode(sheet, self.max_side)],
        )

    def video_frame(self, path: str, time: float) -> ToolOutput:
        """Look at one frame of a video in full detail.

        Args:
            path: The video file, relative to the workspace.
            time: When, in seconds from the beginning, e.g. 12.5.

        Returns:
            The frame closest to that time, and its exact time.
        """
        target = self._file(path)
        with self._open_video(target, path) as container:
            duration = _duration(container, container.streams.video[0])
            if not 0 <= time <= duration:
                raise ToolError(f"time must be between 0 and {duration:.1f} seconds.")
            [(frame_time, picture)] = _grab_frames(container, [time])
        image = encode(picture, self.max_side)
        return ToolOutput(f"{self._rel(target)} at {clock(frame_time)}.", [image])

    def _open_video(self, target: Path, path: str):
        """Open a video file with PyAV.

        Returns:
            The open container; use it in a `with` block.

        Raises:
            ToolError: If the file isn't a video PyAV can read.
        """
        av = _av()
        try:
            container = av.open(str(target))
        except av.FFmpegError:
            raise ToolError(f"{path!r} isn't a video file this tool can read.") from None
        if not container.streams.video:
            container.close()
            raise ToolError(f"{path!r} has no video stream.")
        return container


def _size_of(image: Image) -> tuple[int, int]:
    """Returns: an encoded image's width and height."""
    with _pil().open(io.BytesIO(image.data)) as picture:
        return picture.size


def _duration(container, stream) -> float:
    """A video's length in seconds, from the stream if it says, else the container."""
    if stream.duration and stream.time_base:
        return float(stream.duration * stream.time_base)
    if container.duration:
        return container.duration / 1_000_000  # The container counts in microseconds.
    raise ToolError("Can't tell how long this video is.")


def _grab_frames(container, times: list[float]) -> list[tuple[float, PILImage]]:
    """Decode the frame shown at each of `times`.

    Video is compressed as keyframes plus changes, so you can't jump straight to any
    frame. For each time this seeks to the keyframe before it, then decodes forward to
    the frame on screen at that moment.

    Args:
        container: The open video.
        times: Seconds from the beginning, in any order.

    Returns:
        One (frame's actual time, picture) per requested time, in the same order.

    Raises:
        ToolError: If no frame could be decoded.
    """
    stream = container.streams.video[0]
    offset = float(stream.start_time * stream.time_base) if stream.start_time else 0.0
    fps = float(stream.average_rate or 30)
    grabbed: list[tuple[float, PILImage]] = []
    for time in times:
        container.seek(round((time + offset) / stream.time_base), stream=stream)
        chosen = None
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            chosen = frame
            # A frame stays on screen until the next one: within half a frame counts.
            if frame.time - offset >= time - 0.5 / fps:
                break
        if chosen is None:
            raise ToolError(f"Couldn't decode a frame at {clock(time)}.")
        grabbed.append((chosen.time - offset, chosen.to_image()))
    return grabbed


def _contact_sheet(grabbed: list[tuple[float, PILImage]], width: int) -> PILImage:
    """Lay frames out in a grid, each with its number and time in the corner.

    Args:
        grabbed: (time, picture) pairs, in order.
        width: The sheet's width in pixels.

    Returns:
        The sheet: as close to square as the frame count allows.
    """
    from PIL import ImageDraw, ImageFont

    columns = math.ceil(math.sqrt(len(grabbed)))
    rows = math.ceil(len(grabbed) / columns)
    first = grabbed[0][1]
    tile_width = width // columns
    tile_height = round(tile_width * first.height / first.width)
    sheet = _pil().new("RGB", (tile_width * columns, tile_height * rows), "black")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=max(12, tile_width // 14))
    for index, (time, picture) in enumerate(grabbed):
        x, y = (index % columns) * tile_width, (index // columns) * tile_height
        sheet.paste(picture.resize((tile_width - 2, tile_height - 2)), (x + 1, y + 1))
        label = f"{index + 1}  {clock(time)}"
        box = draw.textbbox((x + 4, y + 4), label, font=font)
        draw.rectangle((box[0] - 3, box[1] - 3, box[2] + 3, box[3] + 3), fill="black")
        draw.text((x + 4, y + 4), label, fill="white", font=font)
    return sheet
