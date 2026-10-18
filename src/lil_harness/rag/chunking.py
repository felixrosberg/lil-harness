"""Split files into chunks small enough to embed and to show the model.

A chunk is a run of whole lines, about `max_chars` long. Neighbouring chunks share a
few lines (`overlap_lines`), so a sentence or function that crosses a boundary still
appears whole in one of them. Chunks prefer to end just before a Markdown heading or a
top-level `def`/`class`, so a chunk tends to hold one section or one function.

Every chunk keeps its file and line range. That's what lets the model cite its
sources, and read more of the file with `read_file` when a chunk isn't enough.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import TypeAlias

from lil_harness.tools.coding import IGNORED_DIRS

ChunkId: TypeAlias = str
"""Where a chunk came from, as "path:first-last", e.g. "README.md:120-140"."""

DEFAULT_PATTERNS = ("**/*.md", "**/*.py", "**/*.txt", "**/*.yaml", "**/*.toml")

# Lines that start a new section: Markdown headings, and top-level Python definitions.
_BOUNDARY = re.compile(r"^(#{1,6} |def |async def |class |@)")


@dataclass
class Chunk:
    """A run of lines from one file.

    Attributes:
        source: The file's path relative to the corpus folder, with forward slashes.
        start_line: The first line, counting from 1.
        end_line: The last line, inclusive.
        text: The lines themselves.
    """

    source: str
    start_line: int
    end_line: int
    text: str

    @property
    def id(self) -> ChunkId:
        """Returns: the chunk's citation, like "README.md:120-140"."""
        return f"{self.source}:{self.start_line}-{self.end_line}"


def chunk_text(
    source: str, text: str, *, max_chars: int = 1600, overlap_lines: int = 2
) -> list[Chunk]:
    """Split one file's text into overlapping chunks of whole lines.

    Args:
        source: The file's path, stored on each chunk.
        text: The file's content.
        max_chars: Roughly the most characters per chunk. 1600 characters is about
            400 tokens: small enough to be specific, big enough to hold a function.
        overlap_lines: How many lines each chunk repeats from the end of the one before.

    Returns:
        The chunks in file order. Chunks that would be only whitespace are dropped.
    """
    lines = text.splitlines()
    chunks: list[Chunk] = []
    start = 0
    while start < len(lines):
        end = start
        size = 0
        while end < len(lines):
            line_size = len(lines[end]) + 1
            if end > start and size + line_size > max_chars:
                break
            # A new section is a good place to stop, once the chunk has some substance.
            if end > start and size > max_chars // 3 and _BOUNDARY.match(lines[end]):
                break
            size += line_size
            end += 1
        body = "\n".join(lines[start:end])
        if body.strip():
            chunks.append(Chunk(source, start + 1, end, body))
        if end >= len(lines):
            break
        # Step back a little for the overlap, but always move forward.
        start = max(end - overlap_lines, start + 1)
    return chunks


def corpus_files(
    root: Path, patterns: tuple[str, ...] = DEFAULT_PATTERNS, exclude: tuple[str, ...] = ()
) -> list[Path]:
    """Find the files to index under `root`.

    Args:
        root: The folder to index.
        patterns: Glob patterns, relative to `root`.
        exclude: Patterns for files to leave out, matched against the path relative to
            `root`, e.g. "evals/retrieval/*".

    Returns:
        Matching files, sorted, skipping IGNORED_DIRS (.git, .venv, ...), hidden folders
        such as the index itself, and files over 1 MB.
    """
    found: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            relative = path.relative_to(root)
            parts = relative.parts
            if (
                path.is_file()
                and not any(fnmatch(relative.as_posix(), skip) for skip in exclude)
                and not IGNORED_DIRS.intersection(parts)
                and not any(part.startswith(".") for part in parts[:-1])
                and path.stat().st_size <= 1_000_000
            ):
                found.add(path)
    return sorted(found)


def load_corpus(
    root: Path,
    patterns: tuple[str, ...] = DEFAULT_PATTERNS,
    exclude: tuple[str, ...] = (),
    **chunk_options: int,
) -> list[Chunk]:
    """Chunk every matching file under `root`.

    Args:
        root: The folder to index.
        patterns: Glob patterns, relative to `root`.
        exclude: Patterns for files to leave out, as in `corpus_files`.
        **chunk_options: Passed to `chunk_text`, e.g. max_chars.

    Returns:
        All chunks, file by file. Files that aren't UTF-8 text are skipped.
    """
    chunks: list[Chunk] = []
    for path in corpus_files(root, patterns, exclude):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        chunks += chunk_text(path.relative_to(root).as_posix(), text, **chunk_options)
    return chunks
