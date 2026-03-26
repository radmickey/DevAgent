"""Language detector: builds a language map from repository file extensions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "csharp",
    ".swift": "swift",
    ".php": "php",
}


@dataclass
class RepoLanguageMap:
    """Maps languages to file counts and paths in a repository."""

    languages: dict[str, int] = field(default_factory=dict)
    primary_language: str = "unknown"


def build_language_map(file_paths: list[str]) -> RepoLanguageMap:
    """Analyze file extensions to determine repository languages."""
    counter: Counter[str] = Counter()
    for fp in file_paths:
        ext = Path(fp).suffix.lower()
        lang = EXTENSION_MAP.get(ext)
        if lang:
            counter[lang] += 1

    languages = dict(counter.most_common())
    primary = counter.most_common(1)[0][0] if counter else "unknown"
    return RepoLanguageMap(languages=languages, primary_language=primary)
