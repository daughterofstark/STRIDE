"""Local Markdown links in the public README must resolve in the repository."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


def test_readme_local_markdown_links_exist() -> None:
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^]]+\]\(([^)]+)\)", text)
    missing = []
    for target in targets:
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path_text = unquote(target.split("#", 1)[0])
        if path_text and not (repo / path_text).exists():
            missing.append(target)
    assert missing == []
