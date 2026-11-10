# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

"""The ``ket`` command.

``ket skills install`` adds the Ket skill to a project, so AI coding agents
(Claude Code, Codex, Cursor, Copilot, ...) know how to write Ket code:

- ``.claude/skills/ket/SKILL.md`` for Claude Code;
- ``.agents/skills/ket/SKILL.md`` for agents that read ``.agents/skills``;
- a short section in ``AGENTS.md`` that points to the skill, for agents that
  only read ``AGENTS.md``.

Running it again updates the files to the installed Ket version.
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata, resources
from pathlib import Path

__all__ = ["main"]

TARGETS = {
    "claude": Path(".claude/skills/ket/SKILL.md"),
    "agents": Path(".agents/skills/ket/SKILL.md"),
}

_BEGIN = "<!-- ket skill: begin -->"
_END = "<!-- ket skill: end -->"


def _agents_section(skill: Path) -> str:
    return f"""{_BEGIN}
## Ket

This project uses Ket (`ket-lang`) for quantum programming. Before writing or
changing Ket code, read `{skill.as_posix()}`.
{_END}
"""


def skill_text() -> str:
    """The Ket skill (``SKILL.md``) of the installed version."""
    return (
        resources.files("ket")
        .joinpath("skills/ket/SKILL.md")
        .read_text(encoding="utf-8")
    )


def _write(path: Path, text: str, root: Path) -> str:
    """Write `text` to `path`; return what happened, for the report."""
    old = path.read_text(encoding="utf-8") if path.exists() else None
    if old == text:
        return f"up to date  {path.relative_to(root)}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return f"{'updated' if old is not None else 'created':<11} {path.relative_to(root)}"


def _agents_md(old: str, skill: Path) -> str:
    """`old` with the Ket section, pointing to `skill`, added or replaced."""
    section = _agents_section(skill)
    pattern = re.compile(re.escape(_BEGIN) + ".*?" + re.escape(_END) + r"\n?", re.S)
    if pattern.search(old):
        return pattern.sub(lambda _: section, old)
    if not old:
        return "# Agent instructions\n\n" + section
    return old.rstrip("\n") + "\n\n" + section


def install(root: Path, targets: list[str], agents_md: bool = True) -> list[str]:
    """Install the Ket skill in the project at `root`.

    Args:
        root: The project directory.
        targets: Keys of :data:`TARGETS` to install.
        agents_md: Also add the Ket section to ``AGENTS.md``.

    Returns:
        One line per file, saying whether it was created, updated, or
        already up to date.
    """
    text = skill_text()
    report = [_write(root / TARGETS[target], text, root) for target in targets]
    if agents_md and targets:
        # Point to the generic copy when installed, else to the first one.
        skill = TARGETS["agents" if "agents" in targets else targets[0]]
        path = root / "AGENTS.md"
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        report.append(_write(path, _agents_md(old, skill), root))
    return report


def _cmd_install(args) -> int:
    root = Path(args.path).resolve()
    if not root.is_dir():
        print(f"ket: {root} is not a directory", file=sys.stderr)
        return 1
    targets = list(TARGETS) if args.only is None else [args.only]
    for line in install(root, targets, agents_md=not args.no_agents_md):
        print(line)
    return 0


def _cmd_show(_args) -> int:
    print(skill_text(), end="")
    return 0


def _parser() -> argparse.ArgumentParser:
    try:
        version = metadata.version("ket-lang")
    except metadata.PackageNotFoundError:  # running from a source checkout
        version = "unknown"

    parser = argparse.ArgumentParser(prog="ket", description="Ket quantum programming.")
    parser.add_argument("--version", action="version", version=f"ket {version}")
    commands = parser.add_subparsers(dest="command", required=True)

    skills = commands.add_parser(
        "skills", help="help AI coding agents write Ket code"
    ).add_subparsers(dest="action", required=True)

    install_cmd = skills.add_parser(
        "install",
        help="add the Ket skill to a project",
        description="Add the Ket skill to a project: "
        + ", ".join(str(p) for p in TARGETS.values())
        + ", and a section in AGENTS.md. Run it again to update.",
    )
    install_cmd.add_argument(
        "path", nargs="?", default=".", help="project directory (default: current)"
    )
    install_cmd.add_argument(
        "--only", choices=list(TARGETS), help="install for one kind of agent only"
    )
    install_cmd.add_argument(
        "--no-agents-md", action="store_true", help="do not change AGENTS.md"
    )
    install_cmd.set_defaults(run=_cmd_install)

    show = skills.add_parser("show", help="print the Ket skill")
    show.set_defaults(run=_cmd_show)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``ket`` command."""
    args = _parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
