# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

"""The `ket` command."""

import pytest

from ket.cli import main, skill_text


def test_skill_has_front_matter():
    text = skill_text()
    assert text.startswith("---\nname: ket\ndescription: ")


def test_install_creates_and_updates(tmp_path, capsys):
    assert main(["skills", "install", str(tmp_path)]) == 0
    for path in [".claude/skills/ket/SKILL.md", ".agents/skills/ket/SKILL.md"]:
        assert (tmp_path / path).read_text() == skill_text()
    assert ".agents/skills/ket/SKILL.md" in (tmp_path / "AGENTS.md").read_text()
    assert capsys.readouterr().out.count("created") == 3

    main(["skills", "install", str(tmp_path)])
    assert capsys.readouterr().out.count("up to date") == 3

    (tmp_path / ".claude/skills/ket/SKILL.md").write_text("old")
    main(["skills", "install", str(tmp_path)])
    assert "updated     .claude" in capsys.readouterr().out


def test_install_keeps_existing_agents_md(tmp_path):
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# My project\n\nUse ruff.\n")
    main(["skills", "install", str(tmp_path), "--only", "claude"])
    main(["skills", "install", str(tmp_path), "--only", "claude"])
    text = agents.read_text()
    assert text.startswith("# My project\n\nUse ruff.\n\n")
    assert text.count("ket skill: begin") == 1
    assert ".claude/skills/ket/SKILL.md" in text
    assert not (tmp_path / ".agents").exists()


def test_install_without_agents_md(tmp_path):
    main(["skills", "install", str(tmp_path), "--no-agents-md"])
    assert not (tmp_path / "AGENTS.md").exists()


def test_install_into_missing_directory(tmp_path):
    assert main(["skills", "install", str(tmp_path / "missing")]) == 1


def test_show(capsys):
    main(["skills", "show"])
    assert capsys.readouterr().out == skill_text()


def test_requires_a_command():
    with pytest.raises(SystemExit):
        main([])
