"""Tests for the explicit completed-stage command dispatcher."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run


def test_stage2_dispatches_to_stage2_runner(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(run, "stage2_main", lambda argv: calls.append(argv))

    run.main(["stage2"])

    assert calls == [[]]


def test_stage3_dispatches_to_stage3_runner(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(run, "stage3_main", lambda argv: calls.append(argv))

    run.main(["stage3"])

    assert calls == [[]]


def test_no_stage_only_shows_help(capsys, monkeypatch) -> None:
    monkeypatch.setattr(run, "stage2_main", lambda argv: (_ for _ in ()).throw(AssertionError("must not run")))
    monkeypatch.setattr(run, "stage3_main", lambda argv: (_ for _ in ()).throw(AssertionError("must not run")))

    run.main([])

    assert "usage:" in capsys.readouterr().out
