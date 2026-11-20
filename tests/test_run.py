"""Tests for the pipeline command dispatcher."""

import pytest

import run


def _recorder(calls: list[str]) -> dict[str, object]:
    return {name: (lambda argv, name=name: calls.append(name)) for name in run.ORDER}


def test_selected_steps_run_in_pipeline_order(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(run, "STEPS", _recorder(calls))

    run.main(["costs", "clean"])

    assert calls == ["clean", "costs"]


def test_all_runs_every_step(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(run, "STEPS", _recorder(calls))

    run.main(["all"])

    assert calls == list(run.ORDER)


def test_no_arguments_only_shows_help(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        run,
        "STEPS",
        {name: (lambda argv: (_ for _ in ()).throw(AssertionError("must not run"))) for name in run.ORDER},
    )

    run.main([])

    assert "usage:" in capsys.readouterr().out


def test_unknown_step_is_rejected() -> None:
    with pytest.raises(SystemExit):
        run.main(["stage7"])
