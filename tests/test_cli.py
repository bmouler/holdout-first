"""The demo command is the smoke test: it must reproduce all three documented outcomes."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from holdout_first import cli
from holdout_first.causality import LookaheadError
from holdout_first.cli import build_parser, main, run_demo


def test_typed_package_marker_is_shipped() -> None:
    marker = Path(__file__).parents[1] / "src" / "holdout_first" / "py.typed"
    assert marker.is_file()


def test_demo_exits_zero_on_the_documented_seed(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo", "--seed", "11"]) == 0
    out = capsys.readouterr().out
    assert "verdict: SURVIVED" in out
    assert "verdict: REJECTED" in out
    assert "LookaheadError raised" in out


def test_demo_shows_the_honest_rule_surviving_and_the_overfitted_one_rejected() -> None:
    honest, overfitted, caught = run_demo(11)
    assert honest.survived is True
    assert overfitted.survived is False
    assert isinstance(caught, LookaheadError)


def test_demo_names_the_rules_the_overfitted_strategy_broke() -> None:
    _, overfitted, _ = run_demo(11)
    failed = {rule.name for rule in overfitted.failed_rules}
    assert "parameter_budget" in failed
    assert "sharpe_retention" in failed


def test_demo_is_reproducible_for_a_given_seed() -> None:
    first, _, _ = run_demo(11)
    second, _, _ = run_demo(11)
    assert first.to_dict() == second.to_dict()


def test_demo_output_names_the_first_divergent_bar(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["demo", "--seed", "11"])
    out = capsys.readouterr().out
    _, _, caught = run_demo(11)
    assert caught is not None
    assert f"first divergent bar: {caught.index}" in out


def test_json_output_is_valid_and_complete(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo", "--seed", "11", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["seed"] == 11
    assert payload["demo_succeeded"] is True
    assert payload["honest"]["survived"] is True
    assert payload["overfitted"]["survived"] is False
    assert payload["peeking"]["error"] == "LookaheadError"
    assert payload["peeking"]["index"] >= 0


def test_json_output_reports_zero_fees_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["demo", "--seed", "11", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["honest"]["settings"]["fees"] == 0.0
    assert payload["honest"]["settings"]["train_fraction"] == 0.30


def test_demo_default_seed_is_eleven() -> None:
    args = build_parser().parse_args(["demo"])
    assert args.seed == 11
    assert args.json is False


def test_a_subcommand_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_unknown_subcommand_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["backtest"])


def test_run_demo_rejects_a_non_integer_seed() -> None:
    with pytest.raises(TypeError, match="seed must be an int"):
        run_demo("11")  # type: ignore[arg-type]


def test_demo_failure_returns_one_and_explains_missing_lookahead_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    honest, overfitted, _ = run_demo(11)
    monkeypatch.setattr(cli, "run_demo", lambda seed: (honest, overfitted, None))

    assert main(["demo", "--seed", "11"]) == 1
    output = capsys.readouterr().out
    assert "no LookaheadError was raised" in output
    assert "outcome: the demo did not behave as documented." in output


def test_cli_module_execution_exits_with_demo_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delitem(sys.modules, "holdout_first.cli")
    monkeypatch.setattr(sys, "argv", ["holdout-first", "demo", "--seed", "11"])
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("holdout_first.cli", run_name="__main__")
    assert caught.value.code == 0
    assert "outcome: the parsimonious rule survived" in capsys.readouterr().out


def test_installed_cli_runs_seed_eleven_from_outside_the_repository(tmp_path: Path) -> None:
    executable = Path(sys.executable).with_name("holdout-first")
    result = subprocess.run(
        [str(executable), "demo", "--seed", "11", "--json"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["seed"] == 11
    assert payload["demo_succeeded"] is True
    assert payload["honest"]["survived"] is True
    assert payload["overfitted"]["survived"] is False
    assert payload["peeking"]["error"] == "LookaheadError"
