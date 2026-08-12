"""The demo command is the smoke test: it must reproduce all three documented outcomes."""

from __future__ import annotations

import hashlib
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


def test_demo_seed_eleven_has_a_stable_complete_behavioral_digest() -> None:
    honest, overfitted, caught = run_demo(11)
    assert caught is not None
    payload = {
        "honest": honest.to_dict(),
        "overfitted": overfitted.to_dict(),
        "peeking": {
            "index": caught.index,
            "prefix_length": caught.prefix_length,
            "prefix_value": caught.prefix_value,
            "full_value": caught.full_value,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == (
        "adb84bd5d84a8461e0b119e8aa0d0a43f820c3f55f1991d77a818f6d686af429"
    )
    assert hashlib.sha256(honest.format_text().encode()).hexdigest() == (
        "63b5e8fe6b99b18cc4fcff6d75ff2aee5e453666948adb704bd99dfad694969d"
    )
    assert hashlib.sha256(overfitted.format_text().encode()).hexdigest() == (
        "b3f0dd115f9c35937c93ae10692edb4a249aff9cb0d6b7533682f48161cb6817"
    )


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


def test_cli_help_has_stable_complete_behavioral_digests() -> None:
    parser = build_parser()
    demo = next(action for action in parser._actions if action.dest == "command").choices["demo"]
    assert hashlib.sha256(parser.format_help().encode()).hexdigest() == (
        "57aaa54423fa59309099c6694e477d5cec013182eea2131811c54c277f214d4e"
    )
    assert hashlib.sha256(demo.format_help().encode()).hexdigest() == (
        "a07efe76536cd5889757cda4c82b867b75405e7566733683bb661111368d0168"
    )


def test_parser_exposes_the_documented_command_contract() -> None:
    parser = build_parser()
    assert parser.prog == "holdout-first"
    assert (
        parser.description
        == "Validation harness that fits on a small slice and demands survival on the "
        "large held-out remainder."
    )
    help_text = parser.format_help()
    assert "usage: holdout-first [-h] {demo} ..." in help_text
    assert "run the honest, overfitted, and non-causal reference strategies" in help_text


def test_demo_parser_exposes_option_help_and_explicit_values() -> None:
    parser = build_parser()
    args = parser.parse_args(["demo", "--seed", "-7", "--json"])
    assert args.command == "demo"
    assert args.seed == -7
    assert args.json is True
    demo = next(action for action in parser._actions if action.dest == "command").choices["demo"]
    help_text = demo.format_help()
    assert "--seed SEED" in help_text
    assert "panel seed (default: 11)" in help_text
    assert "emit the full report as JSON instead of text" in help_text


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


def test_installed_cli_text_and_json_outputs_are_stable(tmp_path: Path) -> None:
    executable = Path(sys.executable).with_name("holdout-first")
    text = subprocess.run(
        [str(executable), "demo", "--seed", "11"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    rendered_json = subprocess.run(
        [str(executable), "demo", "--seed", "11", "--json"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert hashlib.sha256(text.encode()).hexdigest() == (
        "b8a28037e953619f6b49ff08fef5585f51d4c81a42f2833c11a946ab52713f47"
    )
    assert hashlib.sha256(rendered_json.encode()).hexdigest() == (
        "a48580346efc7f44be53d6ad36848b05dcc87eac697d74fa23e7a277dd58378d"
    )


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
