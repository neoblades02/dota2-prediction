def test_cli_module_imports_and_lists_commands():
    from typer.testing import CliRunner

    from dota_predictor.cli import app

    result = CliRunner().invoke(app, ["predict", "--help"])
    assert result.exit_code == 0
    assert "match" in result.output
