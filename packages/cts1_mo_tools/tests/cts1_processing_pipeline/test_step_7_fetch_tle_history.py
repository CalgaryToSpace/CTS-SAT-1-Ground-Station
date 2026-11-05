import pytest
from cts1_mo_tools.cts1_processing_pipeline.common import connect_duckdb
from cts1_mo_tools.cts1_processing_pipeline.step_7_fetch_tle_history import (
    pipeline as step_7_pipeline,
)


def _hf_secrets(token: str, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    monkeypatch.setenv(step_7_pipeline.HF_TOKEN_ENV_VAR, token)
    with connect_duckdb() as con:
        step_7_pipeline._use_hf_token(con)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        return con.execute(
            "SELECT type, secret_string FROM duckdb_secrets()"
        ).fetchall()


def test_hf_token_becomes_a_redacted_duckdb_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secrets = _hf_secrets("hf_not_a_real_token", monkeypatch)

    assert [secret_type for secret_type, _ in secrets] == ["huggingface"]
    assert "scope=hf://" in secrets[0][1]
    assert "hf_not_a_real_token" not in secrets[0][1]


def test_no_hf_token_means_no_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _hf_secrets("", monkeypatch) == []
