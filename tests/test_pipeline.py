"""Offline smoke test of common.pipeline with a fake client on 3 dev examples."""
import json

import pytest

from common import config, data, evaluate
from common.config import RunConfig
from common.llm_client import LLMResult
from common.pipeline import run_model_pipeline

pytestmark = [pytest.mark.requires_dataset, pytest.mark.requires_official]


class GoldClient:
    """Answers each question with its gold SQL, served by 'FakeProv'."""

    def __init__(self, gold_by_q):
        self.gold_by_q = gold_by_q

    def complete(self, messages, cfg):
        user = messages[-1]["content"]
        sql = next(s for q, s in self.gold_by_q.items() if q in user)
        return LLMResult(status="ok", text=f"```sql\n{sql}\n```", prompt_tokens=10,
                         completion_tokens=5, latency_s=0.01, provider="FakeProv",
                         finish_reason="stop", attempts=1)


def test_pipeline_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    monkeypatch.setattr("common.config.out_root", lambda l: tmp_path / f"outputs_limit{l}")
    exs = data.load_examples(3)
    gold = data.gold_lines(3)
    client = GoldClient({e.question: g for e, (g, _) in zip(exs, gold)})
    cfg = RunConfig(run_name="smoke", model_id="qwen/qwen-2.5-7b-instruct", temperature=0.0,
                    provider_order=("FakeProv",), threads=2, limit=3)
    eps = [{"provider_name": "FakeProv", "quantization": "fp8"}]
    res = run_model_pipeline(cfg, client, eps, record_env=False, show_progress=False)

    assert res.eval.table.loc["all", "ex"] == pytest.approx(1.0)
    assert not res.mismatches
    assert len(res.pred_path.read_text().splitlines()) == 3
    assert len(res.gold_path.read_text().splitlines()) == 3
    assert res.output_dir == cfg.output_dir()
    assert (res.output_dir / "per_example.csv").is_file() and (res.output_dir / "providers.csv").is_file()
    assert res.pred_path == res.output_dir / "preds.txt"
    s = json.loads((res.output_dir / "summary.json").read_text())
    assert s["cli_check"]["passed"] is True
    assert res.providers.to_dict("records")[0]["quantization"] == "fp8"
    assert res.eval.per_example["provider"].tolist() == ["FakeProv"] * 3
