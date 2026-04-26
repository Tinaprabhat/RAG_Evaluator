"""
Tests for the meta-evaluation module.

Uses a patched Evaluator (DummyEmbedder + token-overlap NLI) so the entire
flow runs offline without model downloads.
"""

import pytest

from tests._fixtures import DummyEmbedder

from ragwatch.core.config import Config
from ragwatch.core.evaluator import Evaluator
from ragwatch.meta_eval.meta_evaluator import MetaEvaluator
from ragwatch.meta_eval.synthetic_cases import (
    GOOD_CASES,
    HALLUCINATED_CASES,
    OFF_TOPIC_CASES,
    PARTIAL_CASES,
    case_count_summary,
    get_known_bad_cases,
    get_known_good_cases,
    get_synthetic_cases,
)


def _patched_evaluator():
    cfg = Config.cpu_safe()
    cfg.use_ann_validator = False
    cfg.use_self_consistency = False
    ev = Evaluator(cfg)
    ev.embedder = DummyEmbedder()
    if ev.math is not None:
        ev.math.embedder = ev.embedder
    if ev.nli is not None:
        ev.nli.embedder = ev.embedder

        def fake_entail(p, h):
            ps, hs = set(p.lower().split()), set(h.lower().split())
            if not hs:
                return 0.5
            return float(min(1.0, len(ps & hs) / len(hs)))

        ev.nli._entailment_score = fake_entail  # type: ignore[method-assign]
    return ev


# ------------------------------------------------------------
# Synthetic cases
# ------------------------------------------------------------

class TestSyntheticCases:
    def test_total_count(self):
        cases = get_synthetic_cases()
        assert len(cases) == len(GOOD_CASES) + len(HALLUCINATED_CASES) + len(OFF_TOPIC_CASES) + len(PARTIAL_CASES)

    def test_at_least_25_cases(self):
        # we promised at least 25 hand-crafted cases
        assert len(get_synthetic_cases()) >= 25

    def test_good_cases_are_good(self):
        for c in GOOD_CASES:
            assert c.is_good is True
            assert c.true_faithfulness == 1.0

    def test_hallucinated_cases_are_bad(self):
        for c in HALLUCINATED_CASES:
            assert c.is_good is False
            assert c.true_faithfulness == 0.0

    def test_known_good_vs_known_bad_split(self):
        good = get_known_good_cases()
        bad = get_known_bad_cases()
        assert all(c.is_good for c in good)
        assert all(not c.is_good for c in bad)
        # no overlap
        good_queries = {c.eval_input.query for c in good}
        bad_queries = {c.eval_input.query for c in bad}
        # disjoint by content (a query may appear in both 'faithful' and 'hallucinated' groups
        # because we deliberately built contrasting cases). Just check non-empty.
        assert len(good) > 0 and len(bad) > 0

    def test_summary_keys(self):
        s = case_count_summary()
        assert "TOTAL" in s
        assert s["TOTAL"] == len(get_synthetic_cases())
        assert "faithful" in s
        assert "hallucinated" in s


# ------------------------------------------------------------
# MetaEvaluator
# ------------------------------------------------------------

class TestMetaEvaluator:
    def test_runs_on_subset(self):
        ev = _patched_evaluator()
        meta = MetaEvaluator(ev).run(cases=GOOD_CASES[:3] + HALLUCINATED_CASES[:3])
        assert meta.n_cases == 6
        assert -1.0 <= meta.faithfulness_pearson <= 1.0
        assert 0.0 <= meta.faithfulness_auroc <= 1.0

    def test_empty_cases_rejected(self):
        ev = _patched_evaluator()
        with pytest.raises(ValueError):
            MetaEvaluator(ev).run(cases=[])

    def test_distinguishes_good_from_bad(self):
        """Sanity check: with the toy NLI, good cases score higher than bad."""
        ev = _patched_evaluator()
        meta = MetaEvaluator(ev).run()  # full case set
        # good cases should have higher mean composite than off-topic cases
        good_mean = meta.per_tag_composite.get("faithful", 0.0)
        off_topic_mean = meta.per_tag_composite.get("off_topic", 1.0)
        assert good_mean > off_topic_mean

    def test_trust_score_in_range(self):
        ev = _patched_evaluator()
        meta = MetaEvaluator(ev).run()
        assert 0.0 <= meta.trust_score <= 1.0
        assert meta.trust_label in ("TRUSTED", "ACCEPTABLE", "MARGINAL", "UNRELIABLE")

    def test_raw_data_present(self):
        ev = _patched_evaluator()
        meta = MetaEvaluator(ev).run(cases=GOOD_CASES[:2])
        assert "faithfulness" in meta.raw_scores
        assert "composite" in meta.raw_scores
        assert len(meta.raw_scores["faithfulness"]) == 2
        assert len(meta.raw_labels) == 2


class TestReports:
    def test_console_report(self):
        from ragwatch.meta_eval.report import to_console
        ev = _patched_evaluator()
        meta = MetaEvaluator(ev).run(cases=GOOD_CASES[:2] + HALLUCINATED_CASES[:2])
        out = to_console(meta)
        assert "RAGWatch" in out
        assert "TRUST SCORE" in out
        assert "Faithfulness Agreement" in out

    def test_json_report(self, tmp_path):
        from ragwatch.meta_eval.report import to_json
        ev = _patched_evaluator()
        meta = MetaEvaluator(ev).run(cases=GOOD_CASES[:2])
        out = tmp_path / "meta.json"
        to_json(meta, str(out))
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "trust_score" in content
        assert "faithfulness_auroc" in content
