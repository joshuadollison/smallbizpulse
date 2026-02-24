from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class InterventionRule:
    """Maps lexical cues to a normalized intervention recommendation."""

    theme: str
    cues: tuple[str, ...]
    recommendation: str


DEFAULT_RULES: tuple[InterventionRule, ...] = (
    InterventionRule(
        theme="operational_breakdown",
        cues=("slow", "wait", "late", "line", "management", "disorganized", "chaos"),
        recommendation="Refer to Small Business Development Center for operations diagnostics and workflow redesign.",
    ),
    InterventionRule(
        theme="food_supply_issues",
        cues=("stale", "cold", "raw", "ingredient", "supply", "portion", "quality"),
        recommendation="Connect to culinary consulting and supplier network support for food consistency recovery.",
    ),
    InterventionRule(
        theme="staff_service",
        cues=("rude", "service", "staff", "server", "attitude", "host", "ignored"),
        recommendation="Route to workforce/customer-service training program with staffing process coaching.",
    ),
    InterventionRule(
        theme="financial_distress",
        cues=("price", "expensive", "value", "overpriced", "cost", "portion"),
        recommendation="Refer to CDFI loan/grant readiness partner for pricing, margin, and liquidity intervention.",
    ),
)


@dataclass(frozen=True)
class RecommendationArtifacts:
    """Persisted recommendation mapping outputs."""

    output_dir: Path
    topic_recommendation_path: Path



def _normalize_terms(raw_terms: str | Iterable[str]) -> list[str]:
    if isinstance(raw_terms, str):
        parts = [part.strip().lower() for part in raw_terms.split(",") if part.strip()]
        return parts
    return [str(term).strip().lower() for term in raw_terms if str(term).strip()]



def _score_rule_match(terms: list[str], rule: InterventionRule) -> int:
    score = 0
    for cue in rule.cues:
        cue_lower = cue.lower()
        for term in terms:
            if cue_lower in term:
                score += 1
                break
    return score



def map_topic_terms_to_recommendations(
    topic_terms: pd.DataFrame,
    *,
    rules: tuple[InterventionRule, ...] = DEFAULT_RULES,
) -> pd.DataFrame:
    """Map BERTopic term bundles to intervention recommendations."""
    required = {"topic_id", "top_terms"}
    missing = sorted(required - set(topic_terms.columns))
    if missing:
        raise ValueError(f"topic_terms missing required columns: {missing}")

    rows: list[dict[str, str | int]] = []
    for row in topic_terms.itertuples(index=False):
        topic_id = int(getattr(row, "topic_id"))
        terms = _normalize_terms(getattr(row, "top_terms", ""))

        best_rule = None
        best_score = -1
        for rule in rules:
            score = _score_rule_match(terms, rule)
            if score > best_score:
                best_score = score
                best_rule = rule

        if best_rule is None:
            best_rule = DEFAULT_RULES[0]

        rows.append(
            {
                "topic_id": topic_id,
                "theme": best_rule.theme,
                "recommendation": best_rule.recommendation,
                "match_score": int(max(best_score, 0)),
            }
        )

    return pd.DataFrame(rows).sort_values(["match_score", "topic_id"], ascending=[False, True]).reset_index(drop=True)



def build_recommendation_artifacts(
    topic_terms: pd.DataFrame,
    *,
    output_dir: Path | str,
    rules: tuple[InterventionRule, ...] = DEFAULT_RULES,
) -> RecommendationArtifacts:
    """Create and persist topic-to-intervention recommendation mappings."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapped = map_topic_terms_to_recommendations(topic_terms, rules=rules)
    topic_recommendation_path = out_dir / "topic_recommendations.csv"
    mapped.to_csv(topic_recommendation_path, index=False)

    return RecommendationArtifacts(output_dir=out_dir, topic_recommendation_path=topic_recommendation_path)
