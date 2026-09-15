"""Tests for MLOps champion/challenger logic."""

from __future__ import annotations

from mlops.model_registry import evaluate_champion_challenger


def test_promote_when_no_production_model():
    decision = evaluate_champion_challenger(0.85, None)
    assert decision.promoted is True
    assert "No Production" in decision.reason


def test_promote_when_challenger_beats_production():
    decision = evaluate_champion_challenger(0.86, 0.85)
    assert decision.promoted is True


def test_reject_when_challenger_worse():
    decision = evaluate_champion_challenger(0.84, 0.85)
    assert decision.promoted is False


def test_min_improvement_gate():
    decision = evaluate_champion_challenger(0.851, 0.85, min_improvement=0.01)
    assert decision.promoted is False

    decision = evaluate_champion_challenger(0.861, 0.85, min_improvement=0.01)
    assert decision.promoted is True
