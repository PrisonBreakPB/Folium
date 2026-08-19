"""Tests for the CostMeter (per-session LLM cost budget)."""

import pytest

from folium.cost_meter import CostMeter


def test_meter_tracks_spent_and_ratio():
    meter = CostMeter(budget_usd=10.0, soft_ratio=0.8)
    meter.record(2.0)
    meter.record(3.0)
    assert meter.spent() == 5.0
    assert meter.ratio() == pytest.approx(0.5)
    assert meter.soft_reached() is False
    assert meter.exhausted() is False


def test_meter_soft_and_hard_thresholds():
    meter = CostMeter(budget_usd=10.0, soft_ratio=0.8)
    meter.record(8.0)  # 80%
    assert meter.soft_reached() is True
    assert meter.exhausted() is False
    meter.record(2.0)  # 100%
    assert meter.soft_reached() is True
    assert meter.exhausted() is True


def test_meter_disabled_when_budget_zero():
    meter = CostMeter(budget_usd=0.0)  # default: unlimited
    assert meter.enabled is False
    meter.record(100.0)
    assert meter.ratio() == 0.0
    assert meter.soft_reached() is False
    assert meter.exhausted() is False


def test_meter_ignores_none_and_nonpositive_costs():
    meter = CostMeter(budget_usd=10.0)
    meter.record(None)  # model not in pricing table
    meter.record(0.0)
    meter.record(-1.0)
    assert meter.spent() == 0.0


def test_config_budget_from_env(monkeypatch):
    monkeypatch.setenv("FOLIUM_BUDGET_USD", "25")
    monkeypatch.setenv("FOLIUM_BUDGET_SOFT_RATIO", "0.6")
    from folium.config import Config
    cfg = Config.from_env()
    assert cfg.budget_usd == 25
    assert cfg.budget_soft_ratio == pytest.approx(0.6)


def test_config_budget_defaults_unlimited(monkeypatch):
    monkeypatch.setenv("FOLIUM_BUDGET_USD", "0")
    monkeypatch.delenv("FOLIUM_BUDGET_SOFT_RATIO", raising=False)
    from folium.config import Config
    cfg = Config.from_env()
    assert cfg.budget_usd == 0.0  # unlimited by default
    assert cfg.budget_soft_ratio == pytest.approx(0.8)