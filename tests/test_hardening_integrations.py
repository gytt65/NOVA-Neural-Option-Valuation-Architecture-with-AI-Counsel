import types

import numpy as np

from omega_features import set_features
from omega_model import FeatureFactory, OMEGAModel
from nirv_model import RegimeDetector
from unified_pipeline import UnifiedPricingPipeline
from quant_engine import (
    AdaptiveMeshPDE,
    ContagionGraph,
    LevyProcessPricer,
    QuantEngine,
)


def _dummy_nirv_model():
    class _DummyNIRV:
        def __init__(self):
            self.state = {}

        def price_option(self, *args, **kwargs):
            return types.SimpleNamespace(
                fair_value=110.0,
                market_price=float(kwargs.get("market_price", 100.0) or 100.0),
                mispricing_pct=10.0,
                signal="BUY",
                profit_probability=58.0,
                physical_profit_prob=60.0,
                confidence_level=70.0,
                expected_pnl=500.0,
                physical_expected_pnl=450.0,
                regime="Sideways",
                greeks={
                    "delta": 0.5,
                    "gamma": 0.01,
                    "theta": -1.0,
                    "vega": 2.0,
                    "rho": 0.1,
                    "vanna": 0.0,
                    "charm": 0.0,
                },
                tc_details={},
            )

    return _DummyNIRV()


def test_omega_feature_schema_timestamp_toggle():
    set_features(USE_OMEGA_TIMESTAMP_MISSING=False)
    names_off = FeatureFactory.get_feature_names()
    arr_off = FeatureFactory.to_array({})

    set_features(USE_OMEGA_TIMESTAMP_MISSING=True)
    names_on = FeatureFactory.get_feature_names()
    arr_on = FeatureFactory.to_array({})

    assert "timestamp_missing" not in names_off
    assert "timestamp_missing" in names_on
    assert len(arr_on) == len(arr_off) + 1

    set_features()


def test_omega_urgency_drift_wired_and_drift_suppresses_ml(tmp_path):
    set_features(
        USE_OMEGA_URGENCY=True,
        USE_OMEGA_DRIFT_GUARD=True,
        USE_CONFORMAL_INTERVALS=False,
        USE_RESEARCH_HIGH_CONVICTION=False,
        USE_OOS_RELIABILITY_GATE=False,
    )

    omega = OMEGAModel(nirv_model=_dummy_nirv_model(), data_dir=str(tmp_path / "omega"))

    class _DummyML:
        def __init__(self):
            self.is_trained = True
            self._schema_mismatch = False
            self._load_issue = None
            self.called_urgency = False
            self.called_drift = False
            self.training_X = []

        def predict_urgency(self, features):
            self.called_urgency = True
            return {"half_life_minutes": 5.0, "entry_strategy": "MARKET_TAKE"}

        def check_drift(self, features):
            self.called_drift = True
            return {"drift_detected": True, "drift_score": 1.0}

        def predict_correction(self, features):
            # Should be suppressed when drift is detected.
            return 0.25, 0.9

        def predict_correction_with_interval(self, features):
            return 0.25, 0.9, 0.20, 0.30

    dummy_ml = _DummyML()
    omega.ml = dummy_ml

    res = omega.price_option(
        spot=23500,
        strike=23500,
        T=7 / 365.0,
        r=0.065,
        q=0.012,
        option_type="CE",
        market_price=100,
        india_vix=14,
        fii_net_flow=0,
        dii_net_flow=0,
        days_to_rbi=10,
        pcr_oi=1.0,
        returns_30d=np.random.normal(0.0, 0.01, 30),
    )

    assert dummy_ml.called_urgency is True
    assert dummy_ml.called_drift is True
    assert float(res.ml_correction_pct) == 0.0
    assert "ML_DRIFT_DETECTED_CORRECTION_SUPPRESSED" in list(res.actionability_reasons)

    set_features()


def test_nirv_route_tracking_and_weighting():
    det = RegimeDetector()
    regime, probs = det.detect_regime(
        np.random.normal(0.0, 0.01, 30),
        india_vix=14.0,
        fii_net_flow=0.0,
    )
    assert isinstance(regime, str)
    assert isinstance(probs, dict)

    route = det._last_route
    det.update_route_outcome(True, route_name=route)
    stats = det.get_route_stats()
    assert route in stats
    assert stats[route]["total"] >= 1
    assert stats[route]["correct"] >= 1

    weight = det.route_quality_weight(route_name=route, min_samples=1)
    assert 0.7 <= weight <= 1.1


def test_nova_bse_and_shadow_lifecycle_flagged():
    set_features(USE_NOVA_BSE_VALIDATION=True, USE_NOVA_SHADOW_FEEDBACK=True)
    pipe = UnifiedPricingPipeline(
        {
            "njsde_paths": 64,
            "hedge_paths": 64,
            "shadow_delay_seconds": 0,
        }
    )

    state = {
        "vix": 14.0,
        "regime": "Neutral",
        "skew": -0.02,
        "market_price": 100.0,
        "bid": 99.0,
        "ask": 101.0,
        "timestamp": 1000.0,
        "current_market_prices": {(23500.0, "CE"): 100.0},
        "bse_context": {
            "strikes": [75000.0],
            "prices": {75000.0: 250.0},
            "ivs": {75000.0: 0.2},
            "sensex_spot": 75000.0,
            "bids": {75000.0: 249.0},
            "asks": {75000.0: 251.0},
        },
    }

    _ = pipe.price(
        spot=23500.0,
        strike=23500.0,
        T=7 / 365.0,
        sigma=0.15,
        option_type="CE",
        market_state=state,
        historical_returns=np.random.normal(0.0, 0.01, 60),
    )

    state["timestamp"] = 1001.0
    res = pipe.price(
        spot=23500.0,
        strike=23500.0,
        T=7 / 365.0,
        sigma=0.15,
        option_type="CE",
        market_state=state,
        historical_returns=np.random.normal(0.0, 0.01, 60),
    )

    assert isinstance(res.get("bse_validation"), dict)
    assert res["bse_validation"].get("enabled") is True
    assert isinstance(res.get("shadow_status"), dict)
    assert res["shadow_status"].get("enabled") is True
    assert isinstance(res.get("actionability_reasons"), list)

    set_features()


def test_levy_regularization_penalty_exposed():
    pricer = LevyProcessPricer(regularization_lambda=0.5)
    S = 100.0
    T = 0.25
    r = 0.02
    q = 0.0
    strikes = np.array([90.0, 100.0, 110.0])
    market_prices = [
        pricer.price(S, K, T, r, q, sigma=0.2, theta=-0.1, nu=0.25, option_type="CE")
        for K in strikes
    ]

    fit = pricer.calibrate_from_market(S, strikes, market_prices, T, r, q, option_type="CE")
    assert "regularization_penalty" in fit
    assert fit["regularization_penalty"] >= 0.0


def test_contagion_graph_threshold_filters_edges():
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, 250)
    y = np.roll(x, 1) + 0.15 * rng.normal(0.0, 1.0, 250)
    z = rng.normal(0.0, 1.0, 250)

    loose = ContagionGraph.build_graph({"x": x, "y": y, "z": z}, max_lag=3, threshold=0.05)
    strict = ContagionGraph.build_graph({"x": x, "y": y, "z": z}, max_lag=3, threshold=0.0001)
    assert strict["n_significant_edges"] <= loose["n_significant_edges"]


def test_adaptive_mesh_targeted_only_enforcement():
    pde = AdaptiveMeshPDE(targeted_only=True)

    try:
        pde.price(
            S=23500.0,
            K=23500.0,
            T=7 / 365.0,
            r=0.065,
            q=0.012,
            sigma=0.15,
            option_type="CE",
            contract_targeted=False,
            allow_full_chain=False,
        )
        raised = False
    except ValueError:
        raised = True

    assert raised is True

    px, _ = pde.price(
        S=23500.0,
        K=23500.0,
        T=7 / 365.0,
        r=0.065,
        q=0.012,
        sigma=0.15,
        option_type="CE",
        contract_targeted=False,
        allow_full_chain=True,
    )
    assert px >= 0.0


def test_macro_release_calendar_aligned_in_quant_engine():
    qe = QuantEngine(deterministic_mode=True, seed=42)
    res = qe.enhanced_price(
        S=23500.0,
        K=23500.0,
        T=7 / 365.0,
        r=0.065,
        q=0.012,
        sigma=0.15,
        option_type="CE",
        market_price=120.0,
        vix=14.0,
        returns=np.random.normal(0.0, 0.01, 60),
    )
    calendar = res.get("calendar", {})
    assert "near_macro_event" in calendar
    assert "macro_event_name" in calendar
