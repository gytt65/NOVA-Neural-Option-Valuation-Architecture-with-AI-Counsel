from ai_council.agents import load_senate_profiles


def test_load_senate_profiles_has_core_12_and_controls():
    profiles = load_senate_profiles()
    seat_ids = {p.seat_id for p in profiles}
    required = {
        "monetary_policy_chair",
        "global_macro_cio",
        "india_policy_political_risk",
        "corporate_actions_earnings",
        "credit_liquidity_stress",
        "derivatives_microstructure",
        "volatility_surface_arb",
        "systematic_statistical_pm",
        "geopolitical_supply_chain",
        "climate_weather_commodities",
        "risk_manager_cro",
        "quant_model_auditor",
        "red_team_prosecutor",
        "arbiter_chair",
    }
    assert required.issubset(seat_ids)
    assert len(profiles) >= 14
