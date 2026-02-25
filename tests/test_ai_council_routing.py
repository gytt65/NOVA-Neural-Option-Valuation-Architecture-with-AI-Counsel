from ai_council.providers import LLMProvider, ProviderRoutingPolicy


def test_routing_prefers_low_cost_in_mostly_free_mode():
    providers = {
        "gemini": LLMProvider("gemini", "k", "m", cost_tier="low"),
        "openai": LLMProvider("openai", "k", "m", cost_tier="medium"),
        "anthropic": LLMProvider("anthropic", "k", "m", cost_tier="high"),
    }
    policy = ProviderRoutingPolicy(cost_mode="mostly_free")
    chosen = policy.choose_provider(providers, ["anthropic", "openai", "gemini"], high_impact=False)
    assert chosen is not None
    assert chosen.name == "gemini"


def test_routing_allows_priority_on_high_impact():
    providers = {
        "gemini": LLMProvider("gemini", "k", "m", cost_tier="low"),
        "openai": LLMProvider("openai", "k", "m", cost_tier="medium"),
    }
    policy = ProviderRoutingPolicy(cost_mode="mostly_free")
    chosen = policy.choose_provider(providers, ["openai", "gemini"], high_impact=True)
    assert chosen is not None
    assert chosen.name == "openai"


def test_routing_cost_capped_prefers_low_tier_near_budget():
    providers = {
        "gemini": LLMProvider("gemini", "k", "m", cost_tier="low"),
        "openai": LLMProvider("openai", "k", "m", cost_tier="medium"),
    }
    policy = ProviderRoutingPolicy(cost_mode="cost_capped", max_daily_budget_usd=1.0, spent_today_usd=0.95)
    chosen = policy.choose_provider(providers, ["openai", "gemini"], high_impact=True)
    assert chosen is not None
    assert chosen.name == "gemini"


def test_routing_policy_records_spend():
    provider = LLMProvider("openai", "k", "m", cost_tier="medium")
    policy = ProviderRoutingPolicy(cost_mode="cost_capped", max_daily_budget_usd=1.0)
    before = policy.spent_today_usd
    policy.record_call(provider)
    assert policy.spent_today_usd > before
