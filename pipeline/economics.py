"""
Pool economics calculator.
Estimates build cost and projected home value lift.
"""
import config


def calculate_pool_economics(
    home_value: float,
    lot_sqft: int,
    pool_sqft: int,
    state: str = "FL",
) -> dict:
    """
    Returns estimated pool build cost and expected home value lift.

    Pricing model:
      - Base pool cost (standard 15x30 gunite): $45,000
      - Deck / coping / equipment add-on: $12,000
      - Additional sqft premium for larger lots: $50/sqft over 450sqft
      - Regional adjustment factor
      - Home value lift: ~7% of home value (capped at $50K)
    """
    # Regional cost multipliers
    regional_multipliers = {
        "FL": 1.00, "TX": 0.95, "AZ": 0.92, "CA": 1.25,
        "GA": 0.98, "NC": 0.97, "SC": 0.96, "NV": 1.05,
    }
    multiplier = regional_multipliers.get(state.upper(), 1.0)

    base_cost = config.POOL_BASE_COST * multiplier
    deck_cost = 12_000 * multiplier

    # Premium for larger pools
    standard_pool_sqft = 450
    extra_sqft = max(0, pool_sqft - standard_pool_sqft)
    size_premium = extra_sqft * config.POOL_COST_PER_SQFT

    total_build_cost = round(base_cost + deck_cost + size_premium, -2)  # round to $100

    # Home value lift
    raw_lift = home_value * config.HOME_VALUE_LIFT_PCT
    value_lift = round(min(raw_lift, 50_000), -2)

    # ROI
    roi_pct = round((value_lift / total_build_cost) * 100, 1) if total_build_cost > 0 else 0

    return {
        "pool_cost": total_build_cost,
        "value_lift": value_lift,
        "roi_pct": roi_pct,
        "net_gain": value_lift - total_build_cost,
        "cost_breakdown": {
            "pool_shell": round(base_cost, -2),
            "deck_and_coping": round(deck_cost, -2),
            "equipment": round(size_premium, -2),
        },
    }
