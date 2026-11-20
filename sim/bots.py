from __future__ import annotations

import json
import hashlib
import math
import os
import random
import sqlite3
import threading
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing, contextmanager
from typing import Any, Callable

from .bot_joint_pricing import JointPriceTarget, solve_joint_prices
from .bot_market_forecast import forecast_market_sales
from .cpi import (
    PRICE_CPI_TOTAL,
    allocate_city_cpi,
    allocate_city_cpi_for_company,
    investment_average_prices,
    prepare_city_cpi_for_company,
    prepare_investment_average_prices,
)
from .db import all_rows, effective_employee_count, employee_count, get_setting, now_iso, one
from .engine import (
    available_loan_limit, current_company_net_assets, effective_research_probability,
    loan_ceiling_for_round, preview_round_operations,
)


BOT_API_VERSION = 42
_SUPER_BOT_SUBMISSION_LOCK = threading.Lock()
_FORECAST_POOL_LOCK = threading.Lock()
_FORECAST_POOL: ProcessPoolExecutor | None = None


def super_bot_analysis_order(
    conn: sqlite3.Connection,
    round_no: int,
) -> list[int]:
    """Public worker-facing alias for the shared AI analysis order."""
    from .ai_training import super_bot_analysis_order as resolve_order

    return list(resolve_order(conn, round_no))


def _forecast_candidate_task(payload: dict[str, Any]) -> dict[str, Any]:
    """Pickle-safe exact forecast task for local multi-core workers."""
    return forecast_market_sales(**payload)


def _forecast_candidates(
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run independent finalist forecasts in order, optionally on local cores.

    Streamlit Cloud and SCF remain sequential by default. The downloadable Mac
    worker opts into a persistent process pool, and any pool failure falls back
    to the identical in-process formula rather than losing a decision.
    """
    if not payloads:
        return []
    try:
        workers = max(1, min(12, int(os.environ.get("SUPER_BOT_PARALLEL_WORKERS", "1"))))
    except (TypeError, ValueError):
        workers = 1
    if workers <= 1 or len(payloads) <= 1:
        return [_forecast_candidate_task(payload) for payload in payloads]
    global _FORECAST_POOL
    try:
        with _FORECAST_POOL_LOCK:
            if _FORECAST_POOL is None:
                _FORECAST_POOL = ProcessPoolExecutor(max_workers=workers)
            pool = _FORECAST_POOL
        chunksize = max(1, len(payloads) // max(1, workers * 2))
        return list(pool.map(
            _forecast_candidate_task,
            payloads,
            chunksize=chunksize,
        ))
    except Exception:
        with _FORECAST_POOL_LOCK:
            if _FORECAST_POOL is not None:
                _FORECAST_POOL.shutdown(wait=False, cancel_futures=True)
                _FORECAST_POOL = None
        return [_forecast_candidate_task(payload) for payload in payloads]

BOT_PLANS = (
    {"production": 350, "ma": 1340, "markets": 1, "agents": 2, "research": 1_500_000},
    {"production": 850, "ma": 1480, "markets": 1, "agents": 2, "research": 1_800_000},
    {"production": 1700, "ma": 1650, "markets": 2, "agents": 3, "research": 2_000_000},
    {"production": 3000, "ma": 1900, "markets": 2, "agents": 3, "research": 2_200_000},
    {"production": 4800, "ma": 2250, "markets": 3, "agents": 4, "research": 2_000_000},
    {"production": 7200, "ma": 2700, "markets": 3, "agents": 4, "research": 1_600_000},
    {"production": 10000, "ma": 3300, "markets": 4, "agents": 5, "research": 0},
)

# Seven deliberately different opponents. Multipliers apply to CPI indices,
# not raw cash, so they remain meaningful when the KDS changes.
BOT_STYLES = (
    {"ma": 0.72, "qi": 0.85, "mi": 1.05, "high": 0.940, "cut": 80},
    {"ma": 1.00, "qi": 1.15, "mi": 1.25, "high": 0.945, "cut": 130},
    {"ma": 1.38, "qi": 1.75, "mi": 1.55, "high": 0.950, "cut": 190},
    {"ma": 1.82, "qi": 1.05, "mi": 1.90, "high": 0.955, "cut": 260},
    {"ma": 2.30, "qi": 2.25, "mi": 2.25, "high": 0.960, "cut": 340},
    {"ma": 2.85, "qi": 1.45, "mi": 2.70, "high": 0.965, "cut": 440},
    {"ma": 3.55, "qi": 2.90, "mi": 3.20, "high": 0.970, "cut": 560},
)

# MA/QI/MI threshold multiples, price stance, expansion pace, MI opening round
# and city breadth. Independent playbooks use zero to leave out a pool instead
# of varying only the investment amount inside the same three-pool strategy.
NORMAL_INVESTMENT_MIXES = (
    (0.002, 0.0, 0.0, "premium", 1.40, 9, 0),
    (0.0, 0.0, 0.0, "discount", 1.32, 9, 1),
    (0.0, 1.42, 0.0, "premium", 0.85, 9, 0),
    (0.0, 0.0, 1.95, "premium", 1.12, 1, 0),
    (1.12, 1.38, 0.0, "flex", 0.87, 9, 0),
    (1.43, 0.0, 0.0, "discount", 1.10, 9, 0),
    (0.0, 1.19, 1.85, "flex", 1.01, 3, 0),
    (1.24, 0.0, 0.0, "premium", 0.96, 9, 0),
    (1.06, 1.12, 1.17, "flex", 1.00, 4, 0),
    (0.0, 1.10, 0.0, "discount", 1.29, 9, 1),
    (1.33, 0.0, 1.38, "flex", 1.20, 2, 1),
    (0.0, 0.0, 2.18, "discount", 1.18, 2, 0),
)


def _ordinary_mix(slot: int) -> tuple:
    # Keep a stable playbook for each seat; cash, pressure and seeded
    # variation change the actual actions without rerolling its identity.
    return NORMAL_INVESTMENT_MIXES[slot % len(NORMAL_INVESTMENT_MIXES)]


def _ordinary_playbook_order(match_seed: str) -> list[int]:
    # Keep the core set balanced even in smaller lobbies. Shuffle the
    # assignment between matches so BOT01 is not always the same opponent.
    core = list(range(8))
    additional = list(range(8, len(NORMAL_INVESTMENT_MIXES)))
    if match_seed:
        rng = random.Random(match_seed)
        rng.shuffle(core)
        rng.shuffle(additional)
    return core + additional


def _super_attack_priority(
    *, is_human: bool, is_super: bool, opponent_rank: int,
    own_rank: int, predicted_margin: float,
) -> int:
    """Ordered target tiers; investment/price bonuses cannot invert them."""
    higher = opponent_rank < own_rank
    if is_human:
        return 5 if higher else 2
    if is_super and higher:
        return 4
    if not is_super and predicted_margin > 0:
        return 3
    return 1


def _super_target_pressure(
    before: dict[str, Any], after: dict[str, Any],
    players: list[dict[str, Any]],
) -> dict[str, float | int]:
    """Value actual revenue denied, not visible CPI which can be redistributed.

    A target must lose total sales across ALL cities and visible CPI. Revenue
    includes gains in other cities, so relocating sales is not fake damage.
    Tier ordering is separate from monetary value and each unit is counted once.
    """
    targets = []
    human_damage = 0.0
    strategic_super_damage = 0.0
    ordinary_bot_damage = 0.0
    for player in players:
        company_id = int(player["company_id"])
        first = before["companies"].get(company_id, {})
        last = after["companies"].get(company_id, {})
        first_units, last_units = first.get("sold_units", {}), last.get("sold_units", {})
        lost_units = float(first.get("sold_total", 0)) - float(last.get("sold_total", 0))
        cpi_drop = sum(
            max(0.0, float(first.get("breakdown", {}).get(city, {}).get("total_cpi", 0))
                - float(last.get("breakdown", {}).get(city, {}).get("total_cpi", 0)))
            for city in player["cities"]
        )
        lost_revenue = sum(
            (float(first_units.get(city, 0)) - float(last_units.get(city, 0)))
            * float(city_data["price"])
            for city, city_data in player["cities"].items()
        )
        secondary_denied = sum(
            max(
                0.0,
                float(
                    first.get("secondary_by_category", {})
                    .get(city, {})
                    .get("investment", 0.0)
                )
                - float(
                    last.get("secondary_by_category", {})
                    .get(city, {})
                    .get("investment", 0.0)
                ),
            )
            for city in player["cities"]
        )
        is_human = bool(player.get("is_human", False))
        is_super = bool(player.get("is_super", False))
        is_disruptor = bool(player.get("is_disruptor", False))
        effective_damage = max(0.0, lost_revenue)
        if is_human:
            human_damage += effective_damage
        elif is_super and not is_disruptor:
            strategic_super_damage += effective_damage
        else:
            ordinary_bot_damage += effective_damage
        if (
            lost_units < max(1.0, float(first.get("sold_total", 0)) * 0.005)
            or lost_revenue <= 0
            or (cpi_drop < 0.05 and secondary_denied < 1.0)
        ):
            continue
        targets.append({
            "targeted_damage": lost_revenue,
            "target_tier": int(player.get("attack_weight", 0)),
            "target_cpi_drop": cpi_drop,
            "target_surplus": lost_units,
            "target_secondary_denied": secondary_denied,
            "target_company_id": company_id,
            "target_is_human": is_human,
            "target_is_super": is_super,
        })
    selected = max(
        targets,
        key=lambda row: (
            bool(row["target_is_human"]),
            row["target_tier"],
            row["target_cpi_drop"],
            row["target_secondary_denied"],
            row["targeted_damage"],
        ),
        default={
            "targeted_damage": 0.0,
            "target_tier": 0,
            "target_cpi_drop": 0.0,
            "target_surplus": 0.0,
            "target_secondary_denied": 0.0,
            "target_is_human": False,
            "target_is_super": False,
        },
    )
    return {
        **selected,
        "human_damage": human_damage,
        "strategic_super_damage": strategic_super_damage,
        "ordinary_bot_damage": ordinary_bot_damage,
    }


def _select_disruptor_early_candidate(
    candidates: list[dict[str, Any]],
    *,
    leader_assets: float,
) -> dict[str, Any] | None:
    """Compatibility entry point: early disruptors maximize normal profit.

    The former 70%-of-leader target could deliberately select a losing cheap
    price to suppress the Bot's own growth. There is no such asset target now.
    """
    return _select_empirical_super_candidate(
        candidates, 0, tactical_price_allowed=False,
    )


def _select_disruptor_attack_candidate(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Prefer a flat net-asset curve and real low-price shipments."""
    valid = [
        row for row in candidates
        if math.isfinite(float(row.get("predicted_profit", -math.inf)))
    ]
    if not valid:
        return None
    non_loss = [row for row in valid if float(row["predicted_profit"]) >= 0.0]
    if not non_loss:
        # No break-even plan found: never trade more cash for nominal CPI.
        return max(valid, key=lambda row: (
            float(row["predicted_profit"]), float(row.get("low_price_sold", 0.0)),
        ))
    low_price = [row for row in non_loss if float(row.get("low_price_sold", 0.0)) > 0.0]
    pool = low_price or non_loss
    flat = [
        row for row in pool
        if float(row["predicted_profit"]) <= max(1_000.0, float(row.get("committed_cost", 0.0)) * 0.01)
    ]
    if not flat:
        # Do not pick a near-zero shipment solely to get closer to zero.
        # Keep plans that can still ship most of the best low-price volume.
        volume_key = "low_price_sold" if low_price else "predicted_sold"
        best_volume = max(float(row.get(volume_key, 0.0)) for row in pool)
        pool = [row for row in pool if float(row.get(volume_key, 0.0)) >= best_volume * 0.90]
        return min(pool, key=lambda row: (
            float(row["predicted_profit"]), -float(row.get("low_price_sold", 0.0)),
            -float(row.get("predicted_sold", 0.0)),
        ))
    return max(
        flat,
        key=lambda row: (
            float(row.get("low_price_sold", 0.0)),
            float(row.get("predicted_sold", 0.0)),
            -float(row["predicted_profit"]),
            bool(row.get("target_is_human", False)),
            float(row.get("targeted_damage", 0.0)),
        ),
    )


def _disruptor_floor_lock_prices(
    *,
    disruptor_ids: list[int],
    designated_company_id: int,
    market_max_prices: dict[str, float],
    coverable_cities: list[str],
    price_min: float,
) -> dict[int, dict[str, float]]:
    """Lock only stock-covered cities at ``minimum + 1``.

    A dedicated late disruptor uses a pure-price route.  Its inventory is
    assigned to the most valuable player-pressure cities first; each city
    consumes its estimated 40-CPI price capacity.  Cities that cannot be fully
    served stay at their KDS maximum so an empty low-price allocation never
    falls back into the price-only secondary pool.  At least one city is locked
    whenever stock exists, which makes this role a deterministic price anchor.
    """
    coverable = {str(city) for city in coverable_cities}
    result: dict[int, dict[str, float]] = {}
    for company_id in disruptor_ids:
        result[int(company_id)] = {
            str(city): (
                min(float(max_price), max(0.0, float(price_min)) + 1.0)
                if int(company_id) == int(designated_company_id)
                and str(city) in coverable
                else float(max_price)
            )
            for city, max_price in market_max_prices.items()
        }
    return result


def _disruptor_coverable_cities(
    city_options: list[dict[str, Any]],
    *,
    available_units: float,
) -> list[str]:
    """Choose whole low-price pools that the designated Bot can stock."""
    if not city_options or available_units <= 0:
        return []
    ordered = sorted(
        city_options,
        key=lambda option: (
            -float(option.get("attack_weight", 0.0)),
            float(option.get("lock_units", 0.0)),
            str(option["city"]),
        ),
    )
    remaining = max(0.0, float(available_units))
    coverable: list[str] = []
    for option in ordered:
        required = max(1.0, float(option.get("lock_units", 0.0)))
        if remaining + 1e-9 < required:
            continue
        coverable.append(str(option["city"]))
        remaining -= required
    # The designated role must always establish one price floor. Full-output
    # search normally makes this branch unnecessary; it is a safe fallback for
    # an unusually expensive KDS where not even the smallest pool fits.
    if not coverable:
        coverable.append(str(ordered[0]["city"]))
    return coverable


def _apply_super_settlement_forecast(
    candidate: dict[str, Any],
    own_forecast: dict[str, Any],
    own_cities: dict[str, dict[str, float | int]],
    *,
    candidate_available: int,
    home_city: str,
    transportation_cost: float,
    committed_cost: float,
    interest_cost: float,
    tax_rate: float,
    starting_assets: float,
    secondary_confidence: float = 1.0,
) -> dict[str, Any]:
    """Score a Super Bot plan from settlement sales, including redistribution.

    ``forecast_market_sales`` has already performed the same isolated price and
    investment secondary passes as settlement. Pending Super Bot decisions can
    discount only that secondary portion; the cheap visible-CPI estimate must
    never silently erase it.
    """
    exact_sold_units = {
        str(city): max(0, int(units))
        for city, units in own_forecast.get("sold_units", {}).items()
    }
    confidence = min(1.0, max(0.0, float(secondary_confidence)))
    primary_by_city = own_forecast.get("primary", {})
    secondary_by_city = own_forecast.get("secondary", {})
    if confidence < 1.0 and primary_by_city:
        sold_units = {
            str(city): (
                max(0.0, float(primary_by_city.get(city, 0.0)))
                + confidence * max(0.0, float(secondary_by_city.get(city, 0.0)))
            )
            for city in set(primary_by_city) | set(secondary_by_city)
        }
        forecast_total = sum(sold_units.values())
        if forecast_total > max(0, int(candidate_available)) and forecast_total > 0:
            scale = max(0, int(candidate_available)) / forecast_total
            sold_units = {city: units * scale for city, units in sold_units.items()}
    else:
        sold_units = exact_sold_units
    predicted_sold = float(sum(sold_units.values()))
    predicted_revenue = sum(
        float(units) * float(own_cities.get(city, {}).get("price", 0.0))
        for city, units in sold_units.items()
    )
    predicted_transport = sum(
        float(units) * max(0.0, float(transportation_cost))
        for city, units in sold_units.items()
        if city != str(home_city)
    )
    predicted_cost = (
        max(0.0, float(committed_cost))
        + max(0.0, float(interest_cost))
        + predicted_transport
    )
    predicted_pre_tax = predicted_revenue - predicted_cost
    resolved_tax_rate = max(0.0, float(tax_rate))
    predicted_profit = predicted_pre_tax - max(0.0, predicted_pre_tax * resolved_tax_rate)
    risk_pre_tax = predicted_revenue * 0.92 - predicted_cost
    risk_profit = risk_pre_tax - max(0.0, risk_pre_tax * resolved_tax_rate)
    visible_capacity = max(0.0, float(own_forecast.get("visible_total", 0.0)))
    raw_secondary_capacity = max(0.0, float(own_forecast.get("secondary_total", 0.0)))
    secondary_capacity = raw_secondary_capacity * confidence
    effective_capacity = visible_capacity + secondary_capacity
    available = max(0, int(candidate_available))
    candidate.update({
        "predicted_profit": predicted_profit,
        "risk_profit": risk_profit,
        "predicted_sold": predicted_sold,
        "predicted_revenue": predicted_revenue,
        "predicted_transport": predicted_transport,
        "primary_capacity": visible_capacity,
        "secondary_capacity": secondary_capacity,
        "raw_secondary_capacity": raw_secondary_capacity,
        "secondary_confidence": confidence,
        "effective_capacity": effective_capacity,
        "sell_ratio": predicted_sold / max(1, available),
        "coverage": effective_capacity / max(1, available),
        "ending_assets": float(starting_assets) + predicted_profit,
        "rival_damage": 0.0,
    })
    return candidate


def _select_empirical_super_candidate(
    candidates: list[dict[str, Any]],
    profile: int,
    *,
    tactical_price_allowed: bool,
    profit_target: float = 0.0,
    competitive_mode: bool = False,
    sacrifice_allowed: bool = False,
    contest_market: bool = False,
    expand_at_equal_profit: bool = False,
    rival_benchmark: float | None = None,
    race_for_first: bool = False,
    peer_field: bool = False,
) -> dict[str, Any] | None:
    """Choose the plan with the largest relative advantage in the live field.

    Profiles still diversify the candidate grid, rival presets and tie-breaks,
    but may no longer choose a passive plan merely to remain individually safe.
    A plan may accept a large own loss when its forecast denial of rival revenue
    is even larger. Cash affordability remains a hard constraint in candidate
    construction; solvency is deliberately not one here.
    """
    valid = [
        candidate for candidate in candidates
        if math.isfinite(float(candidate.get("predicted_profit", -math.inf)))
        and (not peer_field or not candidate.get("rival_seed") or float(candidate.get("sell_ratio", 0.0)) >= .96)
    ]
    if not valid:
        return None
    if race_for_first:
        contenders = [row for row in valid if "first_place_gap" in row
                      and float(row.get("risk_profit", -math.inf)) >= 0.]
        if peer_field and contenders:
            profit_floor = max(float(row["predicted_profit"]) for row in contenders) * .95
            contenders = [row for row in contenders if float(row["predicted_profit"]) >= profit_floor]
        if contenders:
            # Optimize the actual ranking objective, not this round's profit
            # in isolation. Never take a loss in the stress scenario.
            return max(contenders, key=lambda row: (
                float(row["first_place_gap"]), float(row["predicted_profit"]),
                float(row["predicted_sold"]),
            ))
    if not sacrifice_allowed:
        # Saturation and penultimate-round attackers must first look for a
        # profitable way to apply pressure. Deliberate capital sacrifice is
        # reserved for one trailing attacker in the final round so several
        # aggressive Bots cannot drag the whole field into collective losses.
        profitable_attacks = [
            candidate for candidate in valid
            if float(candidate.get("predicted_profit", -math.inf)) >= 0.0
        ]
        if profitable_attacks:
            valid = profitable_attacks
        else:
            # If all feasible plans lose money, preserve as much capital as
            # possible. Being distressed is not permission to sacrifice more.
            competitive_mode = False
    # Relative advantage is measured against the whole modelled field in the
    # same currency: own net profit plus aggregate rival revenue removed by
    # capacity capture/undercutting. Downside receives a deliberately modest
    # stability penalty; it prevents pointless crashes but still permits a
    # 1.9B own loss when the all-rival damage is materially larger (for example
    # 2.5B). When nobody can be hurt, this collapses to own-profit maximum.
    def strategic_value(candidate: dict[str, Any]) -> float:
        own_profit = float(candidate.get("predicted_profit", -math.inf))
        risk_profit = float(candidate.get("risk_profit", own_profit))
        starting_assets = max(1.0, float(candidate.get("starting_assets", 1.0)))
        ending_assets = float(candidate.get("ending_assets", starting_assets + own_profit))
        capital_ratio = ending_assets / starting_assets
        own_loss_penalty = 0.35 if competitive_mode else 0.10
        risk_loss_penalty = 0.10 if competitive_mode else 0.05
        stability_penalty = (
            max(0.0, -own_profit) * own_loss_penalty
            + max(0.0, -risk_profit) * risk_loss_penalty
            + starting_assets * max(0.0, 0.10 - capital_ratio) ** 2 * 50.0
            + max(0.0, -ending_assets) * 3.0
        )
        targeted_damage = float(candidate.get("targeted_damage", 0.0))
        # Only settlement-forecast damage is valued. The cheap search proxy
        # cannot justify losses and must not double-count the same sale.
        attack_multiplier = 0.90 if competitive_mode else 0.0
        return own_profit + targeted_damage * attack_multiplier - stability_penalty

    if competitive_mode:
        profit_best = max(valid, key=lambda row: float(row["predicted_profit"]))
        baseline_value = strategic_value({**profit_best, "targeted_damage": 0.0})
        worthwhile = [row for row in valid
                      if int(row.get("target_tier", 0)) > 0
                      and strategic_value(row) > baseline_value]
        if worthwhile:
            tier = max(int(row["target_tier"]) for row in worthwhile)
            valid = [row for row in worthwhile if int(row["target_tier"]) == tier]

    style = int(profile) % 7
    best_profit = max(float(row["predicted_profit"]) for row in valid)
    if contest_market and not competitive_mode and best_profit > 0.0:
        # Every strategic SB contests the field, including protected leaders.
        # A modest profit trade-off can buy real denied sales/extra rival stock;
        # the designated attackers retain their wider sacrifice rules above.
        valid = [row for row in valid if float(row["predicted_profit"]) >= best_profit * 0.92]

    def contest_value(candidate: dict[str, Any]) -> float:
        profit = float(candidate["predicted_profit"])
        if not contest_market or best_profit <= 0:
            return profit
        return profit + min(best_profit * 0.15, max(0.0, float(candidate.get("targeted_damage", 0.0))) * 0.20)

    selected = max(
        valid,
        key=lambda candidate: (
            strategic_value(candidate) if competitive_mode else contest_value(candidate),
            int(
                float(candidate.get("predicted_profit", -math.inf)) > 0
                and float(candidate.get("rival_damage", 0.0)) > 0
            ),
            float(candidate.get("targeted_damage", 0.0)) if competitive_mode else 0.0,
            float(candidate.get("target_cpi_drop", 0.0)) if competitive_mode else 0.0,
            float(candidate.get("target_surplus", 0.0)) if competitive_mode else 0.0,
            float(candidate.get("predicted_profit", -math.inf)),
            float(candidate.get("risk_profit", -math.inf)),
            int(float(candidate.get("predicted_profit", -math.inf)) > float(profit_target)),
            float(candidate.get("predicted_sold", 0.0)),
            float(candidate.get("sell_ratio", 0.0)),
            -abs(float(candidate.get("coverage", 0.0)) - 1.0),
            # Deterministic final tie-break preserves small profile differences
            # only when the economic forecast is otherwise identical.
            ((-1.0 if tactical_price_allowed and style == 6 else 1.0)
             * float(candidate.get("price_ratio", 0.0))),
        ),
    )
    if competitive_mode or (not expand_at_equal_profit and rival_benchmark is None):
        return selected
    # For standard bots use the best ordinary plan's own net profit as the
    # floor. An expanded or imitated plan must make at least as much.
    ordinary = [row for row in valid if not row.get("rival_seed") and not row.get("expansion_probe")]
    baseline = max(ordinary or valid, key=lambda row: float(row["predicted_profit"]))
    profit_floor = float(baseline["predicted_profit"])
    eligible = [row for row in valid if float(row["predicted_profit"]) >= profit_floor - 1e-6]
    if rival_benchmark is not None:
        # The rival's selling costs and tax were recalculated for this plan.
        def lead(row):
            rival = float(row.get("projected_rival_profit", rival_benchmark))
            return float(row["predicted_profit"]) - rival
        # Additional own sales can free secondary capacity that helps the
        # target even more. Apply the profit-gap floor before ranking bids.
        counters = [row for row in eligible if lead(row) >= lead(baseline) - 1e-6]
        best_profit = max(float(row["predicted_profit"]) for row in counters)
        if best_profit > 0:
            return max((row for row in counters if float(row["predicted_profit"]) >= best_profit * .995),
                       key=lambda row: (lead(row), float(row["predicted_sold"]), float(row["predicted_profit"])))
    if expand_at_equal_profit and profit_floor > 0:
        # Only exchange a negligible part of the newly found upside for
        # materially more sellable stock; never go below the old plan.
        best = max(eligible, key=lambda row: float(row["predicted_profit"]))
        upside = float(best["predicted_profit"]) - profit_floor
        return max((row for row in eligible if float(row["predicted_profit"]) >= profit_floor + upside * .98
                    and float(row.get("sell_ratio", 0.0)) >= .90),
                   key=lambda row: (float(row["predicted_sold"]), float(row["predicted_profit"])),
                   default=best)
    return selected


def _profitable_patent_budget(
    *, balance: float, remaining_rounds: int, material_spend: float, sold: float,
    available: int, projected_cash: float, reserve: float, risk_profit: float,
    amount_25: float, amount_75: float, hidden_cap: float, hidden_multiplier: float,
    buffer: float, patent_factor: float, tax_rate: float,
) -> float:
    """Buy only funded R&D whose expected future material savings pay back."""
    if remaining_rounds <= 0 or sold <= 0 or material_spend <= 0 or patent_factor >= 1:
        return 0.0
    minimum = max(0., amount_75 * hidden_multiplier + buffer)
    high = max(minimum, amount_75 * 2.)
    base_probability = effective_research_probability(
        balance, amount_25, amount_75, hidden_cap, hidden_multiplier)
    # Use current material prices (already discounted by active patents).
    # Only sold volume supports the projection; discount later rounds and
    # limit projected growth to two steps so one sellout cannot fund a gamble.
    throughput = min(1., sold / max(1., available))
    growth = 1.45 if throughput >= .92 and remaining_rounds >= 3 else 1.
    horizon = sum(growth ** min(step, 2) / 1.12 ** step for step in range(1, remaining_rounds+1))
    saving = material_spend * throughput * (1. - patent_factor) * horizon
    best_budget, best_gain = 0., 0.
    for total in sorted({minimum, high} if projected_cash >= high * 5 else {minimum}):
        spend = max(0., total - balance)
        if spend <= 0 or projected_cash - spend < reserve or risk_profit < spend * .65:
            continue
        probability = effective_research_probability(
            total, amount_25, amount_75, hidden_cap, hidden_multiplier)
        if total < amount_75 * hidden_multiplier or probability <= 0:
            continue
        gain = ((probability - base_probability) * saving * (1.-tax_rate)
                - spend * (1.-tax_rate))
        if gain > best_gain:
            best_budget, best_gain = spend, gain
    return best_budget


def _competitive_super_attackers(
    ranked_super_ids: list[int],
    *,
    official_round: int,
    total_rounds: int,
    markets_saturated: bool,
) -> set[int]:
    """Choose a small deterministic set of lower-ranked aggressive bots."""
    count = len(ranked_super_ids)
    if count <= 1 or official_round <= 0:
        return set()
    protected = 3 if count >= 3 else 1
    eligible = ranked_super_ids[protected:]
    if not eligible:
        return set()
    if official_round >= max(1, total_rounds - 1):
        attack_count = max(1, math.ceil(count / 3))
    elif markets_saturated:
        attack_count = 2 if count >= 6 else 1
    else:
        return set()
    return set(eligible[-min(len(eligible), attack_count):])


def _strict_profit_improvement(value: float, rate: float) -> float:
    value = float(value)
    return value + max(1.0, abs(value) * rate)


def _super_profit_target(
    conn: sqlite3.Connection,
    company_id: int,
    round_no: int,
    previous_super_id: int | None = None,
) -> float:
    """Benchmark against own growth, the preceding Bot and higher ranks.

    Rankings and profits are taken from the latest completed official round;
    current submitted decisions are still read separately by the CPI forecast.
    This keeps the benchmark deterministic for local and remote/split analysis.
    """
    if round_no <= 1:
        return 0.0
    own = one(
        conn,
        "SELECT round_no,net_assets,net_profit FROM results WHERE company_id=? "
        "AND round_no>=1 AND round_no<? ORDER BY round_no DESC LIMIT 1",
        (company_id, round_no),
    )
    if not own:
        return 0.0
    benchmark_round = int(own["round_no"])
    targets = [_strict_profit_improvement(float(own["net_profit"] or 0.0), 0.03)]
    higher = one(
        conn,
        "SELECT MAX(net_profit) AS profit FROM results WHERE round_no=? AND company_id<>? "
        "AND (net_assets>? OR (net_assets=? AND company_id<?))",
        (benchmark_round, company_id, own["net_assets"], own["net_assets"], company_id),
    )
    if higher and higher["profit"] is not None:
        targets.append(_strict_profit_improvement(float(higher["profit"]), 0.01))
    if previous_super_id is not None:
        previous_bot = one(
            conn,
            "SELECT net_profit FROM results WHERE company_id=? AND round_no=?",
            (previous_super_id, benchmark_round),
        )
        if previous_bot:
            targets.append(_strict_profit_improvement(float(previous_bot["net_profit"] or 0.0), 0.01))
    return max(0.0, *targets)


def _previous_report(conn: sqlite3.Connection, company_id: int, round_no: int) -> dict[str, Any]:
    row = one(
        conn,
        "SELECT report_json FROM results WHERE company_id=? AND round_no>=1 AND round_no<? "
        "ORDER BY round_no DESC LIMIT 1",
        (company_id, max(1, round_no)),
    )
    if not row:
        return {}
    try:
        return json.loads(row["report_json"])
    except (TypeError, ValueError):
        return {}


def _bot_salary(
    report: dict[str, Any], role: str, fallback: float, previous: float,
    profile: int, minimum: float, maximum: float, change_limit: float,
) -> tuple[float, float]:
    """Target 1.05--1.10 of last round's home average, capped at useful output."""
    key = "average_worker_salary" if role == "worker" else "average_engineer_salary"
    average = float(report.get("human_resources", {}).get(key, fallback) or fallback)
    target = average * (1.05 + profile * (0.05 / 6.0))
    low = max(minimum, previous - change_limit)
    high = min(maximum, previous + change_limit)
    salary = min(high, max(low, target))
    return salary, min(1.10, max(0.01, salary / max(average, 1.0)))


def _staff_delta(
    conn: sqlite3.Connection, company_id: int, role: str, round_no: int, required: float,
) -> int:
    """Hire shortages and lay off only staff above the calculated requirement."""
    current = employee_count(conn, company_id, role)
    effective = effective_employee_count(conn, company_id, role, round_no)
    required = max(0.0, required)
    if effective < required - 1e-9:
        return math.ceil(required - effective)
    removable = 0
    for cohort in all_rows(
        conn,
        "SELECT count,hire_round FROM employee_cohorts WHERE company_id=? AND role=? "
        "ORDER BY hire_round DESC,id DESC",
        (company_id, role),
    ):
        factor = 1.10 if round_no - int(cohort["hire_round"]) >= 2 else 1.0
        count = min(int(cohort["count"]), max(0, math.floor((effective - required) / factor)))
        removable += count
        effective -= count * factor
    return -min(current, removable)


def _selected_markets(bot: sqlite3.Row | dict[str, Any], markets: list[dict[str, Any]], count: int) -> list[int]:
    profile = int(bot["bot_profile"] if bot["bot_profile"] is not None else bot["id"]) % 7
    home = str(bot["home_city"] or markets[profile % len(markets)]["city"])
    home_index = next((i for i, market in enumerate(markets) if market["city"] == home), 0)
    candidates = [index for index in range(len(markets)) if index != home_index]
    seed_text = f"markets:{bot['id']}:{bot['code']}:{bot['bot_profile']}"
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    random.Random(seed).shuffle(candidates)
    return [home_index, *candidates[:max(0, min(len(markets), count) - 1)]]


def _bot_rng(
    bot: sqlite3.Row | dict[str, Any], round_no: int, super_mode: bool, match_seed: str = "",
) -> random.Random:
    """Stable per-team/per-round randomness, including after a Streamlit rerun."""
    seed_text = f"decision:{bot['id']}:{bot['code']}:{round_no}:{int(super_mode)}"
    if match_seed:
        seed_text += f":{match_seed}"
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    return random.Random(seed)


def _upper_typical(values: list[float]) -> float:
    """Strong rival reference that ignores one irrational all-in outlier."""
    clean = sorted(max(0.0, float(value)) for value in values)
    if not clean:
        return 0.0
    return clean[min(len(clean) - 1, math.floor((len(clean) - 1) * 0.80))]


def _super_index_overbid(threshold, own, rivals, stretch=1.25):
    active = [float(value) for value in rivals if value > 0.0]
    field = max(threshold, _upper_typical(active))
    # The CPI curve prices the marginal return; the strategy may still cross
    # the bend when the extra units sold pay for the higher investment.
    return max(own * stretch, field * stretch, field + threshold * .1)


def _super_city_bid_options(*, quote, cap, minimum, average, price_power, allow_low,
                            rivals, threshold, agents, seed):
    quotes = sorted(float(row["price"]) for row in rivals if row["price"] > 0)
    active_mi = [row["mi_effective"] for row in rivals if row["mi_effective"] > 0]
    local_bid = _upper_typical(active_mi) / (1.0 + .1 * agents)
    # Stable variation per team, round and city; rerunning a decision is deterministic.
    offset = (seed % 67) - 33
    anchor = min(cap, quotes[len(quotes) // 2] if quotes else average)
    def bound(value):
        return round(min(cap, max(minimum, value)), 2)

    # Small markdowns stay available even where deep discounting is locked.
    mild_floor = minimum if allow_low else max(minimum, .75 * cap)
    mild = bound(max(mild_floor, quote - (100 + offset)))
    medium = bound(max(mild_floor, min(quote, anchor) - (200 - offset)))
    premium = bound(max(quote, min(cap - max(1., 110 + offset), anchor + 150 - offset)))
    deep = medium
    if allow_low:
        rival_low = quotes[(len(quotes) - 1) // 4] if quotes else average
        weight_quote = _price_for_weight_advantage(average, rival_low, minimum, price_power, 1.2)
        deep = bound(min(medium, rival_low - (100 + offset), weight_quote))
    return mild, medium, premium, deep, max(threshold * .32, local_bid * .65), max(
        threshold * 1.12, local_bid * (1.20 + (seed % 23) / 100),
    )


def _balanced_production_group(
    component_workers: float,
    component_hours: float,
    product_engineers: float,
    product_hours: float,
    components_per_product: float,
) -> dict[str, float]:
    """Build the F/G/H/I production group defined in the strategy guide."""
    a = max(float(component_workers), 1e-9)
    b = max(float(component_hours), 1e-9)
    c = max(float(product_engineers), 1e-9)
    d = max(float(product_hours), 1e-9)
    e = max(float(components_per_product), 1e-9)
    worker_side = a * b * e
    engineer_side = c * d
    if all(abs(value - round(value)) < 1e-9 for value in (worker_side, engineer_side)):
        divisor = math.gcd(max(1, round(worker_side)), max(1, round(engineer_side)))
        workers = worker_side / divisor
        engineers = engineer_side / divisor
        components = 504.0 / b * workers / a
        products = components / e
    else:
        # A one-product normalized group keeps the same F:G ratio for decimal KDS values.
        products = 1.0
        components = e
        workers = a * b * components / 504.0
        engineers = c * d * products / 504.0
    return {
        "workers": workers,
        "engineers": engineers,
        "components": components,
        "products": products,
    }


def _affordable_group_count(
    available_cash: float,
    fixed_agent_and_mi: float,
    complete_group_cost: float,
    demand_groups: int,
) -> int:
    """Shared normal/super Bot group formula: floor((cash-fixed)/group cost)."""
    cash_after_fixed = max(0.0, float(available_cash) - max(0.0, float(fixed_agent_and_mi)))
    affordable = math.floor(cash_after_fixed / max(float(complete_group_cost), 1.0))
    return max(0, min(int(affordable), max(0, int(demand_groups))))


def _max_budgeted_groups(available_cash, group_products, estimate, budget_for):
    """Find the affordable whole-group limit using payroll and storage costs."""
    low, high = 0, max(1, estimate)
    best = budget_for(0)
    if best["total"] > available_cash + 1e-9:
        return -1, best
    # Check zero separately: a stock-only MA plan can keep one employee.
    while True:
        budget = budget_for(math.floor(high * group_products))
        if budget["total"] > available_cash + 1e-9:
            break
        low, high, best = high, high * 2, budget
    while low + 1 < high:
        middle = (low + high) // 2
        budget = budget_for(math.floor(middle * group_products))
        if budget["total"] <= available_cash + 1e-9:
            low, best = middle, budget
        else:
            high = middle
    return low, best


def _kds_price_depth(base_depth: float, price_power: int) -> float:
    """Translate an 8th-power price style to the current KDS exponent.

    Price CPI compares ``price_gap ** power``.  Transforming the normalized
    gap by ``8 / power`` preserves the intended relative CPI weight instead of
    treating the same cash undercut as equivalent in every competition.
    """
    return min(1.0, max(0.0, float(base_depth))) ** (8.0 / max(1, int(price_power)))


def _price_for_weight_advantage(
    market_average: float,
    rival_price: float,
    price_floor: float,
    price_power: int,
    weight_multiplier: float = 1.5,
) -> float:
    """Return the smallest KDS-aware undercut for a CPI-weight advantage."""
    average = float(market_average)
    rival_gap = max(0.0, average - float(rival_price))
    if rival_gap <= 0:
        return max(float(price_floor), min(average, float(rival_price)))
    target_gap = rival_gap * max(1.0, float(weight_multiplier)) ** (1.0 / max(1, int(price_power)))
    return max(float(price_floor), average - target_gap)


def _all_markets_near_capacity(
    conn: sqlite3.Connection,
    round_no: int,
    markets: list[dict[str, Any]],
    threshold: float = 0.90,
) -> bool:
    """Cash may be held only when every market was full in the same latest round."""
    if round_no <= 1 or not markets:
        return False
    latest = one(
        conn,
        "SELECT MAX(round_no) AS round_no FROM market_round_stats "
        "WHERE round_no>=1 AND round_no<?",
        (round_no,),
    )
    if not latest or latest["round_no"] is None:
        return False
    latest_round = int(latest["round_no"])
    for market in markets:
        stats = one(
            conn,
            "SELECT player_total_volume,market_size FROM market_round_stats "
            "WHERE city=? AND round_no=?",
            (market["city"], latest_round),
        )
        if not stats:
            return False
        utilization = (
            float(stats["player_total_volume"] or 0)
            / max(1.0, float(stats["market_size"] or 0))
        )
        if utilization + 1e-9 < threshold:
            return False
    return True


def _forecast_submitted_market(
    conn: sqlite3.Connection,
    round_no: int,
    markets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Forecast the complete submitted field with the settlement algorithm.

    This uses the same average-price fixed point and category-isolated
    secondary allocation as settlement.  Reading the KDS here (rather than
    accepting cached planner defaults) also means an unlocked KDS change is
    reflected the next time a round is analysed.
    """
    official_round = 1 if round_no < 0 else round_no
    growth = float(get_setting(conn, "market_growth", 1.10))
    total_rounds = max(1, int(get_setting(conn, "total_rounds", 5)))
    ma_threshold = float(get_setting(conn, "cpi_ma_large_threshold", 1300))
    price_power = max(1, int(get_setting(conn, "cpi_price_power", 8)))
    players: dict[int, dict[str, Any]] = {}
    operating_states = preview_round_operations(conn, round_no)
    for row in all_rows(
        conn,
        "SELECT c.id,c.product_inventory,d.worker_delta,d.engineer_delta,d.management_investment,"
        "d.production_volume,d.quality_investment FROM companies c JOIN decisions d "
        "ON d.company_id=c.id WHERE d.round_no=? AND d.submitted_at IS NOT NULL",
        (round_no,),
    ):
        company_id = int(row["id"])
        state = operating_states[company_id]
        players[company_id] = {
            "company_id": company_id,
            "available": state["available"],
            "ma_index": state["ma_index"],
            "qi_index": state["qi_index"],
            "cities": {},
        }
    if not players:
        return {"companies": {}, "player_average_prices": {}, "price_power": price_power}

    resolved_markets: list[dict[str, Any]] = []
    for market in markets:
        city = str(market["city"])
        previous = one(
            conn,
            "SELECT average_price FROM market_round_stats WHERE city=? AND round_no>=1 "
            "AND round_no<? ORDER BY round_no DESC LIMIT 1",
            (city, max(1, round_no)),
        )
        resolved_markets.append({
            "city": city,
            "market_size": (
                float(market["population"])
                * float(market["penetration"])
                * growth ** max(0, official_round - 1)
            ),
            "max_price": float(market["max_price"]),
            "base_average_price": (
                float(previous["average_price"])
                if previous else float(market["initial_avg_price"])
            ),
        })
        for row in all_rows(
            conn,
            "SELECT cd.company_id,cd.agent_delta,cd.marketing_investment,cd.price,COALESCE(a.count,0) AS current_agents "
            "FROM city_decisions cd LEFT JOIN agents a ON a.company_id=cd.company_id AND a.city=cd.city "
            "WHERE cd.round_no=? AND cd.city=?",
            (round_no, city),
        ):
            company_id = int(row["company_id"])
            if company_id not in players:
                continue
            route = operating_states[company_id]["city_decisions"][city]
            agents = int(route["agents_after"])
            if agents > 0:
                players[company_id]["cities"][city] = {
                    "agents": agents,
                    "marketing": float(route["marketing_investment"]),
                    "price": float(route["price"]),
                }
    return forecast_market_sales(
        players=list(players.values()),
        markets=resolved_markets,
        ma_large_threshold=ma_threshold,
        price_power=price_power,
    )


def _forecast_submitted_cpi_capacity(
    conn: sqlite3.Connection,
    round_no: int,
    markets: list[dict[str, Any]],
) -> dict[int, float]:
    """Return visible CPI capacity before stock and secondary allocation."""
    forecast = _forecast_submitted_market(conn, round_no, markets)
    return {
        int(company_id): float(result["visible_total"])
        for company_id, result in forecast["companies"].items()
    }


def _forecast_submitted_sales_capacity(
    conn: sqlite3.Connection,
    round_no: int,
    markets: list[dict[str, Any]],
) -> dict[int, float]:
    """Return exact sellable units, including category-isolated redistribution."""
    forecast = _forecast_submitted_market(conn, round_no, markets)
    return {
        int(company_id): float(result["sold_total"])
        for company_id, result in forecast["companies"].items()
    }


def _coordinate_super_bot_prices(
    conn: sqlite3.Connection,
    round_no: int,
    markets: list[dict[str, Any]],
    *, preserve_ids: set[int] | None = None,
) -> bool:
    """Legacy quota solver, retained for compatibility tests only.

    Production planning no longer calls this solver: market shares must be
    contested independently, not allocated among cooperating Super Bots.

    Individual candidates are deliberately persisted one at a time so an
    interrupted analysis is resumable.  Their prices cannot, however, be
    reconciled one at a time when the KDS uses a high price exponent: a later
    tiny undercut would erase an earlier Bot's forecast.  This final pass works
    in price-CPI weight space and updates the whole vector together.
    """
    if not markets:
        return False
    preserve_ids = preserve_ids or set()
    official_round = 1 if round_no < 0 else round_no
    total_rounds = max(1, int(get_setting(conn, "total_rounds", 5)))
    price_power = max(1, int(get_setting(conn, "cpi_price_power", 8)))
    price_min = float(get_setting(conn, "price_min", 3500))
    global_price_max = float(get_setting(conn, "price_max", 25000))
    growth = float(get_setting(conn, "market_growth", 1.10))
    worker_need = float(get_setting(conn, "component_workers", 3))
    worker_hours = float(get_setting(conn, "component_hours", 7))
    engineer_need = float(get_setting(conn, "product_engineers", 4))
    engineer_hours = float(get_setting(conn, "product_hours", 14))
    component_need = max(1, int(round(get_setting(conn, "components_per_product", 7))))
    patent_factor = float(get_setting(conn, "patent_factor", 0.70))
    transport_cost = float(get_setting(conn, "transport_cost", 0))
    market_by_city = {str(market["city"]): market for market in markets}

    previous_stats: dict[str, dict[str, float]] = {}
    for market in markets:
        city = str(market["city"])
        stats = one(
            conn,
            "SELECT average_price,player_total_volume,market_size FROM market_round_stats "
            "WHERE city=? AND round_no>=1 AND round_no<? ORDER BY round_no DESC LIMIT 1",
            (city, max(1, round_no)),
        )
        previous_stats[city] = {
            "average": float(stats["average_price"]) if stats else float(market["initial_avg_price"]),
            "utilization": (
                float(stats["player_total_volume"] or 0) / max(1.0, float(stats["market_size"] or 0))
                if stats else 0.0
            ),
        }
    all_super_field = not bool(one(
        conn,
        "SELECT 1 FROM companies WHERE is_super_bot=0 LIMIT 1",
    ))
    low_price_open = (
        all_super_field
        or official_round >= max(1, total_rounds - 1)
        or all(value["utilization"] >= 0.90 for value in previous_stats.values())
        or any(
            value["average"] >= min(global_price_max, float(market_by_city[city]["max_price"])) * 0.75
            for city, value in previous_stats.items()
        )
    )
    if not low_price_open:
        return False

    rows = [dict(row) for row in all_rows(
        conn,
        "SELECT c.id,c.is_super_bot,c.bot_profile,c.home_city,c.cash,c.patents,"
        "c.component_inventory,c.product_inventory,c.component_storage_capacity,c.product_storage_capacity,"
        "d.loan_change,d.worker_salary,d.engineer_salary,d.worker_delta,d.engineer_delta,d.production_volume,"
        "d.management_investment,d.quality_investment,cd.city,cd.price,cd.marketing_investment,"
        "cd.agent_delta,COALESCE(a.count,0) AS agents_before,"
        "COALESCE(a.count,0)+cd.agent_delta AS agents_after "
        "FROM companies c JOIN decisions d ON d.company_id=c.id "
        "JOIN city_decisions cd ON cd.company_id=c.id AND cd.round_no=d.round_no "
        "LEFT JOIN agents a ON a.company_id=c.id AND a.city=cd.city "
        "WHERE d.round_no=? AND d.submitted_at IS NOT NULL",
        (round_no,),
    )]
    if not rows:
        return False

    company_rows: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if int(row["agents_after"] or 0) > 0:
            company_rows.setdefault(int(row["id"]), []).append(row)
    price_led_ids: set[int] = set()
    for company_id, active_rows in company_rows.items():
        sample = active_rows[0]
        if not int(sample["is_super_bot"] or 0) or company_id in preserve_ids:
            continue
        workers = max(
            0,
            employee_count(conn, company_id, "worker")
            + int(sample["worker_delta"] or 0),
        )
        engineers = max(
            0,
            employee_count(conn, company_id, "engineer")
            + int(sample["engineer_delta"] or 0),
        )
        available = (
            float(sample["product_inventory"] or 0) * 1.2
            + float(sample["production_volume"] or 0)
        )
        ma_index = float(sample["management_investment"] or 0) / max(1, workers + engineers)
        qi_index = float(sample["quality_investment"] or 0) / max(1.0, available)
        if ma_index <= 1.01 and qi_index <= 1.01:
            # Price-led plans may be pure price or price + per-city MI. Both
            # deliberately chose their undercut in the profit search.
            price_led_ids.add(company_id)
    prior_sellthrough: dict[int, float] = {}
    for company_id in company_rows:
        prior = one(
            conn,
            "SELECT sold,produced,inventory FROM results WHERE company_id=? AND round_no>=1 "
            "AND round_no<? ORDER BY round_no DESC LIMIT 1",
            (company_id, max(1, round_no)),
        )
        previous_available = (
            float(prior["sold"] or 0) + float(prior["inventory"] or 0)
            if prior else 0.0
        )
        prior_sellthrough[company_id] = (
            float(prior["sold"] or 0) / previous_available
            if prior and previous_available > 0 else 1.0
        )

    forecast = _forecast_submitted_market(conn, round_no, markets)
    forecast_companies = forecast["companies"]

    company_active_size = {
        company_id: sum(
            float(market_by_city[str(row["city"])]["population"])
            * float(market_by_city[str(row["city"])]["penetration"])
            * growth ** max(0, official_round - 1)
            for row in active_rows
        )
        for company_id, active_rows in company_rows.items()
    }
    updates: list[tuple[float, int, str]] = []
    original_prices = {
        (int(row["id"]), str(row["city"])): float(row["price"] or 0)
        for row in rows
    }
    for market in markets:
        city = str(market["city"])
        base_average = previous_stats[city]["average"]
        market_size = (
            float(market["population"])
            * float(market["penetration"])
            * growth ** max(0, official_round - 1)
        )
        active_rows = [row for row in rows if str(row["city"]) == city and int(row["agents_after"] or 0) > 0]
        specialists: list[dict[str, Any]] = []
        for row in active_rows:
            if not int(row["is_super_bot"] or 0):
                continue
            company_id = int(row["id"])
            if company_id in price_led_ids:
                # The optimiser explicitly selected price-led competition.
                # Joint coordination may use it as an external anchor but must
                # not raise away the deep undercut it just selected.
                continue
            inventory_crisis = (
                int(row["product_inventory"] or 0) > 0
                and prior_sellthrough.get(company_id, 1.0) < 0.80
            )
            if (
                float(row["price"] or 0) < base_average
                or inventory_crisis
            ):
                specialists.append(row)
        if not specialists:
            continue

        specialist_ids = {int(row["id"]) for row in specialists}
        external_prices = [
            float(row["price"] or 0)
            for row in active_rows
            if int(row["id"]) not in specialist_ids
        ]
        targets: list[JointPriceTarget] = []
        for row in specialists:
            company_id = int(row["id"])
            home = market_by_city.get(str(row["home_city"]), market)
            material_factor = patent_factor ** int(row["patents"] or 0)
            component_labor = (
                component_need * worker_need * worker_hours / 504.0
                * float(row["worker_salary"] or home["worker_initial_salary"]) * 3
            )
            product_labor = (
                engineer_need * engineer_hours / 504.0
                * float(row["engineer_salary"] or home["engineer_initial_salary"]) * 3
            )
            direct_cost = (
                component_need * float(home["component_material"]) * material_factor
                + float(home["product_material"]) * material_factor
                + component_labor + product_labor
                + (transport_cost if city != str(row["home_city"]) else 0.0)
            )
            available = float(row["product_inventory"] or 0) + float(row["production_volume"] or 0)
            company_marketing = sum(
                float(item["marketing_investment"] or 0)
                for item in company_rows.get(company_id, [])
            )
            investment_unit_cost = (
                float(row["management_investment"] or 0)
                + float(row["quality_investment"] or 0)
                + company_marketing
            ) / max(1.0, available)
            active_size = max(1.0, company_active_size.get(company_id, market_size))
            personality = int(row["bot_profile"] or company_id) % 7
            sellthrough_target = 0.78 + personality * 0.01
            desired_city_units = available * sellthrough_target * market_size / active_size
            city_forecast = forecast_companies.get(company_id, {})
            investment_units = float(city_forecast.get("investment", {}).get(city, 0.0))
            required_price_units = max(0.0, desired_city_units - investment_units)
            requested_share = required_price_units / max(1.0, market_size * 0.40)
            requested_share *= 0.92 + personality * 0.025
            # Keep several profitable participants in the price route. A
            # single Bot taking the entire pool is the exact failure mode this
            # pass is designed to prevent.
            requested_share = min(0.22, max(0.025, requested_share))
            targets.append(JointPriceTarget(
                key=company_id,
                target_share=requested_share,
                share_cap=0.24,
                cost_floor=max(price_min, (direct_cost + investment_unit_cost) * 1.04),
            ))
        solved = solve_joint_prices(
            base_average=base_average,
            external_prices=external_prices,
            targets=targets,
            price_power=price_power,
            price_min=price_min,
            price_max=min(global_price_max, float(market["max_price"])),
            total_share_cap=0.68,
            zero_external_anchor_gap=0.025,
        )
        for company_id, result in solved.items():
            # Joint coordination may spread the price CPI, but it must never
            # push a profit-selected candidate below its own price. Production
            # can be controlled instead; forced undercutting caused the old
            # final-round crashes.
            safe_price = max(
                float(result.price),
                original_prices.get((int(company_id), city), price_min),
            )
            updates.append((round(safe_price, 2), int(company_id), city))

    for price, company_id, city in updates:
        conn.execute(
            "UPDATE city_decisions SET price=? WHERE company_id=? AND round_no=? AND city=?",
            (price, company_id, round_no, city),
        )
    return bool(updates or price_led_ids)


def _rebalance_super_bot_production(
    conn: sqlite3.Connection,
    round_no: int,
    markets: list[dict[str, Any]],
) -> None:
    """Reconcile mixed-field joint CPI without erasing chosen aggression."""
    worker_need = float(get_setting(conn, "component_workers", 3))
    worker_hours = float(get_setting(conn, "component_hours", 7))
    engineer_need = float(get_setting(conn, "product_engineers", 4))
    engineer_hours = float(get_setting(conn, "product_hours", 14))
    component_need = max(1, int(round(get_setting(conn, "components_per_product", 7))))
    group = _balanced_production_group(
        worker_need, worker_hours, engineer_need, engineer_hours, component_need,
    )
    worker_training = float(get_setting(conn, "worker_training_cost", 0))
    engineer_training = float(get_setting(conn, "engineer_training_cost", 0))
    add_agent_cost = float(get_setting(conn, "agent_add_cost", 300000))
    remove_agent_cost = float(get_setting(conn, "agent_remove_cost", 100000))
    transportation_cost = max(0.0, float(get_setting(conn, "transport_cost", 0)))
    patent_factor = float(get_setting(conn, "patent_factor", 0.70))
    growth = float(get_setting(conn, "market_growth", 1.10))
    total_rounds = max(1, int(get_setting(conn, "total_rounds", 5)))
    ma_large_threshold = max(1.0, float(get_setting(conn, "cpi_ma_large_threshold", 1300)))
    ma_index_ceiling = ma_large_threshold * (8000.0 / 1300.0)
    official_round = 1 if round_no < 0 else round_no

    market_by_city = {str(market["city"]): market for market in markets}
    home_by_company = {
        int(row["id"]): str(row["home_city"] or "")
        for row in all_rows(conn, "SELECT id,home_city FROM companies WHERE is_super_bot=1")
    }
    all_bot_field = not bool(one(conn, "SELECT 1 FROM companies WHERE is_bot=0 LIMIT 1"))
    contest_round = official_round >= (min(4, total_rounds) if all_bot_field else 4)
    all_super_field = not bool(one(
        conn, "SELECT 1 FROM companies WHERE is_super_bot=0 LIMIT 1",
    ))
    explore_growth = (
        not all_super_field and official_round >= 4
        and not _all_markets_near_capacity(conn, round_no, markets, threshold=0.95)
    )
    ranked = all_rows(conn,
        "SELECT c.id FROM companies c LEFT JOIN results r ON r.company_id=c.id "
        "AND r.round_no=(SELECT MAX(round_no) FROM results WHERE round_no>=1 AND round_no<?) "
        "WHERE c.is_super_bot=1 ORDER BY r.net_assets IS NULL,r.net_assets DESC,c.id",
        (max(1, round_no),))
    preserve_ids = _competitive_super_attackers(
        [int(row["id"]) for row in ranked], official_round=round_no,
        total_rounds=total_rounds,
        markets_saturated=_all_markets_near_capacity(conn, round_no, markets),
    )
    if round_no >= 1 and official_round >= max(1, total_rounds - 1):
        preserve_ids.update(
            int(row["id"])
            for row in all_rows(
                conn,
                "SELECT id FROM companies WHERE is_bot=1 AND is_disruptor_bot=1",
            )
        )
    # Do not run the legacy joint-price quota solver here. Each SB's prices
    # came from its own exact profit/denial search; reserving low-price shares
    # for peers would turn competitors into a cartel. The remaining pass only
    # rechecks affordable production against actual, shared secondary pools.

    # Earlier drafts saw synthetic later peers. Give a squeezed SB a bounded
    # best-response search against the now complete field BEFORE reducing its
    # production. It can buy more CPI, focus MI or undercut rather than simply
    # surrendering market to the last Bot in the analysis order.
    for attempt in range(2):
        live = _forecast_submitted_market(conn, round_no, markets)["companies"]
        squeezed = [
            int(row["id"]) for row in all_rows(
                conn, "SELECT id FROM companies WHERE is_super_bot=1 ORDER BY id",
            )
            if int(row["id"]) not in preserve_ids
            and float(live.get(int(row["id"]), {}).get("available", 0.0)) > 0.0
            and float(live[int(row["id"])]["sold_total"])
            < float(live[int(row["id"])]["available"]) * 0.90
        ]
        if explore_growth and attempt == 0:
            # A sell-out only proves the current draft was too small to
            # expose its upside. Give the best-funded sell-outs a bounded
            # second look after synthetic rivals have been replaced.
            growers = all_rows(
                conn, "SELECT c.id,c.cash,d.production_volume FROM companies c "
                "JOIN decisions d ON d.company_id=c.id AND d.round_no=? "
                "WHERE c.is_super_bot=1 AND d.is_draft=1 "
                "ORDER BY c.cash/(1.0+d.production_volume) DESC,c.id",
                (round_no,),
            )
            extra = [
                int(row["id"]) for row in growers
                if int(row["id"]) not in preserve_ids and int(row["id"]) not in squeezed
                and float(live.get(int(row["id"]), {}).get("available", 0.0)) > 0.0
                and float(live[int(row["id"])]["sold_total"]) >= float(live[int(row["id"])]["available"]) * 0.97
            ]
            squeezed.extend(extra[:2])
        if not squeezed:
            break
        for company_id in squeezed:
            _submit_bots(
                conn, round_no, True, replace_existing=True,
                target_ids={company_id}, defer_rebalance=True,
                use_complete_field=True,
            )

    if (
        not all_super_field
        and not (all_bot_field and contest_round)
        and official_round <= max(2, math.ceil(total_rounds * 0.50))
        and not _all_markets_near_capacity(conn, round_no, markets, threshold=0.60)
    ):
        # During a mixed-field unsaturated expansion phase, keep the maximum
        # affordable complete-group production selected by the search. An
        # all-Super field still needs the joint pass: every Bot's individually
        # forecast secondary pool overlaps the others, so treating all of
        # those provisional shares as simultaneously guaranteed overproduces.
        return

    # The candidate search has already compared affordable production levels.
    # This pass may trim newly revealed surplus after simultaneous price
    # coordination, but it must never expand back to the cash maximum.
    for _ in range(12):
        field_forecast = _forecast_submitted_market(conn, round_no, markets)
        # Production can shrink sharply once the whole field competes for the
        # same secondary pools. Do not leave newly opened cities behind when
        # their exact post-redistribution revenue cannot even repay the Agent
        # opening fee. Closing those routes first also prevents their price or
        # investment CPI from distorting the next secondary pass.
        pruned_routes = False
        for company_id, forecast in field_forecast["companies"].items():
            if int(company_id) in preserve_ids:
                continue
            sold_units = forecast.get("sold_units", {})
            for route in all_rows(
                conn,
                "SELECT cd.city,cd.agent_delta,cd.price FROM city_decisions cd "
                "JOIN companies c ON c.id=cd.company_id "
                "WHERE cd.company_id=? AND cd.round_no=? AND c.is_super_bot=1 "
                "AND cd.agent_delta>0",
                (int(company_id), round_no),
            ):
                city = str(route["city"])
                units = max(0.0, float(sold_units.get(city, 0.0)))
                route_revenue = units * max(0.0, float(route["price"] or 0))
                route_floor = float(route["agent_delta"] or 0) * add_agent_cost
                if city != home_by_company.get(int(company_id), ""):
                    route_floor += units * transportation_cost
                if route_revenue <= route_floor + 1e-9:
                    conn.execute(
                        "UPDATE city_decisions SET agent_delta=0,marketing_investment=0 "
                        "WHERE company_id=? AND round_no=? AND city=?",
                        (int(company_id), round_no, city),
                    )
                    pruned_routes = True
        if pruned_routes:
            # Recompute CPI, average prices and both secondary pools from the
            # changed city set before touching production or investment.
            continue

        # Production follows the exact units this field can sell after the
        # category-isolated secondary pass. Visible CPI alone misses legitimate
        # price-to-price and investment-to-investment redistribution.
        capacities = {
            int(company_id): float(result["sold_total"])
            for company_id, result in field_forecast["companies"].items()
        }
        changed = False
        for row in all_rows(
            conn,
            "SELECT c.*,d.loan_change,d.worker_salary,d.engineer_salary,d.worker_delta,d.engineer_delta,"
            "d.management_investment,d.production_volume,d.quality_investment FROM companies c "
            "JOIN decisions d ON d.company_id=c.id WHERE c.is_super_bot=1 AND d.round_no=?",
            (round_no,),
        ):
            company_id = int(row["id"])
            if company_id in preserve_ids:
                continue
            home = market_by_city.get(str(row["home_city"]), markets[0])
            old_products = max(0, int(row["product_inventory"] or 0))
            old_components = max(0, int(row["component_inventory"] or 0))
            current_production = max(0, int(row["production_volume"] or 0))
            current_workers = max(0, employee_count(conn, company_id, "worker") + int(row["worker_delta"] or 0))
            current_engineers = max(0, employee_count(conn, company_id, "engineer") + int(row["engineer_delta"] or 0))
            ma_index = float(row["management_investment"] or 0) / max(1, current_workers + current_engineers)
            qi_index = float(row["quality_investment"] or 0) / max(1.0, old_products * 1.2 + current_production)
            material_factor = patent_factor ** int(row["patents"] or 0)
            city_cost = one(
                conn,
                "SELECT COALESCE(SUM(CASE WHEN cd.agent_delta>0 THEN cd.agent_delta*? "
                "WHEN cd.agent_delta<0 THEN -cd.agent_delta*? ELSE 0 END),0) AS agent_cost,"
                "COALESCE(SUM(cd.marketing_investment),0) AS marketing FROM city_decisions cd "
                "WHERE cd.company_id=? AND cd.round_no=?",
                (add_agent_cost, remove_agent_cost, company_id, round_no),
            )
            fixed_cost = float(city_cost["agent_cost"] or 0) + float(city_cost["marketing"] or 0)

            def plan_for(groups: int) -> dict[str, float | int]:
                production = max(0, int(math.floor(groups * group["products"])))
                components_to_make = max(0, production * component_need - old_components)
                worker_required = components_to_make * worker_need * worker_hours / 504.0
                engineer_required = production * engineer_need * engineer_hours / 504.0
                worker_delta = _staff_delta(conn, company_id, "worker", round_no, worker_required)
                engineer_delta = _staff_delta(conn, company_id, "engineer", round_no, engineer_required)
                workers = max(0, employee_count(conn, company_id, "worker") + worker_delta)
                engineers = max(0, employee_count(conn, company_id, "engineer") + engineer_delta)
                staff = workers * float(row["worker_salary"]) * 3 + engineers * float(row["engineer_salary"]) * 3
                staff += max(0, worker_delta) * worker_training + max(0, engineer_delta) * engineer_training
                staff += max(0, -worker_delta) * float(row["worker_salary"])
                staff += max(0, -engineer_delta) * float(row["engineer_salary"])
                materials = components_to_make * float(home["component_material"]) * material_factor
                materials += production * float(home["product_material"]) * material_factor
                storage = max(
                    0,
                    old_components + components_to_make - max(old_components, int(row["component_storage_capacity"] or 0)),
                ) * float(home["component_storage"])
                storage += max(
                    0,
                    old_products + production - max(old_products, int(row["product_storage_capacity"] or 0)),
                ) * float(home["product_storage"])
                management = ma_index * max(1, workers + engineers)
                quality = qi_index * max(1.0, old_products * 1.2 + production)
                return {
                    "production": production,
                    "worker_delta": worker_delta,
                    "engineer_delta": engineer_delta,
                    "management": management,
                    "quality": quality,
                    "total": fixed_cost + staff + materials + storage + management + quality,
                }

            total_funds = float(row["cash"]) + max(0.0, float(row["loan_change"] or 0))
            current_groups = max(
                0,
                int(round(current_production / max(group["products"], 1.0))),
            )
            sellable = max(0.0, capacities.get(company_id, 0.0))
            if sellable >= old_products + current_production * 0.97:
                continue
            target_new = max(0.0, sellable * 1.02 - old_products)
            target_groups = min(
                current_groups,
                max(0, int(math.ceil(target_new / max(group["products"], 1.0)))),
            )
            selected_plan = plan_for(target_groups)
            if float(selected_plan["total"]) > total_funds + 1e-9:
                continue
            if int(selected_plan["production"]) == current_production:
                continue
            conn.execute(
                "UPDATE decisions SET worker_delta=?,engineer_delta=?,management_investment=?,"
                "production_volume=?,quality_investment=? WHERE company_id=? AND round_no=?",
                (
                    selected_plan["worker_delta"], selected_plan["engineer_delta"],
                    selected_plan["management"], selected_plan["production"],
                    selected_plan["quality"], company_id, round_no,
                ),
            )
            changed = True
        # Recalculate against the production changes above. If a Super Bot is
        # still buying materially more CPI units than it can stock, taper all
        # three investment indices together. Damping avoids a threshold cliff;
        # later passes let rivals absorb the released CPI before adjusting
        # again. MA/QI keep their documented minimum index of 1.
        capacities = _forecast_submitted_cpi_capacity(conn, round_no, markets)
        for row in all_rows(
            conn,
            "SELECT c.id,c.product_inventory,d.worker_delta,d.engineer_delta,d.management_investment,"
            "d.production_volume,d.quality_investment FROM companies c JOIN decisions d "
            "ON d.company_id=c.id WHERE c.is_super_bot=1 AND d.round_no=?",
            (round_no,),
        ):
            company_id = int(row["id"])
            if company_id in preserve_ids:
                continue
            available = max(0.0, float(row["product_inventory"] or 0) + float(row["production_volume"] or 0))
            capacity = capacities.get(company_id, 0.0)
            if available <= 0 or capacity <= available * 1.06:
                continue
            coverage = capacity / available
            scale = max(0.18, min(0.96, math.sqrt(1.0 / coverage)))
            workers = max(0, employee_count(conn, company_id, "worker") + int(row["worker_delta"] or 0))
            engineers = max(0, employee_count(conn, company_id, "engineer") + int(row["engineer_delta"] or 0))
            headcount = max(1, workers + engineers)
            ma_index = float(row["management_investment"] or 0) / headcount
            qi_denominator = max(
                1.0,
                float(row["product_inventory"] or 0) * 1.2 + float(row["production_volume"] or 0),
            )
            qi_index = float(row["quality_investment"] or 0) / qi_denominator
            conn.execute(
                "UPDATE decisions SET management_investment=?,quality_investment=? "
                "WHERE company_id=? AND round_no=?",
                (
                    max(1.0, ma_index * scale) * headcount,
                    max(1.0, qi_index * scale) * qi_denominator,
                    company_id,
                    round_no,
                ),
            )
            conn.execute(
                "UPDATE city_decisions SET marketing_investment=marketing_investment*? "
                "WHERE company_id=? AND round_no=? AND marketing_investment>0",
                (scale, company_id, round_no),
            )
            changed = True
        if not changed:
            break

    # A joint response may invalidate the sales that funded an early research
    # plan. Recheck the actual final field; never retain an 8M patent purchase
    # from a superseded sell-out forecast after production was cut to one group.
    live = _forecast_submitted_market(conn, round_no, markets)["companies"]
    funded = preview_round_operations(conn, round_no)
    for row in all_rows(
        conn, "SELECT c.id,c.cash,c.home_city,d.loan_change,d.research_investment "
        "FROM companies c JOIN decisions d ON d.company_id=c.id "
        "WHERE c.is_super_bot=1 AND d.round_no=? AND d.research_investment>0",
        (round_no,),
    ):
        company_id = int(row["id"])
        state = funded[company_id]
        sales = live[company_id].get("sold_units", {})
        revenue = sum(float(units) * float(state["city_decisions"][city]["price"])
                      for city, units in sales.items())
        transport = sum(float(units) * transportation_cost for city, units in sales.items()
                        if city != str(row["home_city"]))
        committed = float(row["cash"]) + float(row["loan_change"]) - float(state["cash_pre_sales"])
        research = float(row["research_investment"])
        if revenue - transport - committed < research * 1.25:
            conn.execute(
                "UPDATE decisions SET research_investment=0 WHERE company_id=? AND round_no=?",
                (company_id, round_no),
            )

    if round_no >= 1 and official_round >= max(1, total_rounds - 1):
        _recheck_late_disruptor_drafts(conn, round_no, markets, transportation_cost)
    if round_no > 0 and contest_round:
        _recheck_first_place_draft(conn, round_no)


@contextmanager
def _bot_trial_database(conn: sqlite3.Connection):
    # Read the current transaction, without carrying over its WAL header or
    # committing a speculative Bot onto the authoritative database.
    snapshot = bytearray(conn.serialize())
    snapshot[18:20] = b"\x01\x01"
    with closing(sqlite3.connect(":memory:")) as trial:
        trial.row_factory = sqlite3.Row
        trial.deserialize(bytes(snapshot))
        yield trial


def _save_checked_bot_draft(conn: sqlite3.Connection, trial: sqlite3.Connection,
                            company_id: int, round_no: int) -> None:
    for table in ("decisions", "city_decisions"):
        rows = all_rows(trial, f"SELECT * FROM {table} WHERE company_id=? AND round_no=?",
                        (company_id, round_no))
        if not rows:
            continue
        columns = ", ".join(rows[0].keys())
        placeholders = ",".join("?" for _ in rows[0].keys())
        conn.execute(f"DELETE FROM {table} WHERE company_id=? AND round_no=?", (company_id, round_no))
        conn.executemany(f"INSERT INTO {table}({columns}) VALUES({placeholders})",
                         [tuple(values) for values in rows])


def _recheck_first_place_draft(conn: sqlite3.Connection, round_no: int) -> None:
    """Rerun just one contender against all saved drafts and check exact rank."""
    if one(conn, "SELECT 1 FROM results WHERE round_no=?", (round_no,)):
        return
    contenders = {row["id"] for row in all_rows(conn,
        "SELECT c.id FROM companies c JOIN decisions d ON d.company_id=c.id AND d.round_no=? "
        "WHERE c.is_bot=1 AND c.is_super_bot=1 AND c.is_disruptor_bot=0 AND d.is_draft=1", (round_no,))}
    if not contenders:
        return
    from .engine import settle_round

    def exact_results(database):
        with _bot_trial_database(database) as settled:
            # Worker snapshots deliberately omit rollback snapshots and bonus
            # records; opening cash already includes the credited bonus.
            settled.execute("CREATE TABLE IF NOT EXISTS round_snapshots("
                            "round_no INTEGER PRIMARY KEY,snapshot_json TEXT,created_at TEXT)")
            settled.execute("CREATE TABLE IF NOT EXISTS round_bonuses("
                            "round_no INTEGER,company_id INTEGER,amount REAL)")
            settled.execute("UPDATE decisions SET is_draft=0 WHERE round_no=?", (round_no,))
            settle_round(settled, round_no)
            return all_rows(settled, "SELECT company_id,net_assets,net_profit,sales_revenue,sold,inventory FROM results WHERE round_no=? "
                            "ORDER BY net_assets DESC,company_id", (round_no,))

    def score(rows, company_id):
        own = next(row for row in rows if row["company_id"] == company_id)
        gap = float(own["net_assets"]) - max(float(row["net_assets"]) for row in rows
                                            if row["company_id"] != company_id)
        return gap, float(own["net_profit"]) - .08 * float(own["sales_revenue"])

    # Any absent drafts imply this is a partial local job, not a full field.
    if one(conn, "SELECT 1 FROM companies c WHERE NOT EXISTS (SELECT 1 FROM decisions d "
                 "WHERE d.company_id=c.id AND d.round_no=? AND d.submitted_at IS NOT NULL)", (round_no,)):
        return
    baseline = exact_results(conn)
    if len(baseline) < 2:
        return
    leader_id = int(baseline[0]["company_id"])
    human_present = one(conn, "SELECT 1 FROM companies WHERE is_bot=0")
    if human_present and one(conn, "SELECT 1 FROM companies WHERE id=? AND is_bot=1", (leader_id,)):
        return
    company_id = next((int(row["company_id"]) for row in baseline[1:] if row["company_id"] in contenders), None)
    if company_id is None:
        return
    old_gap, _ = score(baseline, company_id)
    final_round = round_no >= get_setting(conn, "total_rounds", 7, int)

    def trailing_asset_ratio(rows):
        trailing = sorted(float(row["net_assets"]) for row in rows[1:])
        if len(trailing) < 2:
            return 1.
        median = (trailing[(len(trailing) - 1) // 2] + trailing[len(trailing) // 2]) / 2
        return trailing[0] / max(1., median)

    min_asset_ratio = min(.20, trailing_asset_ratio(baseline))
    solvent_ids = {row["company_id"] for row in baseline if row["net_assets"] > 0}
    max_unsold = {
        row["company_id"]: max(.08, row["inventory"] / max(1, row["sold"] + row["inventory"]))
        for row in baseline
    }

    def score_trial(database):
        rows = exact_results(database)
        gap, risk = score(rows, company_id)
        if not human_present:
            # Let the leader pull away. Check that the teams behind it
            # stay competitive and that the move does not bankrupt a peer.
            if trailing_asset_ratio(rows) < min_asset_ratio - 1e-9 or any(
                row["company_id"] in solvent_ids and row["net_assets"] <= 0 for row in rows
            ):
                return gap, -math.inf
            if final_round and any(
                row["inventory"] / max(1, row["sold"] + row["inventory"]) > max_unsold[row["company_id"]] + 1e-9
                for row in rows
            ):
                return gap, -math.inf
        return gap, risk

    with _bot_trial_database(conn) as trial:
        _submit_bots(trial, round_no, True, replace_existing=True, target_ids={company_id},
                     defer_rebalance=True, use_complete_field=True, first_place_override=company_id)
        new_gap, new_risk = score_trial(trial)
        improved = new_gap > old_gap + 1e-6 and new_risk >= 0
        if final_round:
            if not improved:
                _save_checked_bot_draft(trial, conn, company_id, round_no)
                new_gap = old_gap
            new_gap = _refine_final_rank_draft(trial, round_no, company_id, leader_id, new_gap, score_trial)
            improved = new_gap > old_gap + 1e-6
        if improved:
            _save_checked_bot_draft(conn, trial, company_id, round_no)
            conn.commit()


def _refine_final_rank_draft(
    trial: sqlite3.Connection, round_no: int, company_id: int, leader_id: int,
    best_gap: float, evaluate: Callable[[sqlite3.Connection], tuple[float, float]],
) -> float:
    """Search a bounded neighborhood of the final draft using actual settlement."""
    key = (round_no, company_id)
    price_min = max(0., get_setting(trial, "price_min", 3500.))
    global_cap = get_setting(trial, "price_max", 25000.)
    growth = get_setting(trial, "market_growth", 1.10) ** max(0, round_no - 1)
    routes = all_rows(trial,
        "SELECT d.city,m.max_price,l.price AS opponent_price, "
        "COALESCE(h.count,0)+COALESCE(l.agent_delta,0) AS opponent_agents, "
        "COALESCE(a.count,0)+d.agent_delta AS own_agents, "
        "m.population*m.penetration AS market_size "
        "FROM city_decisions d JOIN market_config m ON m.city=d.city "
        "LEFT JOIN city_decisions l ON l.round_no=d.round_no AND l.city=d.city AND l.company_id=? "
        "LEFT JOIN agents a ON a.city=d.city AND a.company_id=d.company_id "
        "LEFT JOIN agents h ON h.city=d.city AND h.company_id=l.company_id "
        "WHERE d.round_no=? AND d.company_id=? AND COALESCE(a.count,0)+d.agent_delta>0 "
        "ORDER BY opponent_agents>0 DESC,market_size DESC,d.city LIMIT 6", (leader_id, *key))

    def consider(sql, params):
        nonlocal best_gap
        trial.execute("SAVEPOINT rank_probe")
        try:
            trial.execute(sql, params)
            gap, risk = evaluate(trial)
            if gap > best_gap + 1e-6 and risk >= 0:
                best_gap = gap
            else:
                trial.execute("ROLLBACK TO rank_probe")
        except BaseException:
            trial.execute("ROLLBACK TO rank_probe")
            raise
        finally:
            trial.execute("RELEASE rank_probe")

    if not routes:
        return best_gap
    # Exact thresholds and payroll can make score rankings move after the
    # forecast. Changing one route at a time keeps improvements already found.
    starting_bid = one(trial, "SELECT management_investment,quality_investment FROM decisions "
                       "WHERE round_no=? AND company_id=?", key)
    for ma_scale, qi_scale in ((.5, .5), (1.5, 1.), (1., 1.75), (.25, 1.), (1., .25)):
        consider("UPDATE decisions SET management_investment=?,quality_investment=? "
                 "WHERE round_no=? AND company_id=?",
                 (starting_bid["management_investment"] * ma_scale,
                  starting_bid["quality_investment"] * qi_scale, *key))
    for route in routes:
        city = route["city"]
        old = one(trial, "SELECT price,marketing_investment FROM city_decisions "
                  "WHERE round_no=? AND company_id=? AND city=?", (*key, city))
        cap = min(global_cap, float(route["max_price"]))
        price = float(old["price"])
        anchor = float(route["opponent_price"] or price) if route["opponent_agents"] > 0 else price
        quotes = dict.fromkeys(min(cap, max(price_min, bid)) for bid in
                               (cap, price * .97, price * .86, anchor - 1., anchor * .91))
        for quote in quotes:
            if abs(quote - price) > 1e-6:
                consider("UPDATE city_decisions SET price=? WHERE round_no=? AND company_id=? AND city=?",
                         (quote, *key, city))
        threshold = float(route["max_price"]) / 50 * route["market_size"] * growth * .20
        threshold /= 3 * (1 + route["own_agents"] * .10)
        for bid in dict.fromkeys((0., max(float(old["marketing_investment"]) * 2., threshold * 2.))):
            if abs(bid - float(old["marketing_investment"])) > 1e-6:
                consider("UPDATE city_decisions SET marketing_investment=? "
                         "WHERE round_no=? AND company_id=? AND city=?", (bid, *key, city))
    return best_gap


def _recheck_late_disruptor_drafts(
    conn: sqlite3.Connection,
    round_no: int,
    markets: list[dict[str, Any]],
    transportation_cost: float,
) -> None:
    """Repair clearly unaffordable attacks with the completed field in view."""
    drafts = all_rows(
        conn,
        "SELECT c.id,c.cash,c.debt,c.home_city,m.interest_rate "
        "FROM decisions d JOIN companies c ON c.id=d.company_id "
        "JOIN market_config m ON m.city=c.home_city "
        "WHERE d.round_no=? AND d.is_draft=1 AND c.is_disruptor_bot=1 ORDER BY c.id",
        (round_no,),
    )
    if not drafts:
        return
    conn.commit()

    def scores(database: sqlite3.Connection) -> dict[int, tuple[float, int]]:
        field = _forecast_submitted_market(database, round_no, markets)["companies"]
        funded = preview_round_operations(database, round_no)
        values = {}
        loans = {
            int(plan["company_id"]): float(plan["loan_change"])
            for plan in all_rows(database,
                "SELECT company_id,loan_change FROM decisions WHERE round_no=?", (round_no,))
        }
        for row in drafts:
            company_id = int(row["id"])
            if company_id not in field:
                continue
            state = funded[company_id]
            units = field[company_id]["sold_units"]
            proceeds = sum(
                count * (float(state["city_decisions"][city]["price"])
                         - (transportation_cost if city != row["home_city"] else 0.0))
                for city, count in units.items()
            )
            loan = loans[company_id]
            spent = float(row["cash"]) + loan - float(state["cash_pre_sales"])
            interest = (float(row["debt"]) + loan) * float(row["interest_rate"])
            values[company_id] = proceeds - spent - interest, sum(units.values())
        return values

    field = scores(conn)
    for row in drafts:
        company_id = int(row["id"])
        profit, sold = field.get(company_id, (0.0, 0))
        if profit >= -max(1_000., float(row["cash"]) * 0.02):
            continue
        # Reanalysis commits each bot separately. Do that in a private copy
        # so a worse draft or an interrupted repair never replaces the old one.
        with _bot_trial_database(conn) as trial:
            _submit_bots(
                trial, round_no, True, replace_existing=True, target_ids={company_id},
                defer_rebalance=True, use_complete_field=True,
            )
            next_field = scores(trial)
            next_profit, next_sold = next_field.get(company_id, (-math.inf, 0))
            # A small squeeze is expected, but do not convert a competitive
            # plan to zero shipments to save pennies, or move the same loss
            # onto a second disruptor without improving the total.
            previous_loss = sum(min(0.0, outcome[0]) for outcome in field.values())
            next_loss = sum(min(0.0, outcome[0]) for outcome in next_field.values())
            if (next_profit <= profit or next_loss < previous_loss
                    or (next_sold < sold * .75 and next_profit < 0)):
                continue
            _save_checked_bot_draft(conn, trial, company_id, round_no)
            field = next_field
            conn.commit()


def _submit_bots(
    conn: sqlite3.Connection,
    round_no: int,
    super_mode: bool,
    progress_callback: Callable[[int, int, str], None] | None = None,
    replace_existing: bool = False,
    *,
    target_ids: set[int] | None = None,
    defer_rebalance: bool = False,
    use_complete_field: bool = False,
    first_place_override: int | None = None,
) -> int:
    bots = all_rows(
        conn,
        "SELECT * FROM companies WHERE is_bot=1 AND is_super_bot=? ORDER BY id",
        (int(super_mode),),
    )
    if super_mode and bots:
        # The order is a game rule: before the attack window strategic Super
        # Bots move first and disruptors shadow them; in the last two official
        # rounds disruptors commit first so the strategic models can react.
        from .ai_training import super_bot_analysis_order

        order = list(super_bot_analysis_order(conn, round_no))
        order_index = {company_id: index for index, company_id in enumerate(order)}
        bots = sorted(
            bots,
            key=lambda row: (order_index.get(int(row["id"]), len(order)), int(row["id"])),
        )
    disruptor_ids = [
        int(row["id"])
        for row in bots
        if int(row["is_disruptor_bot"] or 0)
    ] if super_mode else []
    floor_lock_company_id = min(disruptor_ids, default=0)
    markets = [dict(row) for row in all_rows(conn, "SELECT * FROM market_config ORDER BY city")]
    if not bots or not markets:
        return 0

    official_round = 1 if round_no < 0 else round_no
    all_markets_full = _all_markets_near_capacity(conn, round_no, markets)
    plan_index = min(6, max(0, official_round - 1))
    plan = BOT_PLANS[plan_index]
    setting = lambda key, default: float(get_setting(conn, key, default))
    salary_min, salary_max = setting("salary_min", 1000), setting("salary_max", 10000)
    salary_change = setting("salary_change_limit", 1000)
    price_min, price_max = setting("price_min", 3500), setting("price_max", 25000)
    price_power = max(1, int(setting("cpi_price_power", 8)))
    max_agent_add = max(0, int(setting("max_agent_add_per_city_round", 3)))
    growth = setting("market_growth", 1.10)
    worker_need, worker_hours = setting("component_workers", 3), setting("component_hours", 7)
    engineer_need, engineer_hours = setting("product_engineers", 4), setting("product_hours", 14)
    component_need = max(1, int(round(setting("components_per_product", 7))))
    worker_training = setting("worker_training_cost", 0)
    engineer_training = setting("engineer_training_cost", 0)
    add_agent_cost = setting("agent_add_cost", 300000)
    remove_agent_cost = setting("agent_remove_cost", 100000)
    transport_cost = setting("transport_cost", 0)
    patent_factor = setting("patent_factor", 0.70)
    tax_rate = min(1.0, max(0.0, setting("tax_rate", 0.20)))
    ma_threshold = setting("cpi_ma_large_threshold", 1300)
    qi_safe_multiplier = max(1.0, setting("qi_safe_multiplier", 1.10))
    initial_cash = max(1.0, setting("initial_cash", 15_000_000))
    total_rounds = max(1, int(setting("total_rounds", 5)))
    loan_threshold = setting("loan_asset_threshold", 15_000_000)
    global_max_loan = setting("global_max_loan", 10_000_000)
    research_goal = setting("research_75", 6000000) * setting("research_hidden_threshold_multiplier", 4 / 3)
    research_goal += setting("research_buffer", 150000)

    human_present = bool(one(conn, "SELECT 1 FROM companies WHERE is_bot=0 LIMIT 1"))
    all_super_field = bool(super_mode) and not bool(one(
        conn,
        "SELECT 1 FROM companies WHERE is_super_bot=0 LIMIT 1",
    ))
    benchmark_round_row = one(
        conn,
        "SELECT MAX(round_no) AS round_no FROM results WHERE round_no>=1 AND round_no<?",
        (max(1, round_no),),
    )
    benchmark_round = int(benchmark_round_row["round_no"] or 0) if benchmark_round_row else 0
    ranked_rows = [dict(row) for row in all_rows(
        conn,
        "SELECT c.id,c.is_bot,c.is_super_bot,c.is_disruptor_bot,"
        "r.net_assets,r.net_profit,r.sales_revenue "
        "FROM companies c JOIN results r ON r.company_id=c.id "
        "WHERE r.round_no=? ORDER BY r.net_assets DESC,c.id",
        (benchmark_round,),
    )] if benchmark_round else []
    rank_by_company = {
        int(row["id"]): index + 1 for index, row in enumerate(ranked_rows)
    }
    prior_margin_by_company = {
        int(row["id"]): (
            float(row["net_profit"] or 0.0)
            / max(1.0, float(row["sales_revenue"] or 0.0))
        )
        for row in ranked_rows
    }
    attack_window = round_no > 0 and official_round >= max(1, total_rounds - 1)
    ranked_super_ids = [
        int(row["id"]) for row in ranked_rows
        if int(row["is_super_bot"] or 0) and not (attack_window and int(row["is_disruptor_bot"] or 0))
    ]
    ranked_super_ids.extend(
        int(bot["id"]) for bot in bots
        if not (attack_window and int(bot["is_disruptor_bot"] or 0))
        and int(bot["id"]) not in ranked_super_ids
    )
    competitive_super_ids = (
        _competitive_super_attackers(
            ranked_super_ids,
            official_round=round_no,
            total_rounds=total_rounds,
            markets_saturated=all_markets_full,
        )
        if super_mode else set()
    )
    previous_leader: dict[str, Any] | None = None
    if super_mode and human_present and benchmark_round:
        leader_row = one(
            conn,
            "SELECT r.company_id,r.ma_index,r.qi_index FROM results r "
            "JOIN companies c ON c.id=r.company_id "
            "WHERE r.round_no=? AND c.is_bot=0 ORDER BY r.net_assets DESC,r.company_id LIMIT 1",
            (benchmark_round,),
        )
        if leader_row:
            previous_leader = {
                "ma": float(leader_row["ma_index"] or 0),
                "qi": float(leader_row["qi_index"] or 0),
                "cities": {
                    str(row["city"]): {
                        "marketing": float(row["marketing"] or 0),
                        "price": float(row["price"] or 0),
                    }
                    for row in all_rows(
                        conn,
                        "SELECT city,marketing,price FROM city_results "
                        "WHERE company_id=? AND round_no=? AND (cpi>0 OR sold>0 OR marketing>0)",
                        (leader_row["company_id"], benchmark_round),
                    )
                },
            }

    standard_ids = [bot_id for bot_id in ranked_super_ids if bot_id not in disruptor_ids]
    benchmark_bot_ids = set(standard_ids[:3] if human_present else ranked_super_ids) if super_mode else set()
    first_place_bot = first_place_override or next(iter(standard_ids), None)
    if first_place_override is not None:
        benchmark_bot_ids.add(first_place_override)
    human_ids = {int(row["id"]) for row in all_rows(conn, "SELECT id FROM companies WHERE is_bot=0")} if human_present else set()
    ordinary_seed = "" if super_mode else get_setting(conn, "ordinary_bot_match_seed", "", str)
    playbook_order = _ordinary_playbook_order(ordinary_seed)
    selected_by_bot: dict[int, list[int]] = {}
    for slot, bot in enumerate(bots):
        wealth_multiple = max(1.0, float(bot["cash"]) / initial_cash)
        # Profitable Bots must deploy their growing capital by opening more
        # real sales capacity. A slow, fixed three-city wealth cap was the main
        # reason late-round Bots kept hundreds of millions idle.
        wealth_expansion = min(
            len(markets),
            max(0, int(math.ceil(math.log2(wealth_multiple)))),
        )
        market_count = int(plan["markets"]) + wealth_expansion + (2 if super_mode else 0)
        late_market_expansion_round = max(3, math.ceil(total_rounds * 0.65))
        if super_mode and official_round >= late_market_expansion_round:
            # Late Super Bots compete everywhere. Leaving cities unopened
            # wastes market capacity and the independent per-city MI route.
            market_count = len(markets)
        elif official_round >= total_rounds:
            market_count += 2
        if not super_mode:
            role_slot = playbook_order[slot % len(playbook_order)]
            _, _, _, stance, pace, _, breadth = _ordinary_mix(role_slot)
            # Multi-market plays deploy surplus capital sooner; specialists
            # can continue concentrating in fewer cities on the same budget.
            market_count += breadth
            market_count += math.floor(wealth_expansion * pace) - wealth_expansion
            if stance == "premium":
                market_count = max(1, market_count - 1)
        selected_by_bot[int(bot["id"])] = _selected_markets(bot, markets, min(len(markets), market_count))
        if previous_leader and int(bot["id"]) in benchmark_bot_ids:
            # Pick up the leader's real prior markets as well as our own home.
            for index, market in enumerate(markets):
                city = str(market["city"])
                if city in previous_leader["cities"] and index not in selected_by_bot[int(bot["id"])]:
                    selected_by_bot[int(bot["id"])].append(index)
    city_competitors: dict[int, int] = {}
    for index, market in enumerate(markets):
        seller_ids = {
            int(row["company_id"])
            for row in all_rows(conn, "SELECT company_id FROM agents WHERE city=? AND count>0", (market["city"],))
        }
        for row in all_rows(
            conn,
            "SELECT cd.company_id,COALESCE(a.count,0)+cd.agent_delta AS agents_after "
            "FROM city_decisions cd LEFT JOIN agents a ON a.company_id=cd.company_id AND a.city=cd.city "
            "WHERE cd.round_no=? AND cd.city=?",
            (round_no, market["city"]),
        ):
            if int(row["agents_after"] or 0) > 0:
                seller_ids.add(int(row["company_id"]))
            else:
                seller_ids.discard(int(row["company_id"]))
        seller_ids.update(company_id for company_id, indices in selected_by_bot.items() if index in indices)
        city_competitors[index] = max(1, len(seller_ids))

    submitted = 0
    pending_total = sum(
        1 for bot in bots
        if (target_ids is None or int(bot["id"]) in target_ids)
        and (replace_existing or not one(
            conn,
            "SELECT 1 FROM decisions WHERE company_id=? AND round_no=?",
            (int(bot["id"]), round_no),
        ))
    )
    progress_total = pending_total + (1 if super_mode and pending_total and not defer_rebalance else 0)
    # Later Super Bots see decisions completed earlier in this pass, while
    # unresolved peers stay synthetic. This preserves resumability and gives
    # later positions a real opportunity to counter or deeply undercut.
    frozen_super_ids: set[int] = {
        int(saved["id"])
        for saved in bots
        if one(
            conn,
            "SELECT 1 FROM decisions WHERE company_id=? AND round_no=?",
            (int(saved["id"]), round_no),
        )
        and not (
            replace_existing
            and (target_ids is None or int(saved["id"]) in target_ids)
        )
    }
    order_position = {int(row["id"]): index for index, row in enumerate(bots)}
    for bot_position, bot_row in enumerate(bots):
        bot = dict(bot_row)
        company_id = int(bot["id"])
        if target_ids is not None and company_id not in target_ids:
            continue
        if super_mode and replace_existing and not use_complete_field:
            # Same inputs for full and split reanalysis: earlier saved plans,
            # synthetic later peers. Never delete the old saved drafts here.
            frozen_super_ids = {
                int(saved["id"]) for saved in bots
                if order_position[int(saved["id"])] < bot_position and one(
                    conn, "SELECT 1 FROM decisions WHERE company_id=? AND round_no=?",
                    (int(saved["id"]), round_no),
                )
            }
        existing_decision = one(
            conn,
            "SELECT submitted_at,is_draft FROM decisions WHERE company_id=? AND round_no=?",
            (company_id, round_no),
        )
        if existing_decision and not replace_existing:
            continue
        is_disruptor = bool(super_mode and int(bot["is_disruptor_bot"] or 0))
        late_disruptor = bool(
            is_disruptor
            and round_no >= 1
            and official_round >= max(1, total_rounds - 1)
        )
        floor_lock_disruptor = bool(
            late_disruptor and company_id == floor_lock_company_id
        )
        competitive_mode = bool(
            super_mode
            and (late_disruptor or company_id in competitive_super_ids)
        )
        contest_start = 4 if human_present else min(4, total_rounds)
        race_for_first = bool(company_id == first_place_bot and round_no >= contest_start and not late_disruptor)
        profile = int(bot["bot_profile"] if bot["bot_profile"] is not None else company_id) % 7
        normal_mix = _ordinary_mix(playbook_order[bot_position % len(playbook_order)])
        normal_ma, normal_qi, normal_mi, normal_price_stance, normal_pace, normal_mi_start, _ = normal_mix
        profit_target = 0.0
        if super_mode and not late_disruptor:
            previous_super_id = next(
                (
                    int(previous["id"])
                    for previous in reversed(bots[:bot_position])
                    if not (attack_window and int(previous["is_disruptor_bot"] or 0))
                ),
                None,
            )
            profit_target = _super_profit_target(
                conn, company_id, round_no, previous_super_id,
            )
        chosen_predicted_profit: float | None = None
        chosen_risk_profit = 0.0
        chosen_sold = 0.0
        style = BOT_STYLES[profile]
        rng = _bot_rng(bot, round_no, super_mode, ordinary_seed)
        variation = rng.uniform(0.88, 1.14)
        ma_strength = min(4.25, max(0.62, float(style["ma"]) * rng.uniform(0.84, 1.20)))
        qi_strength = min(3.40, max(0.78, float(style["qi"]) * rng.uniform(0.86, 1.18)))
        mi_strength = min(3.60, max(1.03, float(style["mi"]) * rng.uniform(0.88, 1.16)))
        high_price_ratio = min(0.982, max(0.925, float(style["high"]) + rng.uniform(-0.014, 0.014)))
        if not super_mode and normal_price_stance == "premium":
            high_price_ratio = rng.uniform(.975, 1.0)
        price_cut = max(50.0, float(style["cut"]) * rng.uniform(0.72, 1.32))
        ma_round_buffer = rng.uniform(0.96, 1.10)
        super_aggression = rng.uniform(1.12, 1.34)
        super_mi_cap = rng.uniform(2.80, 3.60)
        sequence_strength = 1.0 + 0.18 * bot_position / max(1, len(bots) - 1)
        if super_mode:
            # Later Super Bots have observed more real decisions. Convert that
            # information advantage into a wider, still capped response range.
            super_aggression *= sequence_strength
            super_mi_cap = min(6.0, super_mi_cap * sequence_strength)
        normal_price_factor = rng.uniform(0.978, 0.994)
        aggressive_price_factor = rng.uniform(0.935, 0.968)
        home = str(bot["home_city"] or markets[profile % len(markets)]["city"])
        home_market = next((market for market in markets if market["city"] == home), markets[0])
        loan_change = 0.0
        if super_mode:
            loan_ceiling = loan_ceiling_for_round(round_no, home_market, global_max_loan)
            loan_change = available_loan_limit(
                current_company_net_assets(conn, bot),
                loan_threshold,
                float(home_market.get("min_loan", 0.0)),
                loan_ceiling,
            )
        report = _previous_report(conn, company_id, round_no)
        previous = one(
            conn,
            "SELECT d.worker_salary,d.engineer_salary FROM decisions d JOIN results r "
            "ON r.company_id=d.company_id AND r.round_no=d.round_no WHERE d.company_id=? "
            "AND d.round_no>=1 AND d.round_no<? ORDER BY d.round_no DESC LIMIT 1",
            (company_id, max(1, round_no)),
        )
        previous_worker = float(previous["worker_salary"]) if previous else float(home_market["worker_initial_salary"])
        previous_engineer = float(previous["engineer_salary"]) if previous else float(home_market["engineer_initial_salary"])
        worker_salary, worker_multiplier = _bot_salary(
            report, "worker", float(home_market["worker_initial_salary"]), previous_worker,
            rng.uniform(0.0, 6.0), salary_min, salary_max, salary_change,
        )
        engineer_salary, engineer_multiplier = _bot_salary(
            report, "engineer", float(home_market["engineer_initial_salary"]), previous_engineer,
            rng.uniform(0.0, 6.0), salary_min, salary_max, salary_change,
        )

        selected = selected_by_bot[company_id]

        agent_plan: dict[int, tuple[int, int]] = {}
        agent_cost = 0.0
        for index, market in enumerate(markets):
            row = one(conn, "SELECT count FROM agents WHERE company_id=? AND city=?", (company_id, market["city"]))
            current = int(row["count"] if row else 0)
            agent_variation = rng.choice((-1, 0, 0, 0, 1))
            desired = max(1, int(plan["agents"]) + agent_variation) if index in selected else current
            delta = max(-current, min(max_agent_add, desired - current))
            if super_mode:
                # Never pay to remove a 10%-MI amplifier or abandon a city.
                delta = max(0, delta)
            agent_plan[index] = (delta, current + delta)
            if delta > 0:
                agent_cost += delta * add_agent_cost
            elif delta < 0:
                agent_cost += -delta * remove_agent_cost

        saturated: dict[int, bool] = {}
        utilization: dict[int, float] = {}
        previous_prices: dict[int, float] = {}
        for index, market in enumerate(markets):
            stats = one(
                conn,
                "SELECT market_size,player_total_volume,average_price FROM market_round_stats "
                "WHERE city=? AND round_no>=1 AND round_no<? ORDER BY round_no DESC LIMIT 1",
                (market["city"], max(1, round_no)),
            )
            utilization[index] = (
                float(stats["player_total_volume"] or 0) / max(1.0, float(stats["market_size"] or 0))
                if stats else 0.0
            )
            saturated[index] = utilization[index] >= 0.60
            previous_prices[index] = float(stats["average_price"]) if stats else float(market["initial_avg_price"])

        late_game_low_price = official_round >= max(1, total_rounds - 1)
        saturated_low_price_indices = {
            index for index in selected if utilization.get(index, 0.0) >= 0.95
        }

        old_products = int(bot["product_inventory"] or 0)
        old_components = int(bot["component_inventory"] or 0)
        own_strategy_assets = max(1.0, current_company_net_assets(conn, bot))
        wealth_multiple = max(1.0, float(bot["cash"]) / initial_cash)
        unsaturated_share = (
            sum(1 for index in selected if not saturated.get(index, False)) / max(1, len(selected))
        )
        # Fixed seven-round volumes are only the starting curve. When prior
        # markets still have room, profitable ordinary Bots compound production
        # with their capital instead of repeating the same small batch forever.
        capital_scale = 1.0 + (
            min(8.0, math.sqrt(wealth_multiple)) - 1.0
        ) * unsaturated_share
        round_market_ramp = (0.34, 0.52, 0.72, 0.88, 1.00, 1.08, 1.15)[plan_index]
        fair_market_units = sum(
            (
                float(markets[index]["population"])
                * float(markets[index]["penetration"])
                * growth ** max(0, official_round - 1)
            ) / max(1, city_competitors[index])
            for index in selected
        )
        # Production anchors scale with the live KDS market size and actual
        # field density. The seven-round table supplies only the strategy
        # phase; no historic event's raw unit count leaks into this KDS.
        desired_available = int(fair_market_units * round_market_ramp * variation * capital_scale)
        if not super_mode:
            desired_available = int(desired_available * normal_pace)
        prior_production = report.get("production", {})
        prior_sold = 0
        prior_total = 0
        prior_surplus = 0
        if prior_production:
            prior_sold = int(prior_production.get("sold", 0) or 0)
            prior_total = int(prior_production.get("old_products", 0) or 0) + int(prior_production.get("produced", 0) or 0)
            prior_surplus = int(prior_production.get("surplus", 0) or 0)
            if prior_total and prior_sold >= prior_total * 0.95 and not any(saturated.values()):
                desired_available = max(desired_available, int(prior_sold * 1.35))
            elif prior_surplus > max(20, prior_sold * 0.30):
                desired_available = min(desired_available, int(prior_sold * 1.12 + old_products))
        prior_surplus_ratio = prior_surplus / max(1, prior_total)
        cash_ratio = float(bot["cash"]) / initial_cash
        # Low price is an emergency tool, not the normal response to a small
        # amount of stock. Keep the profit strategy unless fewer than half of
        # last round's available products were sold.
        distress_liquidation = bool(
            not super_mode
            and prior_surplus > max(10, prior_total * 0.06)
            and prior_sold < prior_total * 0.50
        )
        if distress_liquidation:
            # Do not pay to expand the network while rescuing existing stock.
            # Existing agents remain in place and can sell the inventory.
            for index, (delta, agents_after) in tuple(agent_plan.items()):
                if delta > 0:
                    agent_plan[index] = (0, agents_after - delta)
            agent_cost = sum(
                max(0, delta) * add_agent_cost
                for delta, _ in agent_plan.values()
            )
        production_goal = max(0, desired_available - old_products)
        material_factor = patent_factor ** int(bot["patents"] or 0)
        research_balance = max(0.0, float(bot["research_balance"] or 0))
        research_needed = max(0.0, research_goal - research_balance)
        research = 0.0

        rival_metrics: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(markets))}
        if super_mode:
            # Plans can intentionally be far above capacity (the supplied game
            # has a 363M plan but only 425,907 real products). Cash-funded stock
            # and paid investments, rather than requested numbers, compete for
            # CPI and secondary allocation.
            funded_rivals = preview_round_operations(conn, round_no)
            submitted_rival_ids = {int(row["company_id"]) for row in all_rows(
                conn, "SELECT company_id FROM decisions WHERE round_no=? AND submitted_at IS NOT NULL AND company_id<>?",
                (round_no, company_id),
            )}
            benchmark_rival_ids = human_ids if human_present else {
                rival_id for rival_id in submitted_rival_ids
                if rival_id not in order_position or rival_id in frozen_super_ids
            }
            if human_present and company_id in benchmark_bot_ids and official_round >= 2:
                live_rival_cities = {
                    city for rival_id in benchmark_rival_ids
                    for city, route in funded_rivals.get(rival_id, {}).get("city_decisions", {}).items()
                    if route["agents_after"] > 0
                }
                cash_budget_for_city = max(0., float(bot["cash"]) + loan_change) * .05
                for index, market in enumerate(markets):
                    if str(market["city"]) not in live_rival_cities or index in selected:
                        continue
                    delta, current = agent_plan[index]
                    if current <= 0:
                        if not max_agent_add or cash_budget_for_city < add_agent_cost:
                            continue
                        agent_plan[index] = (delta + 1, current + 1)
                        agent_cost += add_agent_cost
                        cash_budget_for_city -= add_agent_cost
                    selected = [*selected, index]
            rival_asset_weights: dict[int, float] = {}
            own_rank = rank_by_company.get(company_id, len(rank_by_company) + 1)

            def attack_priority(
                rival_id: int,
                *,
                rival_is_bot: bool,
                rival_is_super: bool,
                index: int,
                mi_effective: float,
                price: float,
                predicted_margin: float,
            ) -> float:
                if not competitive_mode:
                    return 0.0
                rival_rank = rank_by_company.get(rival_id, len(rank_by_company) + 1)
                return float(_super_attack_priority(
                    is_human=not rival_is_bot, is_super=rival_is_super,
                    opponent_rank=rival_rank, own_rank=own_rank,
                    predicted_margin=predicted_margin,
                ))

            def rival_asset_weight(rival_id: int) -> float:
                cached = rival_asset_weights.get(rival_id)
                if cached is not None:
                    return cached
                latest_assets = one(
                    conn,
                    "SELECT net_assets FROM results WHERE company_id=? AND round_no>=1 "
                    "AND round_no<? ORDER BY round_no DESC LIMIT 1",
                    (rival_id, max(1, round_no)),
                )
                if latest_assets:
                    rival_assets = max(1.0, float(latest_assets["net_assets"] or 0))
                else:
                    rival_company = one(conn, "SELECT * FROM companies WHERE id=?", (rival_id,))
                    rival_assets = max(
                        1.0,
                        current_company_net_assets(conn, dict(rival_company))
                        if rival_company else 1.0,
                    )
                weight = min(3.0, max(0.35, math.sqrt(rival_assets / own_strategy_assets)))
                rival_asset_weights[rival_id] = weight
                return weight

            rival_rows = all_rows(
                conn,
                "SELECT d.company_id,d.worker_delta,d.engineer_delta,d.management_investment,d.production_volume,"
                "d.quality_investment,cd.city,cd.agent_delta,cd.marketing_investment,cd.price,c.product_inventory,"
                "c.is_bot AS rival_is_bot,c.is_super_bot AS rival_is_super,"
                "c.is_disruptor_bot AS rival_is_disruptor "
                "FROM decisions d JOIN companies c ON c.id=d.company_id "
                "JOIN city_decisions cd ON cd.company_id=d.company_id AND cd.round_no=d.round_no "
                "WHERE d.round_no=? AND d.submitted_at IS NOT NULL AND d.company_id<>?",
                (round_no, company_id),
            )
            market_index = {str(market["city"]): index for index, market in enumerate(markets)}
            for rival in rival_rows:
                if int(rival["rival_is_super"] or 0) and int(rival["company_id"]) not in frozen_super_ids:
                    continue
                index = market_index[str(rival["city"])]
                funded = funded_rivals[int(rival["company_id"])]
                funded_city = funded["city_decisions"][str(rival["city"])]
                agents = int(funded_city["agents_after"])
                if agents <= 0:
                    continue
                active_count = sum(int(city["agents_after"]) > 0 for city in funded["city_decisions"].values())
                ma_index = float(funded["ma_index"])
                qi_index = float(funded["qi_index"])
                available_per_city = float(funded["available"]) / max(1, active_count)
                mi_effective = float(funded_city["marketing_investment"]) * (1.0 + agents * 0.10)
                revenue_proxy = max(0.0, float(rival["price"] or 0)) * available_per_city
                fixed_proxy = (
                    float(rival["management_investment"] or 0)
                    + float(rival["quality_investment"] or 0)
                ) / max(1, active_count) + float(funded_city["marketing_investment"])
                # Use realized net margin (includes production/payroll/tax),
                # not revenue minus investments which labels losing Bots rich.
                predicted_margin = prior_margin_by_company.get(int(rival["company_id"]), 0.0)
                rival_metrics[index].append({
                    "company_id": float(rival["company_id"]),
                    "ma": ma_index,
                    "qi": qi_index,
                    "mi_effective": mi_effective,
                    "price": float(funded_city["price"]),
                    "agents": float(agents),
                    "is_human": not bool(rival["rival_is_bot"]),
                    "is_super": bool(rival["rival_is_super"]),
                    "is_disruptor": bool(rival["rival_is_disruptor"]),
                    "attack_weight": attack_priority(
                        int(rival["company_id"]),
                        rival_is_bot=bool(rival["rival_is_bot"]),
                        rival_is_super=bool(rival["rival_is_super"]),
                        index=index,
                        mi_effective=mi_effective,
                        price=float(rival["price"] or 0),
                        predicted_margin=predicted_margin,
                    ),
                    "strategic_weight": rival_asset_weight(int(rival["company_id"])),
                    "available": available_per_city,
                })

            # Forecast unresolved peers in the same way in segmented and
            # uninterrupted analysis. Saved peers already use their real plans.
            for other_row in bots:
                other = dict(other_row)
                other_id = int(other["id"])
                if other_id == company_id or other_id in frozen_super_ids:
                    continue
                # During the final attack window a disruptor acts only on the
                # live human/ordinary field and earlier saved disruptors.  It
                # must not make room for a hypothetical strategic Super plan
                # that has not been analysed yet.  Before that window the
                # strategic Super group likewise moves without anticipating
                # hidden disruptors; those disruptors then observe and avoid it.
                if late_disruptor and not use_complete_field:
                    continue
                other_profile = int(other["bot_profile"] if other["bot_profile"] is not None else other_id) % 7
                other_style = BOT_STYLES[other_profile]
                other_variation = 0.94 + other_profile * 0.02
                other_selected = selected_by_bot[other_id]
                prior_other = one(
                    conn,
                    "SELECT round_no,ma_index,qi_index,sold,inventory FROM results "
                    "WHERE company_id=? AND round_no>=1 AND round_no<? "
                    "ORDER BY round_no DESC LIMIT 1",
                    (other_id, max(1, round_no)),
                )
                prior_active = 1
                prior_cities: dict[str, sqlite3.Row] = {}
                if prior_other:
                    prior_active_row = one(
                        conn,
                        "SELECT COUNT(*) AS n FROM city_results WHERE company_id=? "
                        "AND round_no=?",
                        (other_id, int(prior_other["round_no"])),
                    )
                    prior_active = max(1, int(prior_active_row["n"] or 0))
                    prior_cities = {
                        str(row["city"]): row
                        for row in all_rows(
                            conn,
                            "SELECT city,price,marketing FROM city_results "
                            "WHERE company_id=? AND round_no=?",
                            (other_id, int(prior_other["round_no"])),
                        )
                    }
                for index in other_selected:
                    market = markets[index]
                    agent_row = one(
                        conn, "SELECT count FROM agents WHERE company_id=? AND city=?",
                        (other_id, market["city"]),
                    )
                    other_agents = max(1, int(agent_row["count"] if agent_row else 0))
                    size = float(market["population"]) * float(market["penetration"]) * growth ** max(0, official_round - 1)
                    qi_large = float(market["max_price"]) / 50.0
                    mi_large = qi_large * size * 0.20 / 1.5 / 2.0
                    cap = min(price_max, float(market["max_price"]))
                    prior_average = previous_prices.get(index, float(market["initial_avg_price"]))
                    prior_city = prior_cities.get(str(market["city"]))
                    if (
                        prior_average >= cap * 0.75
                        or late_game_low_price
                        or index in saturated_low_price_indices
                        or all_super_field
                    ):
                        # Unresolved Super Bots must be modelled as possible
                        # low-price competitors once that strategy is unlocked.
                        # Otherwise every sequential Bot falsely believes it can
                        # monopolise the 40-CPI price pool at exactly 75%.
                        # Model unresolved rivals in price-gap/CPI-weight
                        # space.  A fixed percentage of max price is wildly
                        # wrong when one KDS uses power 8 and another uses 20.
                        base_depths = (0.18, 0.26, 0.34, 0.43, 0.53, 0.64, 0.76)
                        depth = _kds_price_depth(base_depths[other_profile], price_power)
                        expected_price = prior_average - (
                            prior_average - price_min
                        ) * depth
                        if prior_city and float(prior_city["price"] or 0) > 0:
                            # The unresolved peer starts from its proven route,
                            # then applies the current KDS-scaled pressure depth.
                            proven = min(cap, max(price_min, float(prior_city["price"])))
                            expected_price = min(expected_price, proven)
                        expected_price = min(cap, max(price_min, expected_price))
                    else:
                        expected_price = cap * float(other_style["high"])
                    default_ma = (
                        ma_threshold
                        * (1.02 + plan_index * 0.16)
                        * other_variation
                        * float(other_style["ma"])
                    )
                    default_qi = qi_large * float(other_style["qi"]) if plan_index >= 1 else 0.0
                    default_mi = mi_large * float(other_style["mi"]) if plan_index >= 2 else 0.0
                    predicted_available = (
                        size
                        / max(1, city_competitors[index])
                        * round_market_ramp
                        * other_variation
                    )
                    if prior_other:
                        default_ma = max(default_ma, float(prior_other["ma_index"] or 0) * 1.03)
                        default_qi = max(default_qi, float(prior_other["qi_index"] or 0) * 1.03)
                        predicted_available = max(
                            predicted_available,
                            (
                                float(prior_other["sold"] or 0)
                                + float(prior_other["inventory"] or 0)
                            ) / prior_active * growth,
                        )
                    if prior_city:
                        default_mi = max(
                            default_mi,
                            float(prior_city["marketing"] or 0)
                            * (1.0 + other_agents * 0.10)
                            * 1.03,
                        )
                    rival_metrics[index].append({
                        "company_id": float(other_id),
                        "ma": default_ma,
                        "qi": default_qi,
                        "mi_effective": default_mi,
                        "price": expected_price,
                        "agents": float(other_agents),
                        "is_human": False,
                        "is_super": True,
                        "is_disruptor": bool(other["is_disruptor_bot"]),
                        "attack_weight": attack_priority(
                            other_id,
                            rival_is_bot=True,
                            rival_is_super=True,
                            index=index,
                            mi_effective=default_mi,
                            price=expected_price,
                            predicted_margin=prior_margin_by_company.get(other_id, 0.0),
                        ),
                        "strategic_weight": rival_asset_weight(other_id),
                        "available": predicted_available,
                    })

        current_pressure: dict[int, float] = {}
        for index in selected:
            market = markets[index]
            size = float(market["population"]) * float(market["penetration"]) * growth ** max(0, official_round - 1)
            current_pressure[index] = sum(item.get("available", 0.0) for item in rival_metrics[index]) / max(1.0, size)
        market_pressure = max(
            (max(utilization.get(index, 0.0), current_pressure.get(index, 0.0)) for index in selected),
            default=0.0,
        )
        # Ordinary bots open only their playbook's pools. Super Bots still
        # evaluate alternate combinations against current competition.
        use_qi = market_pressure >= 0.55 or (plan_index >= 1 and profile in (2, 4, 6) and market_pressure >= 0.30)
        if super_mode:
            # Super Bots evaluate the full QI range; QI=1 is the effective
            # "closed" candidate and therefore costs almost nothing per unit.
            use_qi = True
        use_mi = plan_index >= 2
        if not super_mode:
            use_qi = normal_qi > 0
            use_mi = normal_mi > 0 and official_round >= normal_mi_start
        mi_city_limit = min(
            len(selected),
            1 + max(0, plan_index - 2) // 2 + (1 if plan_index >= 4 and rng.random() > 0.55 else 0),
        )
        mi_priority = sorted(
            selected,
            key=lambda index: (
                str(markets[index]["city"]) == home,
                agent_plan[index][1],
                float(markets[index]["population"]) * float(markets[index]["penetration"]),
            ),
            reverse=True,
        )
        if super_mode and use_mi:
            # MI owns an independent 20-CPI pool in every city. Once the route
            # is active, fund every opened market rather than only a shortlist.
            mi_selected = {
                index for index in selected if agent_plan[index][1] > 0
            }
        else:
            mi_selected = set(mi_priority[:mi_city_limit]) if use_mi else set()

        # Choose the CPI indices first. The affordable production count is then
        # derived from the strategy guide's complete group cost; investment is
        # never used as a blind cash sink after production has been calculated.
        ma_index_target = (
            ma_threshold
            * max(ma_round_buffer, 1.02 + plan_index * 0.16)
            * variation
            * ma_strength
        )
        normal_phase = (1.00, 1.08, 1.20, 1.38, 1.58, 1.82, 2.08)[plan_index]
        if not super_mode:
            # Early rounds buy just enough of an opened CPI pool to compete;
            # later rounds widen naturally. Distinct profiles give the field
            # MA-, QI-, MI- and balanced plans without an expensive optimiser.
            ma_index_target = ma_threshold * normal_ma * normal_phase * rng.uniform(.90, 1.13)
            if not use_qi and not use_mi and normal_price_stance != "discount":
                ma_index_target = max(1.0, ma_index_target)
            if distress_liquidation and normal_ma > .01:
                ma_index_target = ma_threshold * rng.uniform(1.02, 1.10)
            elif distress_liquidation:
                ma_index_target = 0.0
        if super_mode:
            rival_ma = _upper_typical([item["ma"] for index in selected for item in rival_metrics[index]])
            ma_index_target = min(
                5000.0,
                max(ma_index_target, ma_threshold * super_aggression, rival_ma * super_aggression),
            )
        qi_line = max(float(markets[index]["max_price"]) / 50.0 for index in selected)
        qi_index_target = qi_line * max(qi_safe_multiplier, qi_strength)
        if not super_mode:
            qi_index_target = qi_line * max(
                qi_safe_multiplier,
                normal_qi * normal_phase * rng.uniform(.90, 1.13),
            )
            if distress_liquidation and (normal_ma > .01 or cash_ratio < .75):
                use_qi = False
                qi_index_target = 0.0
            elif distress_liquidation:
                # A quality specialist with working cash may retain a cheap
                # QI seed on the old stock without buying another large pool.
                qi_index_target = min(qi_index_target, rng.uniform(1.0, 5.0))
        if super_mode:
            rival_qi = _upper_typical([item["qi"] for index in selected for item in rival_metrics[index]])
            qi_index_target = min(
                qi_line * 5.0,
                max(qi_index_target, qi_line * super_aggression, rival_qi * super_aggression),
            )
        marketing_targets: dict[int, float] = {}
        mi_thresholds: dict[int, float] = {}
        for index in selected:
            market = markets[index]
            active_agents = agent_plan[index][1]
            size = float(market["population"]) * float(market["penetration"]) * growth ** max(0, official_round - 1)
            threshold = (float(market["max_price"]) / 50.0) * size * 0.20
            threshold /= max(1.0, 1.0 + active_agents * 0.10) * 1.5 * 2.0
            mi_thresholds[index] = threshold
            target_mi = threshold * mi_strength
            if not super_mode:
                target_mi = threshold * max(
                    1.02,
                    normal_mi * normal_phase * rng.uniform(.90, 1.13),
                )
                if distress_liquidation:
                    target_mi = 0.0
            if super_mode:
                rival_mi = _upper_typical([item["mi_effective"] for item in rival_metrics[index]])
                target_mi = min(
                    threshold * super_mi_cap,
                    max(
                        target_mi,
                        threshold * super_aggression,
                        rival_mi * super_aggression / (1.0 + active_agents * 0.10),
                    ),
                )
            marketing_targets[index] = target_mi if index in mi_selected and active_agents else 0.0

        if distress_liquidation:
            # Don't commit to large fixed MI or research while liquidating.
            use_mi = False
            mi_selected = set()
            marketing_targets = {index: 0.0 for index in marketing_targets}

        group = _balanced_production_group(worker_need, worker_hours, engineer_need, engineer_hours, component_need)
        group_cost = (
            group["workers"] * worker_salary * 3
            + group["engineers"] * engineer_salary * 3
            + group["workers"] * worker_training
            + group["engineers"] * engineer_training
            + group["components"]
            * (float(home_market["component_material"]) * material_factor + float(home_market["component_storage"]))
            + group["products"]
            * (float(home_market["product_material"]) * material_factor + float(home_market["product_storage"]))
            + ma_index_target * (group["workers"] + group["engineers"])
            + (qi_index_target * group["products"] if use_qi else 0.0)
        )

        def budget_for(production: int) -> dict[str, Any]:
            components_to_make = max(0, production * component_need - old_components)
            worker_effective = components_to_make * worker_need * worker_hours / 504.0
            engineer_effective = production * engineer_need * engineer_hours / 504.0
            # Do not budget production around an optimistic wage multiplier:
            # the current-round market average is endogenous and can rise when
            # other teams also raise salaries. A multiplier above 1 is upside,
            # not capacity the bot must rely upon to complete its plan.
            worker_planning_multiplier = min(1.0, worker_multiplier)
            engineer_planning_multiplier = min(1.0, engineer_multiplier)
            current_workers = employee_count(conn, company_id, "worker")
            current_engineers = employee_count(conn, company_id, "engineer")
            worker_delta = _staff_delta(conn, company_id, "worker", round_no, worker_effective / worker_planning_multiplier)
            engineer_delta = _staff_delta(conn, company_id, "engineer", round_no, engineer_effective / engineer_planning_multiplier)
            if super_mode and production <= 0 and old_products > 0 and ma_index_target >= ma_threshold:
                # Inventory-only MA play: retain exactly one employee so the
                # whole MA investment becomes the MA index denominator.
                if current_workers > 0:
                    worker_delta = max(worker_delta, 1 - current_workers)
                    engineer_delta = -current_engineers
                elif current_engineers > 0:
                    worker_delta = -current_workers
                    engineer_delta = max(engineer_delta, 1 - current_engineers)
                else:
                    worker_delta = 1
                    engineer_delta = 0
            workers = max(0, current_workers + worker_delta)
            engineers = max(0, current_engineers + engineer_delta)
            staff_cost = workers * worker_salary * 3 + engineers * engineer_salary * 3
            staff_cost += max(0, worker_delta) * worker_training + max(0, engineer_delta) * engineer_training
            staff_cost += max(0, -worker_delta) * worker_salary + max(0, -engineer_delta) * engineer_salary
            new_components = components_to_make
            material_cost = new_components * float(home_market["component_material"]) * material_factor
            material_cost += production * float(home_market["product_material"]) * material_factor
            storage_cost = max(0, old_components + new_components - max(old_components, int(bot["component_storage_capacity"] or 0))) * float(home_market["component_storage"])
            storage_cost += max(0, old_products + production - max(old_products, int(bot["product_storage_capacity"] or 0))) * float(home_market["product_storage"])
            management = ma_index_target * max(1, workers + engineers)
            denominator = old_products * 1.2 + production
            quality = qi_index_target * max(1.0, denominator) if use_qi else 0.0
            marketing = dict(marketing_targets)
            total = staff_cost + material_cost + storage_cost + agent_cost + management + quality + sum(marketing.values())
            return {
                "worker_delta": worker_delta, "engineer_delta": engineer_delta,
                "workers": workers, "engineers": engineers, "management": management,
                "quality": quality, "marketing": marketing, "staff_cost": staff_cost,
                "agent_cost": agent_cost,
                "core_total": staff_cost + material_cost + storage_cost + agent_cost,
                "material_cost": material_cost,
                "total": total,
            }

        last_round = official_round >= total_rounds
        # For strategy construction the new loan is simply additional usable
        # capital. Existing debt is deliberately not subtracted a second time;
        # settlement still records debt and interest normally.
        cash_budget = max(0.0, float(bot["cash"]) + loan_change)

        # Never submit an MI plan that consumes the whole company before a
        # single product can be made. Postpone that line until the large
        # threshold and a minimal operating plan are both affordable.
        if use_mi and float(budget_for(0)["total"]) > cash_budget * 0.72:
            use_mi = False
            mi_selected = set()
            marketing_targets = {index: 0.0 for index in marketing_targets}
        if not super_mode and not use_qi and not use_mi:
            # Keep an investment route if a discount play cannot go below
            # the market average without pricing below its production cost.
            base_unit_cost = (
                (group_cost - ma_index_target * (group["workers"] + group["engineers"]))
                / max(1.0, group["products"])
            )
            price_pool_reachable = any(
                min(previous_prices[index], float(markets[index]["max_price"]))
                > max(price_min, base_unit_cost + (
                    transport_cost if str(markets[index]["city"]) != home else 0.0
                )) * 1.03
                for index in selected
            )
            if normal_price_stance != "discount" or not price_pool_reachable:
                if ma_index_target == 0.0:
                    group_cost += group["workers"] + group["engineers"]
                    ma_index_target = 1.0

        opened_investment_pools = int(ma_index_target > 0) + int(use_qi) + int(use_mi)
        if market_pressure < 0.45:
            target_fraction = rng.uniform(0.54, 0.66)
        elif market_pressure < 0.60:
            target_fraction = min(0.68, (0.16 * opened_investment_pools + 0.10) * rng.uniform(0.90, 1.10))
        else:
            target_fraction = min(
                0.70,
                (0.18 * opened_investment_pools + (0.18 if super_mode else 0.10)) * rng.uniform(0.88, 1.08),
            )
        safe_market_units = 0.0
        for index in selected:
            market = markets[index]
            city_size = float(market["population"]) * float(market["penetration"]) * growth ** max(0, official_round - 1)
            safe_market_units += city_size * target_fraction / city_competitors[index]
        target_available = int(safe_market_units * variation)
        if not super_mode:
            target_available = int(target_available * normal_pace)
        if market_pressure < 0.45:
            target_available = max(target_available, desired_available)
        if prior_production and prior_total > 0:
            prior_sell_ratio = prior_sold / prior_total
            if prior_sell_ratio < 0.80:
                target_available = min(
                    target_available,
                    max(old_products, int(prior_sold * (1.04 + profile * 0.01) + old_products)),
                )
            elif prior_sell_ratio >= 0.95 and market_pressure < 0.45:
                # The market grew only one step; forecast that other competent
                # teams also expand after a sell-out instead of treating last
                # round's empty capacity as ours alone.
                expansion_factor = rng.uniform(1.55, 2.12)
                target_available = min(
                    target_available,
                    max(desired_available, int(prior_sold * expansion_factor + old_products)),
                )
        if not prior_production and old_products >= desired_available:
            # Existing inventory already covers the plan: control stock and
            # release surplus staff instead of treating new city capacity as a
            # reason to manufacture another batch immediately.
            target_available = old_products
        demand_target = max(0, target_available - old_products)
        fixed_before_groups = agent_cost + sum(marketing_targets.values())
        # b = floor((cash - Agent - MI) / complete group cost).  A group
        # already contains payroll, component/product costs, MA and QI, so all
        # five outputs stay in the exact F:G:H:I ratio supplied by the KDS.
        demand_groups = (
            math.ceil(demand_target / max(group["products"], 1.0))
            if demand_target > 0 else 0
        )
        # Hard cash rule supplied by the organiser: after Agent and MI, every
        # Bot buys the maximum number of complete groups. The only exception is
        # when every market was at least 95% full in the previous round.
        formula_groups = math.floor(
            max(0.0, cash_budget - fixed_before_groups) / max(group_cost, 1.0)
        )
        low_groups = 0
        if all_markets_full:
            high_groups = max(0, demand_groups)
        else:
            high_groups = max(1, demand_groups, formula_groups)
            while budget_for(int(math.floor(high_groups * group["products"])))["total"] <= cash_budget + 1e-9:
                low_groups = high_groups
                high_groups *= 2
        while low_groups < high_groups:
            middle_groups = (low_groups + high_groups + 1) // 2
            middle_products = int(math.floor(middle_groups * group["products"]))
            if budget_for(middle_products)["total"] <= cash_budget + 1e-9:
                low_groups = middle_groups
            else:
                high_groups = middle_groups - 1
        production = int(math.floor(low_groups * group["products"]))
        budget = budget_for(production)
        inventory_heavy = (
            old_products > max(20, prior_sold * 0.20)
            or (prior_production and prior_total > 0 and prior_sold / prior_total < 0.90)
            or old_products + production > demand_target
        )

        def make_city_rows(
            aggressive: bool = False,
            price_ratio_override: float | None = None,
        ) -> list[dict[str, float | int | str]]:
            rows: list[dict[str, float | int | str]] = []
            active_sizes = [
                float(markets[index]["population"]) * float(markets[index]["penetration"])
                for index in selected if agent_plan[index][1] > 0
            ]
            non_home_share = 0.0
            if active_sizes:
                non_home_share = sum(
                    size for index, size in zip([i for i in selected if agent_plan[i][1] > 0], active_sizes)
                    if str(markets[index]["city"]) != home
                ) / sum(active_sizes)
            available_units = max(1, old_products + production)
            allocated_operating_cost = (
                float(budget["core_total"]) + available_units * transport_cost * non_home_share
            ) / available_units
            for index, market in enumerate(markets):
                delta, agents_after = agent_plan[index]
                marketing = budget["marketing"].get(index, 0.0)
                cap = min(price_max, float(market["max_price"]))
                city_pressure = max(utilization.get(index, 0.0), current_pressure.get(index, 0.0))
                previous_price = previous_prices.get(index, float(market["initial_avg_price"]))
                low_price_unlocked = (
                    previous_price >= cap * 0.75
                    or late_game_low_price
                    or index in saturated_low_price_indices
                    or (super_mode and (all_super_field or current_pressure.get(index, 0.0) >= 0.60))
                )
                if not super_mode:
                    # A premium specialist keeps its route unless inventory
                    # distress forces a recovery. Price specialists look for
                    # below-average entry earlier without going below cost.
                    low_price_unlocked = (
                        distress_liquidation
                        or normal_price_stance == "discount"
                        or (normal_price_stance == "flex" and low_price_unlocked)
                    )
                reference = cap * high_price_ratio
                if price_ratio_override is not None and index in selected:
                    reference = cap * (
                        price_ratio_override
                        if low_price_unlocked or price_ratio_override >= 0.75
                        else high_price_ratio
                    )
                if (
                    price_ratio_override is None
                    and index in selected
                    and low_price_unlocked
                    and (city_pressure >= 0.60 or aggressive)
                ):
                    reference = previous_price - price_cut
                if (
                    super_mode
                    and price_ratio_override is None
                    and index in selected
                    and low_price_unlocked
                    and (city_pressure >= 0.40 or aggressive)
                ):
                    rival_prices = [item["price"] for item in rival_metrics[index] if item["price"] > 0]
                    rival_floor = min(rival_prices, default=previous_price)
                    factor = aggressive_price_factor if aggressive else normal_price_factor
                    reference = min(previous_price - price_cut, rival_floor * factor)
                component_labor = component_need * worker_need * worker_hours / 504.0 * worker_salary * 3
                product_labor = engineer_need * engineer_hours / 504.0 * engineer_salary * 3
                direct_unit_cost = (
                    component_need * float(home_market["component_material"]) * material_factor
                    + float(home_market["product_material"]) * material_factor
                    + component_labor + product_labor
                    + (transport_cost if str(market["city"]) != home else 0.0)
                )
                margin = 1.12 if super_mode else 1.06
                ordinary_strategic_low_price = bool(
                    not super_mode
                    and index in selected
                    and agents_after > 0
                    and (normal_price_stance == "discount" or (
                        normal_price_stance == "flex"
                        and (late_game_low_price or index in saturated_low_price_indices)
                    ))
                )
                if (distress_liquidation or ordinary_strategic_low_price) and index in selected and agents_after > 0:
                    # Flexible liquidation price: move from the last market
                    # average toward cost as inventory/cash pressure rises.
                    # Never deliberately sell below the real per-unit cost.
                    cost_floor = max(
                        price_min,
                        direct_unit_cost * 1.03,
                        (float(budget["total"]) / available_units + (
                            transport_cost if str(market["city"]) != home else 0.0
                        )) * 1.03 if not super_mode else 0.0,
                    )
                    average_ceiling = min(cap, max(price_min, previous_price))
                    if distress_liquidation:
                        pressure_depth = (
                            0.48
                            + min(0.30, prior_surplus_ratio * 0.70)
                            + min(0.14, max(0.0, 0.90 - cash_ratio) * 0.40)
                            + rng.uniform(-0.06, 0.07)
                        )
                        pressure_depth = min(0.94, max(0.42, pressure_depth))
                    else:
                        # Late/saturated markets undercut more gently than an
                        # emergency liquidation, preserving positive margin.
                        pressure_depth = (
                            0.22
                            + (0.10 if official_round >= total_rounds else 0.0)
                            + (0.12 if index in saturated_low_price_indices else 0.0)
                            + min(0.16, prior_surplus_ratio * 0.45)
                            + rng.uniform(-0.05, 0.07)
                        )
                        if normal_price_stance == "discount":
                            pressure_depth += .24
                        pressure_depth = min(0.88, max(0.16, pressure_depth))
                    reference = (
                        average_ceiling - (average_ceiling - cost_floor) * pressure_depth
                        if average_ceiling >= cost_floor else cost_floor
                    )
                elif super_mode and price_ratio_override is not None and not late_disruptor:
                    # An explicit optimiser candidate may run a pure low-price
                    # strategy. Candidate scoring already rejects a total-plan
                    # loss, so only protect the direct unit contribution here.
                    reference = max(reference, direct_unit_cost * 1.03)
                else:
                    reference = max(reference, direct_unit_cost * 1.12, allocated_operating_cost * margin)
                rows.append({
                    "index": index, "city": str(market["city"]), "agent_delta": delta,
                    "agents_after": agents_after, "marketing": marketing,
                    "price": min(cap, max(price_min, reference)),
                })
            return rows

        def forecast_super_capacity(rows: list[dict[str, float | int | str]]) -> float:
            if not super_mode or old_products + production <= 0:
                return 0.0
            capacity = 0.0
            ma_index = float(budget["management"]) / max(1, int(budget["workers"]) + int(budget["engineers"]))
            qi_index = float(budget["quality"]) / max(1.0, old_products * 1.2 + production)
            for row in rows:
                index = int(row["index"])
                if index not in selected or int(row["agents_after"]) <= 0:
                    continue
                entries: list[dict[str, float | int]] = []
                weighted_prices: list[tuple[float, float]] = []
                for rival_number, rival in enumerate(rival_metrics[index]):
                    rival_agents = max(1.0, rival.get("agents", 1.0))
                    entries.append({
                        "company_id": int(rival.get("company_id", -(rival_number + 1))),
                        "ma_index": rival["ma"],
                        "qi_index": rival["qi"],
                        "mi_investment": rival["mi_effective"] / (1.0 + rival_agents * 0.10),
                        "price": rival["price"], "agents": rival_agents,
                        "attack_weight": float(rival.get("attack_weight", 0.0)),
                        "available": float(rival.get("available", 0.0)),
                    })
                    weighted_prices.append((rival["price"], max(1.0, rival.get("available", 1.0))))
                entries.append({
                    "company_id": company_id, "ma_index": ma_index, "qi_index": qi_index,
                    "mi_investment": float(row["marketing"]), "price": float(row["price"]),
                    "agents": int(row["agents_after"]),
                })
                weighted_prices.append((float(row["price"]), max(1.0, old_products + production)))
                market = markets[index]
                size = float(market["population"]) * float(market["penetration"]) * growth ** max(0, official_round - 1)
                average_price = investment_average_prices(
                    entries,
                    [weight for _, weight in weighted_prices],
                    fallback=previous_prices[index],
                    market_size=size,
                    max_price=float(market["max_price"]),
                    ma_large_threshold=ma_threshold,
                )
                own_cpi = allocate_city_cpi_for_company(
                    entries, market_size=size, max_price=float(market["max_price"]),
                    ma_large_threshold=ma_threshold, average_price=average_price,
                    market_average_price=previous_prices[index],
                    price_power=price_power,
                    target_company_id=company_id,
                )
                capacity += size * own_cpi / 100.0
            return capacity

        city_rows = make_city_rows(inventory_heavy)
        if super_mode:
            # Search the permitted strategy space instead of hard-coding a
            # low-price personality. Every candidate is passed through the same
            # CPI allocator used by settlement and through the same complete
            # group-cost formula used by normal Bots.
            base_group_cost = max(
                0.0,
                group_cost
                - ma_index_target * (group["workers"] + group["engineers"])
                - qi_index_target * group["products"],
            )
            # Keep the organiser's original 1–8000 / 1–3000 search range for
            # the default KDS, but express it as live threshold multiples so a
            # different event scales correctly.
            ma_ceiling = max(1.0, ma_threshold * (8000.0 / 1300.0))
            qi_ceiling = max(1.0, qi_line * 6.0)
            mi_ratio_ceiling = 6.0 if use_mi else 0.0
            if competitive_mode:
                ma_ceiling = max(ma_ceiling, ma_threshold * 10.0)
                qi_ceiling = max(qi_ceiling, qi_line * 10.0)
                if use_mi:
                    rival_mi_ratios = []
                    for index in selected:
                        benefit = 1.0 + agent_plan[index][1] * 0.10
                        base_threshold = mi_thresholds.get(index, 0.0) * benefit
                        if base_threshold <= 0.0:
                            continue
                        rival_mi_ratios.extend(
                            float(item.get("mi_effective", 0.0)) / base_threshold
                            for item in rival_metrics[index]
                            if float(item.get("attack_weight", 0.0)) > 0.0
                        )
                    mi_ratio_ceiling = max(
                        mi_ratio_ceiling,
                        min(
                            max(0.0, cash_budget - agent_cost)
                            / max(1.0, sum(mi_thresholds.get(i, 0.0) for i in mi_selected)),
                            max(rival_mi_ratios, default=0.0) * 1.30,
                        ),
                    )
            field_ma = min(ma_ceiling, max(1.0, ma_index_target))
            field_qi = min(qi_ceiling, max(1.0, qi_index_target))
            field_mi_ratio = max(
                (
                    marketing_targets.get(index, 0.0) / mi_thresholds[index]
                    for index in selected
                    if index in mi_selected and mi_thresholds.get(index, 0.0) > 0
                ),
                default=0.0,
            )
            field_mi_ratio = min(mi_ratio_ceiling, max(0.0, field_mi_ratio))
            leader_ma = field_ma
            leader_qi = field_qi
            leader_mi_ratio = field_mi_ratio
            price_ratio_candidates = {
                min(0.98, max(0.75, high_price_ratio)), 1.0, 0.9995, 0.995, 0.95, 0.88, 0.81, 0.75,
            }
            if previous_leader:
                leader_ma = min(ma_ceiling, max(1.0, float(previous_leader["ma"])))
                leader_qi = min(qi_ceiling, max(1.0, float(previous_leader["qi"])))
                for index in selected:
                    leader_city = previous_leader["cities"].get(str(markets[index]["city"]))
                    if not leader_city:
                        continue
                    if use_mi and mi_thresholds.get(index, 0.0) > 0:
                        leader_mi_ratio = max(
                            leader_mi_ratio,
                            min(
                                mi_ratio_ceiling,
                                max(0.0, float(leader_city["marketing"]) / mi_thresholds[index]),
                            ),
                        )
                    cap = min(price_max, float(markets[index]["max_price"]))
                    if cap > 0:
                        price_ratio_candidates.add(min(
                            0.98,
                            max(price_min / cap, float(leader_city["price"]) / cap),
                        ))
            if late_game_low_price or saturated_low_price_indices or any(
                previous_prices[index] >= min(price_max, float(markets[index]["max_price"])) * 0.75
                for index in selected
            ) or all_super_field or any(
                current_pressure.get(index, 0.0) >= 0.60 for index in selected
            ):
                # Once the low-price condition opens, search continuously down
                # to the configured floor and include small undercuts of every
                # visible rival price. MA/QI=1 and MI=0 form the pure-price path.
                minimum_ratio = max(
                    price_min / max(1.0, min(price_max, float(markets[index]["max_price"])))
                    for index in selected
                )
                # Low-price candidates use normalized price gaps transformed
                # to the live KDS exponent.  This gives comparable CPI-weight
                # steps across an 8th-power regional game and a 20th-power
                # national game without blindly jumping thousands of yuan.
                representative_ratios = [
                    previous_prices[index]
                    / max(1.0, min(price_max, float(markets[index]["max_price"])))
                    for index in selected
                ]
                previous_ratio = sum(representative_ratios) / max(1, len(representative_ratios))
                for base_depth in (0.08, 0.16, 0.28, 0.42, 0.58, 0.74, 0.88, 0.97):
                    depth = _kds_price_depth(base_depth, price_power)
                    candidate_ratio = previous_ratio - (previous_ratio - minimum_ratio) * depth
                    price_ratio_candidates.add(round(min(0.98, max(minimum_ratio, candidate_ratio)), 5))
                price_ratio_candidates.add(minimum_ratio)
                for index in selected:
                    cap = min(price_max, float(markets[index]["max_price"]))
                    if cap <= 0 or not (
                        previous_prices[index] >= cap * 0.75
                        or late_game_low_price
                        or index in saturated_low_price_indices
                        or all_super_field
                        or current_pressure.get(index, 0.0) >= 0.60
                    ):
                        continue
                    rival_prices = [
                        float(rival["price"])
                        for rival in rival_metrics[index]
                        if float(rival["price"]) > 0
                    ]
                    if rival_prices:
                        candidate_price = _price_for_weight_advantage(
                            previous_prices[index],
                            min(rival_prices),
                            price_min,
                            price_power,
                        )
                        price_ratio_candidates.add(round(
                            min(0.98, max(minimum_ratio, candidate_price / cap)),
                            5,
                        ))

            # Rival decisions and city KDS stay fixed during this Bot's search.
            # Build them once, while retaining the same summation order and
            # every candidate/CPI calculation used by the original search.
            active_market_count = max(1, sum(1 for index in selected if agent_plan[index][1] > 0))
            candidate_city_data: list[tuple[Any, ...]] = []
            forecast_interest = (
                max(0.0, float(bot["debt"] or 0) + float(loan_change))
                * float(home_market["interest_rate"] or 0)
            )
            component_labor = component_need * worker_need * worker_hours / 504.0 * worker_salary * 3
            product_labor = engineer_need * engineer_hours / 504.0 * engineer_salary * 3
            for index in selected:
                agents_after = agent_plan[index][1]
                if agents_after <= 0:
                    continue
                market = markets[index]
                cap = min(price_max, float(market["max_price"]))
                low_price_unlocked = (
                    previous_prices[index] >= cap * 0.75
                    or late_game_low_price
                    or index in saturated_low_price_indices
                    or all_super_field
                    or current_pressure.get(index, 0.0) >= 0.60
                )
                direct_unit_cost = (
                    component_need * float(home_market["component_material"]) * material_factor
                    + float(home_market["product_material"]) * material_factor
                    + component_labor + product_labor
                    + (transport_cost if str(market["city"]) != home else 0.0)
                )
                entries: list[dict[str, float | int]] = []
                rival_weights: list[float] = []
                rival_weighted_prices: list[float] = []
                rival_asset_weighted_supply: list[float] = []
                for rival_number, rival in enumerate(rival_metrics[index]):
                    rival_agents = max(1.0, rival.get("agents", 1.0))
                    entries.append({
                        "company_id": int(rival.get("company_id", -(rival_number + 1))),
                        "ma_index": rival["ma"],
                        "qi_index": rival["qi"],
                        "mi_investment": rival["mi_effective"] / (1.0 + rival_agents * 0.10),
                        "price": rival["price"], "agents": rival_agents,
                        "attack_weight": float(rival.get("attack_weight", 0.0)),
                        "available": float(rival.get("available", 0.0)),
                    })
                    weight = max(1.0, rival.get("available", 1.0))
                    rival_weights.append(weight)
                    rival_weighted_prices.append(rival["price"] * weight)
                    rival_asset_weighted_supply.append(
                        weight * float(rival.get("strategic_weight", 1.0))
                    )
                own_entry = {
                    "company_id": company_id, "ma_index": 0.0, "qi_index": 0.0,
                    "mi_investment": 0.0, "price": 0.0, "agents": agents_after,
                }
                entries.append(own_entry)
                size = float(market["population"]) * float(market["penetration"]) * growth ** max(0, official_round - 1)
                cpi_evaluator = prepare_city_cpi_for_company(
                    entries, target_company_id=company_id, market_size=size,
                    max_price=float(market["max_price"]), ma_large_threshold=ma_threshold,
                    price_power=price_power,
                    market_average_price=previous_prices[index],
                )
                average_evaluator = prepare_investment_average_prices(
                    entries,
                    [*rival_weights, 0.0],
                    target_index=len(entries) - 1,
                    fallback=previous_prices[index],
                    market_size=size,
                    max_price=cap,
                    ma_large_threshold=ma_threshold,
                )
                # Price CPI is exceptionally fragile when the KDS uses a high
                # exponent: one later Bot can take almost the whole 40-point
                # pool with a tiny undercut.  Candidate scoring therefore keeps
                # a KDS-scaled shadow competitor instead of valuing a temporary
                # price monopoly as guaranteed revenue.  This is only a Bot
                # risk model; settlement continues to use the exact field.
                shadow_price_competitors = min(3.0, max(0.75, price_power / 12.0))
                aggregate_rival_weight = (
                    sum(rival_asset_weighted_supply) / sum(rival_weights)
                    if rival_weights and sum(rival_weights) > 0 else 1.0
                )
                candidate_city_data.append((
                    index, cap, low_price_unlocked, direct_unit_cost,
                    rival_weights, rival_weighted_prices, entries, size,
                    cpi_evaluator, average_evaluator,
                    shadow_price_competitors, aggregate_rival_weight,
                ))

            def evaluate_candidate(
                candidate_ma: float,
                candidate_qi: float,
                candidate_mi_ratio: float,
                candidate_price_ratio: float,
                force_full_output: bool = False,
                mi_focus: tuple[int, ...] | None = None,
            ) -> dict[str, Any]:
                candidate_marketing = {
                    index: (
                        mi_thresholds.get(index, 0.0) * candidate_mi_ratio
                        if index in mi_selected and agent_plan[index][1] > 0
                        and (mi_focus is None or index in mi_focus) else 0.0
                    )
                    for index in selected
                }
                candidate_group_cost = (
                    base_group_cost
                    + candidate_ma * (group["workers"] + group["engineers"])
                    + candidate_qi * group["products"]
                )
                # QI also applies to old stock. Omitting this fixed part made
                # inventory-heavy low-price plans look cheaper than they are.
                candidate_fixed = (
                    agent_cost
                    + sum(candidate_marketing.values())
                    + candidate_qi * old_products * 1.2
                )
                candidate_group_limit = math.floor(
                    max(0.0, cash_budget - candidate_fixed) / max(candidate_group_cost, 1.0)
                )
                candidate_groups = _affordable_group_count(
                    cash_budget,
                    candidate_fixed,
                    candidate_group_cost,
                    candidate_group_limit,
                )
                candidate_production = int(math.floor(candidate_groups * group["products"]))
                candidate_available = old_products + candidate_production
                if candidate_available <= 0:
                    return {
                        "score": (-1, -math.inf, 0.0, -math.inf, candidate_price_ratio),
                        "ma": candidate_ma, "qi": candidate_qi,
                        "marketing": candidate_marketing, "groups": candidate_groups,
                        "price_ratio": candidate_price_ratio, "coverage": 0.0,
                        "predicted_profit": -math.inf, "risk_profit": -math.inf,
                        "predicted_sold": 0.0, "sell_ratio": 0.0,
                        "marketing_total": sum(candidate_marketing.values()),
                        "rival_damage": 0.0, "full_output": force_full_output,
                        "starting_assets": own_strategy_assets,
                        "ending_assets": -math.inf,
                    }

                city_capacities: list[tuple[int, float, float]] = []
                rival_damage = 0.0
                for (
                    index, cap, low_price_unlocked, direct_unit_cost,
                    rival_weights, rival_weighted_prices, _average_entries, size,
                    cpi_evaluator, average_evaluator,
                    shadow_price_competitors, aggregate_rival_weight,
                ) in candidate_city_data:
                    effective_ratio = (
                        candidate_price_ratio
                        if low_price_unlocked or candidate_price_ratio >= 0.75
                        else min(0.98, max(0.75, high_price_ratio))
                    )
                    candidate_price = cap * effective_ratio
                    candidate_price = min(
                        cap,
                        max(
                            price_min,
                            candidate_price,
                            0.0 if late_disruptor else direct_unit_cost * 1.03,
                        ),
                    )
                    own_weight = max(1.0, candidate_available / active_market_count)
                    average_price = average_evaluator.resolve(
                        ma_index=candidate_ma,
                        qi_index=candidate_qi,
                        mi_investment=candidate_marketing[index],
                        price=candidate_price,
                        quantity=own_weight,
                        agents=agent_plan[index][1],
                    )
                    own_cpi = cpi_evaluator.evaluate(
                        ma_index=candidate_ma, qi_index=candidate_qi,
                        mi_investment=candidate_marketing[index],
                        price=candidate_price, average_price=average_price,
                    )
                    market_average = float(previous_prices[index])
                    if candidate_price < market_average:
                        own_price_weight = (market_average - candidate_price) ** price_power
                        known_price_weight = sum(cpi_evaluator.price_weights)
                        known_price_cpi = (
                            PRICE_CPI_TOTAL * own_price_weight
                            / max(own_price_weight + known_price_weight, 1e-300)
                        )
                        resilient_price_cpi = (
                            PRICE_CPI_TOTAL * own_price_weight
                            / max(
                                known_price_weight
                                + own_price_weight * (1.0 + shadow_price_competitors * 1.35),
                                1e-300,
                            )
                        )
                        own_cpi = max(0.0, own_cpi - known_price_cpi + resilient_price_cpi)
                    own_capacity = size * own_cpi / 100.0
                    city_capacities.append((index, own_capacity, candidate_price))
                    rival_available = sum(rival_weights)
                    rival_average_price = (
                        sum(rival_weighted_prices) / rival_available
                        if rival_available > 0 else candidate_price
                    )
                    # In a crowded market, every unit of capacity captured by
                    # this candidate removes revenue that the preset rivals
                    # could otherwise serve. A profitable undercut also reduces
                    # their pricing room. This is a search objective only; the
                    # engine's CPI and settlement formulas remain untouched.
                    own_city_supply = min(
                        own_capacity,
                        float(candidate_available) / active_market_count,
                    )
                    displaced = min(
                        own_city_supply,
                        max(0.0, rival_available + own_city_supply - size),
                    )
                    congestion = min(
                        1.0,
                        max(
                            0.0,
                            (rival_available + own_city_supply) / max(1.0, size) - 0.75,
                        ) / 0.25,
                    )
                    undercut_ratio = max(
                        0.0,
                        (rival_average_price - candidate_price)
                        / max(1.0, rival_average_price),
                    )
                    pressured = (
                        min(own_city_supply, rival_available)
                        * undercut_ratio
                        * 0.25
                        * congestion
                    )
                    rival_damage += (
                        (displaced + pressured)
                        * max(1.0, rival_average_price)
                        * aggregate_rival_weight
                    )

                total_capacity = sum(capacity for _, capacity, _ in city_capacities)
                # High investment does not require maximum production. Keep a
                # small demand buffer, but never pay for complete groups whose
                # products are not forecast to sell. This is the high-CPI,
                # controlled-output strategy used in saturated/high-price play.
                capacity_safety = (
                    0.94 if last_round
                    else 0.98 if official_round >= max(1, total_rounds - 1)
                    else 1.03
                )
                profitable_group_limit = max(
                    0,
                    int(math.ceil(
                        max(0.0, total_capacity * capacity_safety - old_products)
                        / max(1.0, group["products"])
                    )),
                )
                early_expansion = bool(
                    official_round <= max(2, math.ceil(total_rounds * 0.50))
                    and not all_markets_full
                    and market_pressure < 0.60
                    and prior_surplus_ratio < 0.20
                )
                if (
                    not force_full_output
                    and not late_disruptor
                    and not early_expansion
                    and profitable_group_limit < candidate_groups
                ):
                    candidate_groups = profitable_group_limit
                    candidate_production = int(math.floor(candidate_groups * group["products"]))
                    candidate_available = old_products + candidate_production
                predicted_sold = min(float(candidate_available), total_capacity)
                sell_ratio = predicted_sold / max(1.0, float(candidate_available))
                cpi_coverage = total_capacity / max(1.0, float(candidate_available))
                capacity_weight = sum(capacity for _, capacity, _ in city_capacities)
                predicted_price = sum(capacity * price for _, capacity, price in city_capacities) / max(1.0, capacity_weight)
                non_home_capacity = sum(
                    capacity for index, capacity, _ in city_capacities
                    if str(markets[index]["city"]) != home
                )
                predicted_transport = predicted_sold * transport_cost * non_home_capacity / max(1.0, capacity_weight)
                # With inventory but no new production, a pure MA strategy
                # keeps one employee; model that one-person MA spend here.
                control_ma_cost = candidate_ma if candidate_groups == 0 and old_products > 0 else 0.0
                predicted_cost = (
                    candidate_fixed
                    + candidate_groups * candidate_group_cost
                    + control_ma_cost
                    + predicted_transport
                )
                predicted_revenue = predicted_sold * predicted_price
                predicted_pre_tax = predicted_revenue - predicted_cost - forecast_interest
                predicted_tax = max(0.0, predicted_pre_tax * tax_rate)
                predicted_profit = predicted_pre_tax - predicted_tax
                # A conservative revenue haircut protects the late-game
                # strategy from a later undercut or small CPI forecast error.
                # Safe profit wins first; near break-even market filling is
                # allowed only when no safely profitable candidate exists.
                risk_pre_tax = predicted_revenue * 0.92 - predicted_cost - forecast_interest
                risk_profit = risk_pre_tax - max(0.0, risk_pre_tax * tax_rate)
                non_loss = int(predicted_profit >= 0)
                # Profit is the primary objective. Sell-through and CPI/stock
                # fit break ties between similarly profitable strategies, so a
                # pure low-price plan is selected only when it truly earns more.
                return {
                    "score": (
                        non_loss,
                        predicted_profit,
                        risk_profit,
                        sell_ratio,
                        -abs(cpi_coverage - 1.0),
                        candidate_price_ratio,
                    ),
                    "ma": candidate_ma, "qi": candidate_qi,
                    "marketing": candidate_marketing, "groups": candidate_groups,
                    "price_ratio": candidate_price_ratio,
                    "coverage": cpi_coverage,
                    "predicted_profit": predicted_profit,
                    "risk_profit": risk_profit,
                    "predicted_sold": predicted_sold,
                    "search_revenue": predicted_revenue,
                    "sell_ratio": sell_ratio,
                    "marketing_total": sum(candidate_marketing.values()),
                    "rival_damage": rival_damage,
                    "full_output": force_full_output,
                    "starting_assets": own_strategy_assets,
                    "ending_assets": own_strategy_assets + predicted_profit,
                    "production_cost": predicted_cost - predicted_transport,
                }

            # Use field-informed upper bounds plus deliberately high legal
            # bounds. For each strategy family, binary-search the point where
            # CPI capacity reaches stock. Every visited point still competes on
            # predicted profit, so the search does not blindly maximise CPI.
            strategy_upper_bounds = {
                (ma_ceiling, qi_ceiling, mi_ratio_ceiling),
                (ma_ceiling, 1.0, 0.0),
                (1.0, qi_ceiling, 0.0),
                (1.0, 1.0, mi_ratio_ceiling),
                (ma_ceiling, qi_ceiling, 0.0),
                (ma_ceiling, 1.0, mi_ratio_ceiling),
                (1.0, qi_ceiling, mi_ratio_ceiling),
                (ma_ceiling, field_qi, field_mi_ratio),
                (field_ma, qi_ceiling, field_mi_ratio),
                (field_ma, field_qi, mi_ratio_ceiling),
                (ma_ceiling, qi_ceiling, field_mi_ratio),
                (ma_ceiling, field_qi, mi_ratio_ceiling),
                (field_ma, qi_ceiling, mi_ratio_ceiling),
                (field_ma, field_qi, field_mi_ratio),
                (leader_ma, leader_qi, leader_mi_ratio),
            }
            best_candidate: dict[str, Any] | None = None
            candidate_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

            def test_candidate(
                candidate_ma: float,
                candidate_qi: float,
                candidate_mi_ratio: float,
                candidate_price_ratio: float,
                force_full_output: bool = False,
                mi_focus: tuple[int, ...] | None = None,
            ) -> dict[str, Any]:
                nonlocal best_candidate
                key = (
                    round(candidate_ma, 8), round(candidate_qi, 8),
                    round(candidate_mi_ratio, 8), round(candidate_price_ratio, 8),
                    bool(force_full_output),
                    mi_focus,
                )
                candidate = candidate_cache.get(key)
                if candidate is None:
                    candidate = evaluate_candidate(
                        candidate_ma,
                        candidate_qi,
                        candidate_mi_ratio,
                        candidate_price_ratio,
                        force_full_output,
                        mi_focus,
                    )
                    candidate_cache[key] = candidate
                if best_candidate is None or candidate["score"] > best_candidate["score"]:
                    best_candidate = candidate
                return candidate

            expansion_room = bool(
                not all_markets_full
                and official_round <= max(2, math.ceil(total_rounds * 0.50))
                and max((utilization.get(i, 0.0) for i in selected), default=0.0) < 0.60
            )
            focus_order = sorted(
                selected,
                key=lambda i: float(markets[i]["population"]) * float(markets[i]["penetration"])
                * (1.0 + agent_plan[i][1] * 0.10),
                reverse=True,
            )
            for candidate_price_ratio in sorted(price_ratio_candidates, reverse=True):
                # Pure-price path is always retained.
                test_candidate(1.0, 1.0, 0.0, candidate_price_ratio)
                # Learn the winning early-game route: a small investment seed
                # keeps its investment secondary pool open while complete
                # groups and near-ceiling prices compound sales cash. Exact
                # settlement, not visible CPI alone, decides if this sells.
                if candidate_price_ratio >= 0.75:
                    for ma_fraction in (0.02, 0.05, 0.15):
                        test_candidate(
                            max(1.0, ma_threshold * ma_fraction), 0.0, 0.0,
                            candidate_price_ratio, force_full_output=expansion_room,
                        )
                    if expansion_room:
                        test_candidate(1.0, 0.0, 0.0, candidate_price_ratio, force_full_output=True)
                if use_mi and mi_selected:
                    # The old spending shape remains an option, not a floor.
                    phase_bid = min(mi_ratio_ceiling, 1.02 + max(0, plan_index - 2) * .55
                                    + (sequence_strength - 1.0) * 2.0)
                    test_candidate(field_ma, field_qi, max(0., phase_bid), candidate_price_ratio)
                    # Concentrated MI is compared with broad MI; it is never
                    # forced to pay the large threshold in every open city.
                    for focus_count in sorted({1, min(3, len(focus_order))}):
                        test_candidate(
                            max(1.0, ma_threshold * 0.05), 0.0,
                            min(mi_ratio_ceiling, 1.10), candidate_price_ratio,
                            mi_focus=tuple(sorted(focus_order[:focus_count])),
                        )
                first_full_low_price = bool(
                    (floor_lock_disruptor or (all_super_field and bot_position == 0))
                    and candidate_price_ratio < 0.75
                )
                if first_full_low_price:
                    # The first sequential Bot has no real current-round Super
                    # decisions to copy. Explicitly compare a cash-max output
                    # low-price route against its controlled-output candidates.
                    test_candidate(
                        1.0, 1.0, 0.0, candidate_price_ratio,
                        force_full_output=True,
                    )
                for upper_ma, upper_qi, upper_mi_ratio in strategy_upper_bounds:
                    low_scale = 0.0
                    high_scale = 1.0
                    test_candidate(upper_ma, upper_qi, upper_mi_ratio, candidate_price_ratio)
                    if first_full_low_price:
                        test_candidate(
                            upper_ma, upper_qi, upper_mi_ratio,
                            candidate_price_ratio,
                            force_full_output=True,
                        )
                    for _ in range(7):
                        scale = (low_scale + high_scale) / 2.0
                        candidate = test_candidate(
                            1.0 + (upper_ma - 1.0) * scale,
                            1.0 + (upper_qi - 1.0) * scale,
                            upper_mi_ratio * scale,
                            candidate_price_ratio,
                        )
                        if float(candidate["coverage"]) >= 1.0:
                            high_scale = scale
                        else:
                            low_scale = scale

            if candidate_cache:
                # Preserve the full cheap search; verify bounded diverse finalists
                # with shared inventory, secondary sales and live pool averages.
                candidate_values = [row for row in candidate_cache.values()
                                    if math.isfinite(float(row.get("predicted_profit", -math.inf)))]
                shortlisted: dict[int, dict[str, Any]] = {}

                def include_shortlist(rows: list[dict[str, Any]], limit: int = 6) -> None:
                    for row in rows[:limit]:
                        shortlisted[id(row)] = row

                include_shortlist(sorted(candidate_values,
                    key=lambda row: float(row["predicted_profit"]), reverse=True), 16)
                if competitive_mode:
                    include_shortlist(sorted(candidate_values,
                        key=lambda row: float(row["predicted_profit"]) + float(row["rival_damage"]),
                        reverse=True), 8)
                if late_disruptor:
                    # Keep strong attack routes in the exact shortlist. They
                    # can win only if low-priced sales cover their costs.
                    include_shortlist(sorted(
                        candidate_values,
                        key=lambda row: float(row.get("rival_damage", 0.0)),
                        reverse=True,
                    ), 20)
                include_shortlist(sorted(candidate_values,
                    key=lambda row: float(row["marketing_total"]), reverse=True))
                include_shortlist(sorted(candidate_values,
                    key=lambda row: float(row["price_ratio"])))
                include_shortlist(sorted(candidate_values,
                    key=lambda row: float(row["ma"]) + float(row["qi"]), reverse=True))
                include_shortlist(sorted(
                    (row for row in candidate_values if row.get("full_output") and float(row["price_ratio"]) >= 0.75),
                    key=lambda row: (float(row["price_ratio"]), int(row["groups"])),
                    reverse=True,
                ), 12)

                # Reserve exact forecasts for low-spend high-price plans at
                # several affordable volumes. A tiny primary CPI must not
                # eliminate them before shared secondary stock is allocated.
                if not late_disruptor and not expansion_room:
                    for ratio in (1.0, 0.996, 0.95, 0.88):
                        for ma, qi in ((1.0, 1.0), (1.0, 0.0)):
                            seed = evaluate_candidate(ma, qi, 0.0, ratio, True, ())
                            groups = int(seed["groups"])
                            if groups <= 0 and old_products <= 0:
                                continue
                            for amount in sorted({max(1, round(groups * part)) for part in (.22, .40, .65, 1.0)} if groups else {0}):
                                row = {**seed, "groups": amount, "lean_secondary_route": True}
                                shortlisted[id(row)] = row
                if late_disruptor:
                    # Use full affordable output; vary price around the
                    # production cost, not volume around visible CPI.
                    unit_cost = (
                        base_group_cost + group["workers"] + group["engineers"]
                    ) / max(1.0, group["products"]) + 1.0
                    city_caps_by_name = {
                        str(markets[index]["city"]): min(price_max, float(markets[index]["max_price"]))
                        for index in selected
                    }
                    reference_cap = sum(city_caps_by_name.values()) / max(1, len(city_caps_by_name))
                    for markup in (.92, .97, 1.0, 1.015, 1.03, 1.06, 1.10, 1.16, 1.25, 1.40, 1.6, 2.0):
                        ratio = min(1.0, max(price_min, unit_cost * markup) / max(1.0, reference_cap))
                        row = evaluate_candidate(1.0, 1.0, 0.0, ratio, True, ())
                        row["uniform_late_price"] = True
                        shortlisted[id(row)] = row

                # A first-pass CPI estimate can underrate a route that is
                # eligible for a large price-to-price or investment-to-
                # investment second pass. Keep a bounded spread of
                # under-covered candidates from every CPI family so the exact
                # settlement forecast decides whether they survive.
                def secondary_family(row: dict[str, Any]) -> str:
                    active = []
                    if float(row.get("ma", 0.0)) > 1.00000001:
                        active.append("ma")
                    if float(row.get("qi", 0.0)) > 1.00000001:
                        active.append("qi")
                    if float(row.get("marketing_total", 0.0)) > 1e-9:
                        active.append("mi")
                    if not active:
                        return "price"
                    return active[0] if len(active) == 1 else "mixed"

                for family in ("price", "ma", "qi", "mi", "mixed"):
                    family_rows = [row for row in candidate_values
                                   if secondary_family(row) == family]
                    for target_coverage in (0.20, 0.45, 0.70, 0.95):
                        if family_rows:
                            row = min(
                                family_rows,
                                key=lambda item: (
                                    abs(float(item.get("coverage", 0.0)) - target_coverage),
                                    -float(item.get("predicted_profit", -math.inf)),
                                ),
                            )
                            shortlisted[id(row)] = row

                forecast_markets = [
                    {"city": str(market["city"]),
                     "market_size": float(market["population"]) * float(market["penetration"])
                                    * growth ** max(0, official_round - 1),
                     "max_price": float(market["max_price"]),
                     "base_average_price": previous_prices[index]}
                    for index, market in enumerate(markets)
                ]
                rival_players: dict[int, dict[str, Any]] = {}
                for index, rivals in rival_metrics.items():
                    for rival in rivals:
                        rival_id = int(rival["company_id"])
                        state = rival_players.setdefault(rival_id, {
                            "company_id": rival_id, "available": 0.0,
                            "ma_index": float(rival["ma"]), "qi_index": float(rival["qi"]),
                            "attack_weight": float(rival.get("attack_weight", 0)),
                            "is_human": bool(rival.get("is_human", False)),
                            "is_super": bool(rival.get("is_super", False)),
                            "is_disruptor": bool(rival.get("is_disruptor", False)),
                            "cities": {},
                        })
                        state["available"] += max(0.0, float(rival["available"]))
                        state["cities"][str(markets[index]["city"])] = {
                            "agents": int(rival["agents"]),
                            "marketing": float(rival["mi_effective"]) / (1 + float(rival["agents"]) * .1),
                            "price": float(rival["price"]),
                        }
                forecast_args = dict(markets=forecast_markets,
                                     ma_large_threshold=ma_threshold, price_power=price_power)
                baseline_forecast = (
                    forecast_market_sales(players=list(rival_players.values()), **forecast_args)
                    if super_mode else None
                )
                rival_baseline = {}
                active_benchmarks = (benchmark_rival_ids & rival_players.keys()
                                     if company_id in benchmark_bot_ids else set())
                real_rivals = (
                    (submitted_rival_ids & rival_players.keys()) if race_for_first and use_complete_field
                    else active_benchmarks
                )
                for rival_id in sorted(real_rivals):
                    actual = funded_rivals[rival_id]
                    company = one(conn, "SELECT cash,debt,home_city FROM companies WHERE id=?", (rival_id,))
                    plan_row = one(conn, "SELECT loan_change,research_investment FROM decisions WHERE company_id=? AND round_no=?",
                                   (rival_id, round_no))
                    if not plan_row:
                        continue
                    rival_home = next(m for m in markets if m["city"] == company["home_city"])
                    sites = rival_players[rival_id]["cities"]
                    fixed = (float(company["cash"]) + float(plan_row["loan_change"]) - float(actual["cash_pre_sales"])
                             + float(plan_row["research_investment"])
                             + (float(company["debt"]) + float(plan_row["loan_change"])) * float(rival_home["interest_rate"]))
                    rival_baseline[rival_id] = (fixed, str(company["home_city"]), float(company["cash"]) - float(company["debt"]), sites)

                def rival_profit(rival_id, after):
                    fixed, rival_home, _, sites = rival_baseline[rival_id]
                    units = after["companies"][rival_id]["sold_units"]
                    revenue = sum(count * (sites[city]["price"] - (transport_cost if city != rival_home else 0.))
                                  for city, count in units.items() if city in sites)
                    pre_tax = revenue - fixed
                    return pre_tax - max(0., pre_tax * tax_rate)

                projected_rival = None
                eligible_benchmarks = active_benchmarks & rival_baseline.keys()
                if eligible_benchmarks:
                    projected_rival = max(eligible_benchmarks, key=lambda i: (
                        rival_baseline[i][2] + rival_profit(i, baseline_forecast), -i))
                    # Copy funded current decisions, plus a previous champion
                    # if its plan differs. Re-budget in our own home/capacity.
                    benchmark = rival_players[projected_rival]
                    profiles = [benchmark]
                    if previous_leader:
                        profiles.append({**previous_leader, "ma_index": previous_leader["ma"],
                                         "qi_index": previous_leader["qi"], "available": benchmark["available"]})
                    for rival in profiles:
                        for investment, quote, volume in (
                            (1., 1., 1.), (.8, 1., 1.), (1., .99, 1.),
                            (1., 1.01, 1.15), (.72, .985, 1.25), (1.12, .98, 1.35),
                        ):
                            ma = float(rival["ma_index"]) * investment
                            qi = float(rival["qi_index"]) * investment
                            marketing = {i: investment * float(rival["cities"].get(str(markets[i]["city"]), {}).get("marketing", 0.))
                                         for i in selected if agent_plan[i][1] > 0}
                            prices = {city: float(values["price"]) * quote for city, values in rival["cities"].items()}
                            groups = max(0, math.ceil((float(rival["available"]) * volume - old_products) / group["products"]))
                            row = {"ma": ma, "qi": qi, "marketing": marketing, "groups": groups,
                                   "price_ratio": .95, "rival_seed": True, "forecast_prices": prices,
                                   "marketing_total": sum(marketing.values())}
                            shortlisted[id(row)] = row
                    if race_for_first and (human_present or use_complete_field):
                        # Compare the two naturally opposing routes: sell
                        # premium into the investment pool or take price CPI
                        # in the cities where the leader actually ships.
                        active = [i for i in selected if agent_plan[i][1] > 0]
                        leader_routes = benchmark["cities"]
                        for depth in (0., .02, .06, .13, .24, .38):
                            for ma, qi, mi in ((1., 1., 0.), (ma_threshold * 1.2, 1., 0.),
                                              (ma_threshold * 3, 1., .9), (1., qi_line * 2, 1.4)):
                                budget_seed = evaluate_candidate(ma, qi, 0., .95, True, ())
                                affordable = int(budget_seed["groups"])
                                for fraction in (.16, .31, .50, .73, 1.):
                                    quotes, marketing = {}, {}
                                    for i in active:
                                        city = str(markets[i]["city"])
                                        cap = min(price_max, float(markets[i]["max_price"]))
                                        rival = leader_routes.get(city)
                                        if depth and rival:
                                            anchor = min(float(rival["price"]), previous_prices[i] * .999)
                                            quotes[city] = max(price_min, anchor * (1.-depth))
                                        else:
                                            quotes[city] = cap
                                        if mi and rival:
                                            marketing[i] = mi_thresholds[i] * mi
                                    row = {**budget_seed, "marketing": marketing,
                                           "marketing_total": sum(marketing.values()),
                                           "groups": max(0, round(affordable * fraction)),
                                           "rival_seed": True, "forecast_prices": quotes}
                                    shortlisted[id(row)] = row
                    # The champion may be funded for many more units than we
                    # can sell at their price. Probe lower-cost openers both
                    # on their best routes and on the cities they leave room in.
                    benchmark_routes = benchmark["cities"]
                    by_city_capacity = {m["city"]: m["market_size"] for m in forecast_markets}
                    benchmark_sold = baseline_forecast["companies"][projected_rival]["sold_units"]
                    targets = sorted(
                        (i for i in selected if str(markets[i]["city"]) in benchmark_routes and agent_plan[i][1] > 0),
                        key=lambda i: (
                            -benchmark_sold.get(str(markets[i]["city"]), 0)
                             * benchmark_routes[str(markets[i]["city"])]["price"], i),
                    )
                    focus = targets[:max(1, min(3, len(targets)))]
                    if race_for_first and use_complete_field:
                        # A rival can sell in every city. Undercut only its best
                        # routes and keep premium prices on the rest of the map.
                        active_routes = [i for i in selected if agent_plan[i][1] > 0]
                        for target_count in sorted({2, max(2, len(targets) // 2), max(2, len(targets) - 2)}):
                            contested = set(targets[:target_count])
                            if not contested or contested.issuperset(active_routes):
                                continue
                            for depth in (.015, .08, .23):
                                for ma, qi, mi_strength in ((1., 0., 0.), (ma_threshold * 1.25, 1., 1.25),
                                                            (1., qi_line * 1.3, 1.9)):
                                    seed = evaluate_candidate(ma, qi, 0., .95, True, ())
                                    marketing = {i: mi_thresholds[i] * mi_strength for i in contested if mi_strength}
                                    prices = {}
                                    for i in active_routes:
                                        city = str(markets[i]["city"])
                                        cap = min(price_max, float(markets[i]["max_price"]))
                                        prices[city] = (min(benchmark_routes[city]["price"], previous_prices[i] * .999)
                                                        * (1. - depth) if i in contested else cap)
                                    for fraction in (.50, .78, 1.):
                                        row = {**seed, "marketing": marketing, "marketing_total": sum(marketing.values()),
                                               "groups": max(0, round(int(seed["groups"]) * fraction)),
                                               "rival_seed": True, "forecast_prices": prices}
                                        shortlisted[id(row)] = row
                    focus_profiles = ((1., 0., 0.), (ma_threshold * 1.1, 1., .9),
                                      (float(benchmark["ma_index"]) * .62, float(benchmark["qi_index"]) * .65, 1.2))
                    if not human_present and not use_complete_field:
                        focus_profiles = focus_profiles[:1]
                    for ma, qi, mi_strength in focus_profiles:
                        marketing = {i: max(
                            benchmark_routes[str(markets[i]["city"])]["marketing"] * .6,
                            mi_thresholds[i] * mi_strength,
                        ) for i in focus if mi_strength > 0.}
                        seed = evaluate_candidate(ma, qi, 0., .95, True, ())
                        groups = int(seed["groups"])
                        for fraction, undercut in ((.24, .98), (.46, .995), (.67, .94)):
                            prices = {}
                            for i in selected:
                                city = str(markets[i]["city"])
                                cap = min(price_max, float(markets[i]["max_price"]))
                                if i in focus:
                                    prices[city] = min(
                                        benchmark_routes[city]["price"] * undercut,
                                        previous_prices[i] * .99,
                                    )
                                else:
                                    prices[city] = cap * (
                                        .995 if by_city_capacity[city] > sum(by_city_capacity.values()) / len(markets) else .95
                                    )
                            row = {**seed, "marketing": marketing, "groups": max(0, round(groups * fraction)),
                                   "rival_seed": True, "forecast_prices": prices,
                                   "marketing_total": sum(marketing.values())}
                            shortlisted[id(row)] = row
                # Use the whole submitted field to find free demand, rather
                # than treating the previous round's saturation as a quota.
                # A share already sold by a rival remains contestable.
                baseline_city_sales = {
                    market["city"]: sum(
                        company["sold_units"].get(market["city"], 0)
                        for company in baseline_forecast["companies"].values()
                    )
                    for market in forecast_markets
                }
                open_units = sum(
                    max(0.0, market["market_size"] - baseline_city_sales[market["city"]])
                    for market in forecast_markets
                )
                field_size = sum(market["market_size"] for market in forecast_markets)
                # Focus the extra expensive searches where the audit found
                # a real midgame supply opportunity, not in the opening.
                growth_search = (
                    official_round >= 4
                    and (not all_super_field or late_disruptor)
                    and open_units >= field_size * 0.08
                )
                if growth_search or late_disruptor:
                    # MA/QI-heavy plans are expensive to scale. Also try
                    # sensible single-pool routes, including a small MA
                    # entry to access investment secondary sales, and
                    # route even a well-funded MI bid only where useful.
                    available_focus = sorted(selected, key=lambda idx: (
                        -max(0.0, forecast_markets[idx]["market_size"]
                             - baseline_city_sales[forecast_markets[idx]["city"]]),
                        -forecast_markets[idx]["market_size"], idx,
                    ))
                    focus = tuple(sorted(available_focus[:max(1, min(3, len(available_focus)))]))
                    quotes = (1.0, .95, .81)
                    if any(item[2] for item in candidate_city_data):
                        # At least two affordable undercuts: actual city
                        # margins will still be checked after redistribution.
                        center = sorted(previous_prices[idx] / max(1., min(price_max, float(markets[idx]["max_price"])))
                                        for idx in selected)[len(selected) // 2]
                        quotes += (min(.95, max(.01, center * .99)),
                                   min(.88, max(.01, center * .88)))
                    for ratio in dict.fromkeys(quotes):
                        for ma, qi, mi, cities in (
                            (1., 0., 0., ()),
                            (ma_threshold * 1.08, 0., 0., ()),
                            (1., 0., 1.15, focus),
                            (1., 0., 2.2, None),
                        ):
                            if mi and (not use_mi or floor_lock_disruptor):
                                continue
                            seed = evaluate_candidate(ma, qi, mi, ratio, True, cities)
                            maximum = int(seed["groups"])
                            fractions = (
                                (.20, .35, .55, .80, 1.) if late_disruptor and use_complete_field
                                else (.45, .75, 1.) if late_disruptor else (.55, 1.)
                            )
                            for amount in sorted({max(1, round(maximum * part)) for part in fractions} if maximum else {0}):
                                row = {**seed, "groups": amount, "lean_secondary_route": not late_disruptor}
                                if late_disruptor:
                                    row["uniform_late_price"] = True
                                shortlisted[id(row)] = row
                    if late_disruptor:
                        # Keep the floor-lock route available at supply sizes
                        # spanning competing disruptors and existing stock.
                        for row in list(shortlisted.values()):
                            if not row.get("uniform_late_price") or row.get("marketing_total", 0.0):
                                continue
                            max_groups = int(row["groups"])
                            if max_groups <= 1:
                                continue
                            for part in (.55, .8):
                                probe = {**row, "groups": max(1, round(max_groups * part))}
                                shortlisted[id(probe)] = probe
                saved_targets = ma_index_target, qi_index_target, marketing_targets
                rival_player_list = list(rival_players.values())
                prepared_finalists: list[tuple[
                    dict[str, Any], dict[str, Any], int, dict[str, Any]
                ]] = []
                forecast_payloads: list[dict[str, Any]] = []
                for candidate in shortlisted.values():
                    ma_index_target = float(candidate["ma"])
                    qi_index_target = float(candidate["qi"])
                    marketing_targets = dict(candidate["marketing"])
                    candidate_groups = max(0, int(candidate["groups"]))
                    actual_budget = budget_for(int(math.floor(candidate_groups * group["products"])))
                    if float(actual_budget["total"]) > cash_budget + 1e-9:
                        zero_budget = budget_for(0)
                        feasible = 0 if float(zero_budget["total"]) <= cash_budget + 1e-9 else -1
                        feasible_budget = zero_budget
                        low, high = 1, candidate_groups
                        while low <= high:
                            middle = (low + high) // 2
                            trial = budget_for(int(math.floor(middle * group["products"])))
                            if float(trial["total"]) <= cash_budget + 1e-9:
                                feasible, feasible_budget = middle, trial
                                low = middle + 1
                            else:
                                high = middle - 1
                        if feasible < 0:
                            continue
                        candidate_groups, actual_budget = feasible, feasible_budget
                    candidate["groups"] = candidate_groups
                    candidate_available = old_products + int(math.floor(
                        candidate_groups * group["products"]))
                    floor_lock_prices: dict[str, float] = {}
                    if floor_lock_disruptor and not candidate.get("uniform_late_price"):
                        lock_options = []
                        market_max_prices: dict[str, float] = {}
                        for (
                            lock_index, lock_cap, _low_price_unlocked,
                            _direct_unit_cost, _weights, _prices, _entries,
                            lock_size, _evaluator, _average_evaluator,
                            _shadow, _aggregate,
                        ) in candidate_city_data:
                            city = str(markets[lock_index]["city"])
                            market_max_prices[city] = float(lock_cap)
                            human_pressure = sum(
                                (
                                    100.0 + float(rival.get("attack_weight", 0.0))
                                ) * max(1.0, float(rival.get("available", 0.0)))
                                for rival in rival_metrics[lock_index]
                                if bool(rival.get("is_human", False))
                            )
                            lock_options.append({
                                "city": city,
                                "max_price": float(lock_cap),
                                "lock_units": float(lock_size) * PRICE_CPI_TOTAL / 100.0,
                                "attack_weight": human_pressure,
                            })
                        coverable_cities = _disruptor_coverable_cities(
                            lock_options,
                            available_units=candidate_available,
                        )
                        floor_lock_prices = _disruptor_floor_lock_prices(
                            disruptor_ids=disruptor_ids,
                            designated_company_id=floor_lock_company_id,
                            market_max_prices=market_max_prices,
                            coverable_cities=coverable_cities,
                            price_min=price_min,
                        ).get(company_id, {})
                    own_cities = {}
                    for (
                        index, cap, low_price_unlocked, direct_unit_cost,
                        _weights, _prices, _entries, _size, _evaluator,
                        _average_evaluator, _shadow, _aggregate,
                    ) in candidate_city_data:
                        ratio = float(candidate["price_ratio"])
                        effective_ratio = (ratio if low_price_unlocked or ratio >= .75
                                           else min(.98, max(.75, high_price_ratio)))
                        city = str(markets[index]["city"])
                        copied_price = candidate.get("forecast_prices", {}).get(city, cap * effective_ratio)
                        own_cities[city] = {
                            "agents": agent_plan[index][1],
                            "marketing": float(candidate["marketing"].get(index, 0.0)),
                            "price": floor_lock_prices.get(
                                city,
                                min(
                                    cap,
                                    max(
                                        price_min,
                                        copied_price,
                                        0.0 if late_disruptor else direct_unit_cost * 1.03,
                                    ),
                                ),
                            ),
                        }
                    # The submitted city prices must match the settled forecast.
                    candidate["resolved_city_prices"] = {
                        city: float(data["price"]) for city, data in own_cities.items()
                    }
                    players = [
                        *rival_player_list,
                        {"company_id": company_id, "available": candidate_available,
                         "ma_index": float(candidate["ma"]), "qi_index": float(candidate["qi"]),
                         "cities": own_cities},
                    ]
                    prepared_finalists.append((
                        candidate, own_cities, candidate_available, actual_budget,
                    ))
                    forecast_payloads.append({"players": players, **forecast_args})
                unresolved_super_peers = (
                    0
                    if late_disruptor
                    else sum(
                        1 for peer in bots
                        if int(peer["id"]) != company_id
                        and int(peer["id"]) not in frozen_super_ids
                    )
                )
                secondary_confidence = (
                    1.0 if expansion_room
                    else 1.0 / (1.0 + 0.45 * unresolved_super_peers)
                )
                verified_candidates = []
                base_prices = {
                    market["city"]: market["base_average_price"] for market in forecast_markets
                }

                def score_exact(prepared, after_forecast):
                    candidate, own_cities, candidate_available, actual_budget = prepared
                    own_forecast = after_forecast["companies"][company_id]
                    candidate["forecast_city_sold"] = own_forecast["sold_units"]
                    # Secondary allocation is part of the plan, but an early
                    # sequential Super Bot cannot assume every synthetic peer
                    # will leave that pool untouched. Confidence rises to 100%
                    # as later real decisions replace those presets. The final
                    # joint pass then uses the exact complete field.
                    _apply_super_settlement_forecast(
                        candidate,
                        own_forecast,
                        own_cities,
                        candidate_available=candidate_available,
                        home_city=home,
                        transportation_cost=transport_cost,
                        committed_cost=float(actual_budget["total"]),
                        interest_cost=forecast_interest,
                        tax_rate=tax_rate,
                        starting_assets=own_strategy_assets,
                        secondary_confidence=secondary_confidence,
                    )
                    if late_disruptor:
                        candidate["low_price_sold"] = sum(
                            units for city, units in own_forecast["sold_units"].items()
                            if city in own_cities and own_cities[city]["price"] < base_prices[city]
                        )
                        candidate["committed_cost"] = float(actual_budget["total"])
                    if baseline_forecast is not None:
                        candidate.update(_super_target_pressure(
                            baseline_forecast, after_forecast, rival_player_list))
                    if projected_rival is not None:
                        candidate["projected_rival_profit"] = rival_profit(projected_rival, after_forecast)
                    if race_for_first and rival_baseline:
                        candidate["first_place_gap"] = own_strategy_assets + candidate["predicted_profit"] - max(
                            rival_baseline[i][2] + rival_profit(i, after_forecast) for i in rival_baseline
                        )
                    verified_candidates.append(candidate)

                for prepared, after_forecast in zip(
                    prepared_finalists, _forecast_candidates(forecast_payloads),
                ):
                    score_exact(prepared, after_forecast)

                # Refine a few plans using settled sales, not visible CPI.
                # Different output can change the weighted price and the
                # second pass, so every refinement gets a fresh full forecast.
                refinements = []
                refinement_keys = set()
                full_group_budgets = {}
                if prepared_finalists and candidate_city_data and not late_disruptor:
                    # Compare city-specific prices and bids using the same
                    # exact settlement as the existing uniform grid.
                    bid_cities = candidate_city_data
                    field = {entry["company_id"]: entry for row in bid_cities for entry in row[6][:-1]}
                    stretch = 1.22 + (company_id % 11) * .025
                    over_ma = _super_index_overbid(ma_threshold, 0., [r["ma_index"] for r in field.values()], stretch)
                    over_qi = _super_index_overbid(qi_line, 0., [r["qi_index"] for r in field.values()], stretch)
                    priority = sorted(bid_cities, key=lambda row: (
                        baseline_city_sales.get(str(markets[row[0]]["city"]), 0.) - row[7] * 1.12,
                        (row[0] - company_id) % len(markets),
                    ))
                    focus_count = 3 if len(priority) > 4 else 2
                    focus = {row[0] for row in priority[:focus_count]}
                    ranked = sorted(prepared_finalists, key=lambda p: (p[0]["predicted_profit"], p[0]["risk_profit"]), reverse=True)
                    seeds = [ranked[0]]
                    premium_seed = next((p for p in ranked if any(
                        p[1][str(markets[i]["city"])]["price"] >= cap * .80 for i, cap, *_ in bid_cities)), None)
                    growth_seed = max(ranked, key=lambda p: (
                        p[0]["predicted_profit"] + .10 * p[0]["predicted_revenue"]
                        if p[0]["sell_ratio"] >= .78 else float("-inf"), p[2],
                    ))
                    for seed in (premium_seed, growth_seed):
                        if seed is not None and all(seed[0] is not item[0] for item in seeds):
                            seeds.append(seed)

                    for candidate, cities, _, _ in seeds:
                        options = {}
                        for index, cap, unlocked, direct_cost, *_ in bid_cities:
                            city = str(markets[index]["city"])
                            options[index] = _super_city_bid_options(
                                quote=cities[city]["price"], cap=cap,
                                minimum=max(price_min, direct_cost * 1.03),
                                average=previous_prices[index], price_power=price_power,
                                allow_low=unlocked, rivals=rival_metrics[index],
                                threshold=mi_thresholds[index], agents=cities[city]["agents"],
                                seed=company_id * 137 + index * 41 + official_round * 59,
                            )
                        base_ma, base_qi = float(candidate["ma"]), float(candidate["qi"])
                        variants = [
                            ("mild", base_ma, base_qi, 1.), ("discount", base_ma, base_qi, 1.25),
                            ("price_only", 0., 0., 1.42),
                            ("lean", ma_threshold * (.12 + (company_id % 7) * .025),
                             qi_line * .22 if use_qi else 0., 1.35),
                            ("ma_push", max(base_ma * stretch, over_ma), base_qi, 1.13),
                        ]
                        if use_qi:
                            variants.append(("qi_push", base_ma, max(base_qi * stretch, over_qi), 1.13))
                        if plan_index >= 2:
                            variants.extend((
                                ("mi_push", base_ma, base_qi, 1.14),
                                ("mi_trim", base_ma, base_qi, 1.25),
                                ("mi_shift", base_ma, base_qi, 1.05),
                                ("mi_discount", min(base_ma, ma_threshold), min(base_qi, qi_line), 1.30),
                                ("mixed_push", max(base_ma * 1.12, over_ma),
                                 max(base_qi * 1.12, over_qi) if use_qi else 0., 1.10),
                            ))
                        # Move some existing MI from low-converting cities to
                        # promising ones without raising total MI spend.
                        city_sales = candidate["forecast_city_sold"]
                        mi_order = sorted(priority, key=lambda row: (
                            -city_sales.get(str(markets[row[0]]["city"]), 0.)
                            * cities[str(markets[row[0]]["city"])]["price"]
                            / max(1., cities[str(markets[row[0]]["city"])]["marketing"] + mi_thresholds[row[0]] * .35),
                        ))
                        mi_receivers = {row[0] for row in mi_order[:focus_count]}
                        mi_donations = {
                            i: cities[str(markets[i]["city"])]["marketing"] * .35
                            for i in options if i not in mi_receivers
                        }
                        mi_reassigned = sum(mi_donations.values())
                        if not mi_reassigned:
                            variants = [variant for variant in variants if variant[0] != "mi_shift"]
                        for kind, ma, qi, scale in variants:
                            revised = {city: dict(route) for city, route in cities.items()}
                            for i, (mild, medium, premium, deep, small, strong) in options.items():
                                route = revised[str(markets[i]["city"])]
                                focused = i in focus
                                if kind == "mild":
                                    route["price"] = mild
                                elif kind == "discount":
                                    route["price"] = medium if focused else mild
                                elif kind == "price_only":
                                    route.update(price=deep if focused else medium, marketing=0.)
                                elif kind == "mi_trim":
                                    # A small markdown may replace expensive MI;
                                    # keep only the more relevant local bids.
                                    route["price"] = mild if route["marketing"] else route["price"]
                                    route["marketing"] = min(route["marketing"], small) if focused else route["marketing"] * .65
                                elif kind == "mi_shift":
                                    if i in mi_receivers:
                                        route["marketing"] += mi_reassigned / len(mi_receivers)
                                    else:
                                        route["marketing"] -= mi_donations[i]
                                        route["price"] = mild
                                elif kind in {"lean", "mi_discount"}:
                                    deep_mix = kind == "mi_discount"
                                    route["price"] = (deep if deep_mix else medium) if focused else mild
                                    allowance = cash_budget * (.27 if deep_mix else .09) / max(1, len(focus))
                                    route["marketing"] = min(strong if deep_mix else small, allowance) if focused and plan_index >= 2 else 0.
                                else:
                                    route["price"] = premium if focused else route["price"]
                                    if kind in {"mi_push", "mixed_push"} and focused:
                                        bid = max(strong, route["marketing"] * stretch)
                                        route["marketing"] = max(route["marketing"], min(bid, cash_budget * .32 / max(1, len(focus))))

                            marketing = {i: revised[str(markets[i]["city"])]["marketing"] for i in options}
                            ma_index_target, qi_index_target, marketing_targets = ma, qi, marketing
                            budget_key = (ma, qi, tuple(sorted(marketing.items())))
                            if budget_key not in full_group_budgets:
                                full_group_budgets[budget_key] = _max_budgeted_groups(
                                    cash_budget, group["products"], max(candidate["groups"], formula_groups), budget_for,
                                )
                            limit, max_budget = full_group_budgets[budget_key]
                            groups_to_try = {min(max(1, math.ceil(candidate["groups"] * scale)), limit)}
                            if kind in {"discount", "price_only", "lean", "mi_trim", "mi_discount", "mixed_push"}:
                                groups_to_try.add(limit)
                            prices = {city: route["price"] for city, route in revised.items()}
                            for count in sorted(groups_to_try):
                                key = ("city_bid", budget_key, count, tuple(prices.items()))
                                if count < 0 or (not count and not old_products) or key in refinement_keys:
                                    continue
                                refinement_keys.add(key)
                                output = math.floor(count * group["products"])
                                refinements.append(({
                                    **candidate, "ma": ma, "qi": qi, "groups": count,
                                    "marketing": marketing, "marketing_total": sum(marketing.values()),
                                    "resolved_city_prices": prices, "budget_max_output": count == limit,
                                }, revised, old_products + output, max_budget if count == limit else budget_for(output)))
                expand_late = (not late_disruptor and not competitive_mode
                               and official_round >= max(4, math.ceil(total_rounds * .60)))
                if expand_late:
                    # Start from actual plans that sell out. Try scaling output
                    # and testing different city-level bids, not buying the
                    # same MA/QI/MI ratios in every market.
                    seeds = sorted(
                        (p for p in prepared_finalists if p[0]["sell_ratio"] >= .95 and p[0]["predicted_profit"] > 0),
                        key=lambda p: float(p[0]["predicted_profit"]), reverse=True,
                    )
                    unique = set()
                    for candidate, cities, _, _ in seeds:
                        key = (candidate["ma"], candidate["qi"], tuple(
                            (city, v["price"], v["marketing"]) for city, v in cities.items()))
                        if key in unique:
                            continue
                        unique.add(key)
                        if len(unique) > 3:
                            break
                        ma_index_target, qi_index_target = float(candidate["ma"]), float(candidate["qi"])
                        if not cities:
                            continue
                        own_order = sorted(cities, key=lambda city: (
                            -max(0., next(m["market_size"] for m in forecast_markets if m["city"] == city)
                                 - baseline_city_sales[city]), city))
                        emphasis = (own_order[:max(1,len(own_order)//2)] if profile % 3 != 2
                                    else own_order[::2])
                        for scale in (1.18, 1.42):
                            for diversify in (False, True):
                                count = max(int(candidate["groups"])+1, math.ceil(candidate["groups"]*scale))
                                output = int(math.floor(count*group["products"]))
                                revised = {city: dict(route) for city, route in cities.items()}
                                if diversify:
                                    for city in emphasis:
                                        route = revised[city]
                                        if profile % 3 != 1:
                                            route["price"] = max(price_min, route["price"] * (.99 if profile % 3 == 0 else .975))
                                        if profile % 3 != 0:
                                            route["marketing"] *= 1.15
                                marketing = {i: revised[str(markets[i]["city"])]["marketing"] for i in selected
                                             if str(markets[i]["city"]) in revised}
                                marketing_targets = marketing
                                cost = budget_for(output)
                                if float(cost["total"]) <= cash_budget + 1e-9:
                                    refinements.append(({**candidate, "groups": count, "marketing": marketing,
                                                          "expansion_probe": True,
                                                          "resolved_city_prices": {city: route["price"] for city, route in revised.items()}},
                                                         revised, old_products+output, cost))
                lean = sorted(
                    (p for p in prepared_finalists if p[0].get("lean_secondary_route")),
                    key=lambda p: float(p[0]["predicted_profit"]), reverse=True,
                )[:10]
                for candidate, own_cities, available, _ in lean:
                    sold = float(candidate["predicted_sold"])
                    if sold <= 0 or available - sold < group["products"]:
                        continue
                    count = max(0, math.ceil((sold - old_products) / group["products"]))
                    key = (count, candidate["ma"], candidate["qi"], candidate["price_ratio"])
                    if key in refinement_keys or count >= candidate["groups"]:
                        continue
                    refinement_keys.add(key)
                    ma_index_target = float(candidate["ma"])
                    qi_index_target = float(candidate["qi"])
                    marketing_targets = dict(candidate["marketing"])
                    output = int(math.floor(count * group["products"]))
                    budget_at_sales = budget_for(output)
                    if float(budget_at_sales["total"]) <= cash_budget + 1e-9:
                        refinements.append((
                            {**candidate, "groups": count}, own_cities,
                            old_products + output, budget_at_sales,
                        ))

                if late_disruptor:
                    flat_seeds = sorted(
                        (p for p in prepared_finalists
                         if p[0].get("uniform_late_price") and p[0]["low_price_sold"] > 0),
                        key=lambda p: abs(float(p[0]["predicted_profit"])),
                    )[:3]
                    for candidate, own_cities, available, cost in flat_seeds:
                        revenue = float(candidate["predicted_revenue"])
                        if revenue <= 0:
                            continue
                        # Find the near break-even quote at the observed sales
                        # mix. This is a probe, not an assumption of monotone CPI.
                        cost_to_cover = (
                            float(cost["total"]) + forecast_interest
                            + float(candidate["predicted_transport"])
                        )
                        for margin in (.001, .005):
                            scale = (1 + margin) * cost_to_cover / revenue
                            adjusted = {
                                city: {**route, "price": min(
                                    city_caps_by_name[city],
                                    max(price_min, float(route["price"]) * scale),
                                )}
                                for city, route in own_cities.items()
                            }
                            key = (
                                candidate["groups"],
                                tuple(round(float(route["price"]), 4) for route in adjusted.values()),
                            )
                            if key in refinement_keys:
                                continue
                            refinement_keys.add(key)
                            refinements.append((
                                {**candidate, "resolved_city_prices": {
                                    city: route["price"] for city, route in adjusted.items()
                                }},
                                adjusted, available, cost,
                            ))
                refinement_payloads = [
                    dict(forecast_args, players=[
                        *rival_player_list,
                        {"company_id": company_id, "available": available,
                         "ma_index": float(candidate["ma"]), "qi_index": float(candidate["qi"]),
                         "cities": cities},
                    ])
                    for candidate, cities, available, _ in refinements
                ]
                for prepared, after_forecast in zip(
                    refinements, _forecast_candidates(refinement_payloads),
                ):
                    score_exact(prepared, after_forecast)
                ma_index_target, qi_index_target, marketing_targets = saved_targets
                finalist_candidates = verified_candidates
            else:
                finalist_candidates = []

            # Select by relative advantage across every evaluated plan. Profiles
            # still generate different candidate grids and rival responses, but
            # they cannot override the economic result after the search.
            if late_disruptor:
                # Keep the floor-lock plan in the search, but let an
                # affordable alternative win when the floor cannot pay costs.
                empirical_candidate = _select_disruptor_attack_candidate(
                    finalist_candidates,
                )
            else:
                empirical_candidate = _select_empirical_super_candidate(
                    finalist_candidates,
                    profile,
                    tactical_price_allowed=bool(
                        late_game_low_price
                        or saturated_low_price_indices
                        or prior_surplus_ratio >= 0.50
                        or all_super_field
                        or market_pressure >= 0.60
                    ),
                    profit_target=profit_target,
                    competitive_mode=competitive_mode,
                    contest_market=True,
                    expand_at_equal_profit=bool(expand_late) if finalist_candidates else False,
                    rival_benchmark=(rival_profit(projected_rival, baseline_forecast)
                                     if finalist_candidates and projected_rival is not None else None),
                    race_for_first=race_for_first,
                    peer_field=not human_present,
                    sacrifice_allowed=bool(
                        last_round
                        and competitive_mode
                        and next(
                            (
                                rival_id for rival_id in reversed(ranked_super_ids)
                                if rival_id in competitive_super_ids
                            ),
                            None,
                        ) == company_id
                    ),
                )
            if empirical_candidate is not None:
                best_candidate = empirical_candidate

            if best_candidate is not None:
                chosen_predicted_profit = float(best_candidate.get("predicted_profit", 0.0))
                chosen_risk_profit = float(best_candidate.get("risk_profit", chosen_predicted_profit))
                chosen_sold = float(best_candidate.get("predicted_sold", 0.0))
                ma_index_target = float(best_candidate["ma"])
                qi_index_target = float(best_candidate["qi"])
                marketing_targets = dict(best_candidate["marketing"])
                # Preserve the profit-selected production level. The former
                # post-search expansion silently replaced it with the maximum
                # cash-affordable output and forced low-price liquidation.
                chosen_groups = max(0, int(best_candidate["groups"]))
                production = int(math.floor(chosen_groups * group["products"]))
                budget = budget_for(production)
                if float(budget["total"]) > cash_budget + 1e-9:
                    # The search uses normalized groups; real cohorts also
                    # carry layoff, training and old-stock costs. Recheck those
                    # obligations before saving the selected investments.
                    def affordable_selected_groups(limit: int) -> tuple[int, dict[str, Any]]:
                        zero_budget = budget_for(0)
                        low, high = 1, limit
                        affordable = 0 if float(zero_budget["total"]) <= cash_budget + 1e-9 else -1
                        selected_budget = zero_budget
                        # Positive complete groups have nondecreasing costs.
                        # Zero output is checked separately because an
                        # inventory-only MA plan retains one control employee.
                        while low <= high:
                            middle = (low + high) // 2
                            trial = budget_for(int(math.floor(middle * group["products"])))
                            if float(trial["total"]) <= cash_budget + 1e-9:
                                affordable, selected_budget = middle, trial
                                low = middle + 1
                            else:
                                high = middle - 1
                        return affordable, selected_budget

                    affordable_groups, affordable_budget = affordable_selected_groups(chosen_groups)
                    if affordable_groups < 0:
                        # If fixed obligations already exhaust the cash,
                        # request no optional investments or agent changes.
                        # Settlement still applies any unavoidable severance
                        # debt under the ordinary game rules.
                        ma_index_target = qi_index_target = 0.0
                        use_qi = use_mi = False
                        marketing_targets = {index: 0.0 for index in marketing_targets}
                        agent_plan = {
                            index: (0, agents_after - delta)
                            for index, (delta, agents_after) in agent_plan.items()
                        }
                        agent_cost = 0.0
                        distress_liquidation = True
                        affordable_groups, affordable_budget = affordable_selected_groups(chosen_groups)
                    chosen_groups = max(0, affordable_groups)
                    production = int(math.floor(chosen_groups * group["products"]))
                    budget = affordable_budget
                    chosen_predicted_profit = 0.0
                    chosen_risk_profit = chosen_sold = 0.0
                city_rows = make_city_rows(
                    False,
                    float(best_candidate["price_ratio"]),
                )
                # Keep each city price exactly as tested, for both uniform
                # and differentiated candidates.
                resolved_city_prices = best_candidate.get("resolved_city_prices", {})
                if resolved_city_prices:
                    for city_row in city_rows:
                        city = str(city_row["city"])
                        if city in resolved_city_prices:
                            city_row["price"] = float(resolved_city_prices[city])

        active_rows = [row for row in city_rows if int(row["agents_after"]) > 0 and int(row["index"]) in selected]
        weight_total = sum(
            float(markets[int(row["index"])]["population"]) * float(markets[int(row["index"])]["penetration"])
            for row in active_rows
        )
        average_sale_price = (
            sum(
                float(row["price"]) * float(markets[int(row["index"])]["population"]) * float(markets[int(row["index"])]["penetration"])
                for row in active_rows
            ) / max(1.0, weight_total)
        )
        prior_sell_ratio = prior_sold / prior_total if prior_production and prior_total > 0 else 0.95
        expected_sell_ratio = 0.95 if market_pressure < 0.45 else max(0.68, min(0.90, prior_sell_ratio))
        if super_mode:
            expected_sell_ratio = min(1.0, forecast_super_capacity(city_rows) / max(1, old_products + production))
            expected_sell_ratio = max(0.0, min(0.90, expected_sell_ratio))
        non_home_weight = sum(
            float(markets[int(row["index"])]["population"]) * float(markets[int(row["index"])]["penetration"])
            for row in active_rows if str(row["city"]) != home
        )
        projected_transport = (old_products + production) * expected_sell_ratio * transport_cost * non_home_weight / max(1.0, weight_total)
        projected_cash = (
            cash_budget - float(budget["total"])
            + (old_products + production) * expected_sell_ratio * average_sale_price
            - projected_transport
        )
        if super_mode and chosen_predicted_profit is not None:
            # The exact candidate profit already includes tax and capitalised
            # interest. Research is paid from sales cash, so allow early R&D
            # without reserving its cost before production or double-counting
            # the incoming revenue.
            projected_cash = cash_budget + chosen_predicted_profit + forecast_interest
        operating_reserve = max(float(bot["cash"]) * 0.06, float(budget["staff_cost"]) * 0.50)
        research_ready = (
            not last_round and research_needed > 0
            and not distress_liquidation
            and projected_cash - research_needed >= operating_reserve
        )
        research = research_needed if research_ready and not super_mode else 0.0
        if super_mode and not late_disruptor and chosen_predicted_profit is not None:
            research = _profitable_patent_budget(
                balance=research_balance, remaining_rounds=total_rounds-official_round,
                material_spend=float(budget["material_cost"]), sold=chosen_sold,
                available=old_products+production, projected_cash=projected_cash,
                reserve=operating_reserve, risk_profit=chosen_risk_profit,
                amount_25=setting("research_25", 1_500_000), amount_75=setting("research_75", 6_000_000),
                hidden_cap=setting("research_probability_cap", .43),
                hidden_multiplier=setting("research_hidden_threshold_multiplier", 4/3),
                buffer=setting("research_buffer", 150_000), patent_factor=patent_factor, tax_rate=tax_rate,
            )

        # Replace only this Bot, and only after its complete candidate search
        # has succeeded. If analysis stops before this point, its old decision
        # and every previously saved Bot remain untouched.
        if super_mode and existing_decision:
            conn.execute(
                "DELETE FROM city_decisions WHERE company_id=? AND round_no=?",
                (company_id, round_no),
            )
            conn.execute(
                "DELETE FROM decisions WHERE company_id=? AND round_no=?",
                (company_id, round_no),
            )
        conn.execute(
            "INSERT INTO decisions(company_id,round_no,loan_change,worker_delta,worker_salary,engineer_delta,"
            "engineer_salary,management_investment,production_volume,quality_investment,research_investment,submitted_at,is_draft) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (company_id, round_no, loan_change, budget["worker_delta"], worker_salary, budget["engineer_delta"],
             engineer_salary, budget["management"], production, budget["quality"], research,
             (existing_decision["submitted_at"] if use_complete_field and existing_decision else now_iso()),
             (int(existing_decision["is_draft"]) if use_complete_field and existing_decision else int(super_mode))),
        )
        for row in city_rows:
            conn.execute(
                "INSERT INTO city_decisions(company_id,round_no,city,agent_delta,marketing_investment,price,order_report) "
                "VALUES(?,?,?,?,?,?,0)",
                (company_id, round_no, row["city"], row["agent_delta"], row["marketing"], row["price"]),
            )
        # Release the SQLite writer lock between expensive Super Bot searches.
        # A rerun resumes after this saved Bot instead of rebuilding it.
        if super_mode:
            conn.commit()
            frozen_super_ids.add(company_id)
        submitted += 1
        if progress_callback:
            progress_callback(submitted, max(1, progress_total), str(bot["code"]))
    if super_mode and submitted and not defer_rebalance:
        _rebalance_super_bot_production(conn, round_no, markets)
        if progress_callback:
            progress_callback(progress_total, progress_total, "联合复算")
    return submitted


def submit_bot_decisions(conn: sqlite3.Connection, round_no: int) -> int:
    """Ordinary bots submit immediately when a round opens."""
    from .ordinary_bots import submit_ordinary_decisions

    return submit_ordinary_decisions(conn, round_no)


def _submit_super_bot_decisions_locked(
    conn: sqlite3.Connection,
    round_no: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
    replace_existing: bool = False,
    *,
    target_ids: set[int] | None = None,
    defer_rebalance: bool = False,
) -> int:
    """Super bots wait until every non-super team has submitted."""
    missing = one(
        conn,
        "SELECT COUNT(*) AS n FROM companies c WHERE c.is_super_bot=0 AND NOT EXISTS "
        "(SELECT 1 FROM decisions d WHERE d.company_id=c.id AND d.round_no=? "
        "AND d.submitted_at IS NOT NULL AND d.is_draft=0)",
        (round_no,),
    )
    if missing and int(missing["n"]) > 0:
        raise ValueError("仍有真人玩家或普通 Bot 未提交，超级 Bot 暂不能读取本轮数据。")
    # Existing decisions remain in place during analysis. A normal call resumes
    # only missing Bots; explicit reanalysis replaces each Bot individually
    # after its new decision is ready.
    conn.commit()
    return _submit_bots(
        conn,
        round_no,
        True,
        progress_callback,
        replace_existing=replace_existing,
        target_ids=target_ids,
        defer_rebalance=defer_rebalance,
    )


def submit_super_bot_decisions(
    conn: sqlite3.Connection,
    round_no: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
    replace_existing: bool = False,
    *,
    target_ids: set[int] | None = None,
    defer_rebalance: bool = False,
) -> int:
    """Serialize expensive submissions inside one Streamlit worker process."""
    with _SUPER_BOT_SUBMISSION_LOCK:
        return _submit_super_bot_decisions_locked(
            conn,
            round_no,
            progress_callback,
            replace_existing,
            target_ids=target_ids,
            defer_rebalance=defer_rebalance,
        )


def finalize_super_bot_decisions(conn: sqlite3.Connection, round_no: int) -> int:
    """Turn a complete analysed Super Bot batch into locked submissions."""
    with _SUPER_BOT_SUBMISSION_LOCK:
        total = int(one(
            conn,
            "SELECT COUNT(*) AS n FROM companies WHERE is_bot=1 AND is_super_bot=1",
        )["n"])
        if total <= 0:
            return 0
        market_total = int(one(conn, "SELECT COUNT(*) AS n FROM market_config")["n"])
        ready = int(one(
            conn,
            "SELECT COUNT(*) AS n FROM companies c JOIN decisions d ON d.company_id=c.id "
            "WHERE c.is_bot=1 AND c.is_super_bot=1 AND d.round_no=? "
            "AND d.submitted_at IS NOT NULL AND (SELECT COUNT(*) FROM city_decisions cd "
            "WHERE cd.company_id=c.id AND cd.round_no=?)=?",
            (round_no, round_no, market_total),
        )["n"])
        if ready != total:
            raise ValueError("超级 Bot 草稿尚未全部分析完成，不能正式提交。")
        conn.execute(
            "UPDATE decisions SET submitted_at=?,is_draft=0 WHERE round_no=? "
            "AND company_id IN (SELECT id FROM companies WHERE is_bot=1 AND is_super_bot=1)",
            (now_iso(), round_no),
        )
        conn.commit()
        return total


def rebalance_super_bot_decisions(conn: sqlite3.Connection, round_no: int) -> int:
    """Finish a split submission after every team's decision has been saved."""
    with _SUPER_BOT_SUBMISSION_LOCK:
        missing = one(
            conn,
            "SELECT COUNT(*) AS n FROM companies c WHERE NOT EXISTS "
            "(SELECT 1 FROM decisions d WHERE d.company_id=c.id AND d.round_no=? "
            "AND d.submitted_at IS NOT NULL)",
            (round_no,),
        )
        if missing and int(missing["n"]) > 0:
            raise ValueError("仍有玩家或 Bot 未提交，暂不能进行联合复算。")
        markets = [dict(row) for row in all_rows(conn, "SELECT * FROM market_config ORDER BY city")]
        if not markets:
            return 0
        _rebalance_super_bot_production(conn, round_no, markets)
        conn.commit()
        count = one(conn, "SELECT COUNT(*) AS n FROM companies WHERE is_bot=1 AND is_super_bot=1")
        return int(count["n"] if count else 0)
