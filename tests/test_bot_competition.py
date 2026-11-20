from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sim.bot_market_forecast import forecast_market_sales


class BotCompetitionTest(unittest.TestCase):
    def setUp(self):
        # Importing the planner initializes sim.db, so isolate that first import.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        environment = patch.dict(
            os.environ,
            {"SIM_DB_PATH": str(Path(directory.name) / "competition.db")},
        )
        environment.start()
        self.addCleanup(environment.stop)
        from sim import bots

        self.bots = bots

    def test_city_bid_options_probe_small_discounts_and_different_team_quotes(self):
        def options(seed, rival_price=19_000, rival_mi=600_000, agents=2, allow_low=True):
            return self.bots._super_city_bid_options(
                quote=19_000, cap=25_000, minimum=3_500, average=20_000,
                price_power=20, allow_low=allow_low, threshold=100_000,
                rivals=[{"price": rival_price, "mi_effective": rival_mi}],
                agents=agents, seed=seed,
            )

        first = options(33)
        self.assertEqual(first[:2], (18_900, 18_800))
        self.assertEqual(first, options(33))
        self.assertNotEqual(first[:2], options(39)[:2])
        self.assertLess(options(33, rival_price=17_000)[1], first[1])
        self.assertLess(options(33, agents=5)[5], first[5])
        self.assertGreater(options(33, rival_mi=1_200_000)[5], first[5])
        self.assertGreaterEqual(options(33, rival_price=6_000, allow_low=False)[3], 18_750)

    def test_investment_overbid_can_cross_the_rival_index_and_old_cap(self):
        self.assertGreater(self.bots._super_index_overbid(1300, 2500, [2500], 1.6), 2500)
        self.assertGreater(self.bots._super_index_overbid(1300, 5000, [9000]), 8000)

    def test_price_pool_bids_price_the_actual_city_competition(self):
        bid = self.bots._super_low_pool_quote
        params = dict(average=20_000., cap=25_000., minimum=3500., price_power=20, seed=17)
        opponents = [{"price": 19_000.}, {"price": 23_000.}]
        shallow = bid(**params, rivals=opponents, weight_share=.55)
        deep = bid(**params, rivals=opponents, weight_share=1.45)
        self.assertLess(deep, 19_000.)
        self.assertGreater(shallow, 19_000.)
        self.assertLess(shallow, 20_000.)
        self.assertEqual(bid(**{**params, "minimum": 20_000.}, rivals=opponents, weight_share=1.), None)

    def test_low_price_overflow_has_a_single_city_alternative(self):
        markets = [("广州", 90_000, 250_000), ("深圳", 20_000, 85_000), ("杭州", 10_000, 50_000)]
        select = self.bots._super_low_pool_subsets
        self.assertEqual(select(markets, 70_000)[0], {"广州"})
        self.assertIn({"广州", "深圳"}, select(markets, 100_000))
        self.assertNotIn({"广州", "深圳", "杭州"}, select(markets, 180_000))

    def test_small_price_step_changes_investment_cpi_as_well_as_price_cpi(self):
        def cpi(price):
            entries = [
                {"company_id": 1, "ma_index": 3500, "qi_index": 1100,
                 "mi_investment": 4_000_000, "agents": 3, "price": price},
                {"company_id": 2, "ma_index": 3500, "qi_index": 1100,
                 "mi_investment": 4_000_000, "agents": 3, "price": 19_000},
            ]
            rules = {"market_size": 100_000, "max_price": 25_000, "ma_large_threshold": 1300}
            averages = self.bots.investment_average_prices(entries, [2000, 4000], fallback=20_000, **rules)
            return self.bots.allocate_city_cpi(
                entries, **rules, average_price=averages, market_average_price=20_000, price_power=20,
            )[0]

        equal, small, larger = [cpi(price) for price in (19_000, 18_900, 18_800)]
        for pool in ("ma_cpi", "qi_cpi", "mi_cpi", "price_cpi"):
            with self.subTest(pool=pool):
                self.assertLess(equal[pool], small[pool])
                self.assertLess(small[pool], larger[pool])

    def test_target_priority_preserves_team_and_rank_order(self):
        cases = (
            ("higher human", True, False, 1, 0.0, 5),
            ("higher Super Bot", False, True, 2, 0.0, 4),
            ("profitable normal Bot", False, False, 3, 0.5, 3),
            ("lower human", True, False, 9, 0.9, 2),
            ("lower Super Bot", False, True, 9, 0.9, 1),
            ("unprofitable normal Bot", False, False, 1, -0.2, 1),
        )
        priorities = []
        for label, is_human, is_super, opponent_rank, margin, expected in cases:
            with self.subTest(target=label):
                actual = self.bots._super_attack_priority(
                    is_human=is_human,
                    is_super=is_super,
                    opponent_rank=opponent_rank,
                    own_rank=4,
                    predicted_margin=margin,
                )
                self.assertEqual(actual, expected)
                priorities.append(actual)
        self.assertEqual(priorities, [5, 4, 3, 2, 1, 1])

    def test_super_candidate_profit_includes_secondary_sales(self):
        candidate = {
            "predicted_sold": 10,
            "predicted_profit": -1,
            "coverage": 0.10,
            "rival_damage": 123,
        }
        forecast = {
            "sold_units": {"Home": 40, "Away": 60},
            "sold_total": 100,
            "visible_total": 10,
            "secondary_total": 90,
        }
        updated = self.bots._apply_super_settlement_forecast(
            candidate,
            forecast,
            {"Home": {"price": 10}, "Away": {"price": 20}},
            candidate_available=100,
            home_city="Home",
            transportation_cost=2,
            committed_cost=100,
            interest_cost=0,
            tax_rate=0,
            starting_assets=1_000,
        )

        self.assertEqual(updated["predicted_sold"], 100)
        self.assertEqual(updated["primary_capacity"], 10)
        self.assertEqual(updated["secondary_capacity"], 90)
        self.assertEqual(updated["coverage"], 1)
        self.assertEqual(updated["predicted_revenue"], 1_600)
        self.assertEqual(updated["predicted_transport"], 120)
        self.assertEqual(updated["predicted_profit"], 1_380)
        self.assertEqual(updated["ending_assets"], 2_380)

    def test_pending_super_peers_discount_only_secondary_sales(self):
        updated = self.bots._apply_super_settlement_forecast(
            {},
            {
                "sold_units": {"Home": 100},
                "primary": {"Home": 40},
                "secondary": {"Home": 60},
                "visible_total": 40,
                "secondary_total": 60,
            },
            {"Home": {"price": 10}},
            candidate_available=100,
            home_city="Home",
            transportation_cost=0,
            committed_cost=0,
            interest_cost=0,
            tax_rate=0,
            starting_assets=0,
            secondary_confidence=0.5,
        )

        self.assertEqual(updated["predicted_sold"], 70)
        self.assertEqual(updated["primary_capacity"], 40)
        self.assertEqual(updated["secondary_capacity"], 30)
        self.assertEqual(updated["raw_secondary_capacity"], 60)
        self.assertEqual(updated["predicted_profit"], 700)

    def test_attacker_counts_protect_leaders_and_change_by_phase(self):
        cases = (
            ([], 6, True, set()),
            ([11], 6, True, set()),
            ([11, 12], 6, False, {12}),
            ([11, 12, 13], 6, True, set()),
            ([11, 12, 13, 14], 6, False, {14}),
            ([1, 2, 3, 4, 5], 3, True, {5}),
            ([1, 2, 3, 4, 5, 6], 3, True, {5, 6}),
            (list(range(1, 10)), 3, False, set()),
            (list(range(1, 10)), 6, False, {7, 8, 9}),
            (list(range(1, 10)), 7, False, {7, 8, 9}),
        )
        for ranked, round_no, saturated, expected in cases:
            with self.subTest(ranked=ranked, round_no=round_no, saturated=saturated):
                self.assertEqual(
                    self.bots._competitive_super_attackers(
                        ranked,
                        official_round=round_no,
                        total_rounds=7,
                        markets_saturated=saturated,
                    ),
                    expected,
                )

    def test_losing_field_does_not_enable_unapproved_capital_sacrifice(self):
        choices = [
            {"name": "least-loss", "predicted_profit": -50, "starting_assets": 10000},
            {"name": "extra-loss", "predicted_profit": -500, "starting_assets": 10000,
             "targeted_damage": 1000, "target_tier": 5},
        ]
        chosen = self.bots._select_empirical_super_candidate(
            choices, 0, tactical_price_allowed=True,
            competitive_mode=True, sacrifice_allowed=False,
        )
        self.assertEqual(chosen["name"], "least-loss")

    def test_every_super_can_contest_real_sales_without_being_a_designated_attacker(self):
        choices = [
            {"name": "yield-market", "predicted_profit": 100},
            {"name": "capture-market", "predicted_profit": 96, "targeted_damage": 40,
             "target_surplus": 10, "target_cpi_drop": 5},
            {"name": "unnecessary-crash", "predicted_profit": 50, "targeted_damage": 10000},
        ]
        chosen = self.bots._select_empirical_super_candidate(
            choices, 0, tactical_price_allowed=True, contest_market=True,
        )
        self.assertEqual(chosen["name"], "capture-market")
        # Visible CPI suppression without actual lost sales is not a reason
        # to pay extra: redistribution may still let the rival sell out.
        choices[1]["targeted_damage"] = 0
        chosen = self.bots._select_empirical_super_candidate(
            choices, 0, tactical_price_allowed=True, contest_market=True,
        )
        self.assertEqual(chosen["name"], "yield-market")

    def test_large_visible_cpi_drop_can_cause_no_target_surplus(self):
        markets = [{
            "city": "A", "market_size": 100_000,
            "max_price": 25_000, "base_average_price": 10_000,
        }]
        target = {
            "company_id": 1, "available": 30_000, "ma_index": 0, "qi_index": 0,
            "attack_weight": 5,
            "cities": {"A": {"agents": 1, "marketing": 0, "price": 8_000}},
        }
        attacker = {
            "company_id": 2, "available": 1_000, "ma_index": 0, "qi_index": 0,
            "cities": {"A": {"agents": 1, "marketing": 0, "price": 6_000}},
        }
        before_forecast = forecast_market_sales(
            players=[target], markets=markets,
            ma_large_threshold=1_300, price_power=8,
        )
        after_forecast = forecast_market_sales(
            players=[target, attacker], markets=markets,
            ma_large_threshold=1_300, price_power=8,
        )
        before = before_forecast["companies"][1]
        after = after_forecast["companies"][1]

        # Visible CPI alone would claim almost 30,000 newly unsold units.
        claimed_loss = min(target["available"], before["visible_total"]) - min(
            target["available"], after["visible_total"],
        )
        self.assertGreater(claimed_loss, 29_000)
        # Unused price capacity returns through settlement's secondary pool.
        self.assertGreater(after["secondary_total"], 29_000)
        self.assertEqual(before["sold_total"], 30_000)
        self.assertEqual(after["sold_total"], 30_000)
        self.assertEqual(after["available"] - after["sold_total"], 0)
        pressure = self.bots._super_target_pressure(before_forecast, after_forecast, [target])
        self.assertEqual(pressure["targeted_damage"], 0)
        self.assertEqual(pressure["target_surplus"], 0)

    def test_real_denial_values_lost_revenue_once(self):
        markets = [{"city": "A", "market_size": 100_000, "max_price": 25_000, "base_average_price": 10_000}]
        target = {
            "company_id": 1, "available": 40_000, "ma_index": 0, "qi_index": 0,
            "attack_weight": 5,
            "cities": {"A": {"agents": 1, "marketing": 0, "price": 8_000}},
        }
        attacker = {
            "company_id": 2, "available": 40_000, "ma_index": 0, "qi_index": 0,
            "cities": {"A": {"agents": 1, "marketing": 0, "price": 6_000}},
        }
        before = forecast_market_sales(players=[target], markets=markets, ma_large_threshold=1300, price_power=8)
        after = forecast_market_sales(players=[target, attacker], markets=markets, ma_large_threshold=1300, price_power=8)
        pressure = self.bots._super_target_pressure(before, after, [target])
        lost_units = before["companies"][1]["sold_total"] - after["companies"][1]["sold_total"]
        self.assertGreater(lost_units, 39_000)
        self.assertEqual(pressure["targeted_damage"], lost_units * 8_000)
        self.assertEqual(pressure["target_surplus"], lost_units)
        self.assertEqual(pressure["target_tier"], 5)
        self.assertEqual(pressure["target_company_id"], 1)

    def test_sales_relocated_to_another_city_are_not_denial(self):
        markets = [
            {"city": city, "market_size": 100_000, "max_price": 25_000, "base_average_price": 10_000}
            for city in ("A", "B")
        ]
        target = {
            "company_id": 1, "available": 30_000, "ma_index": 0, "qi_index": 0,
            "attack_weight": 5,
            "cities": {city: {"agents": 1, "marketing": 0, "price": 8_000} for city in ("A", "B")},
        }
        attacker = {
            "company_id": 2, "available": 60_000, "ma_index": 0, "qi_index": 0,
            "cities": {"A": {"agents": 1, "marketing": 0, "price": 6_000}},
        }
        before = forecast_market_sales(players=[target], markets=markets, ma_large_threshold=1300, price_power=8)
        after = forecast_market_sales(players=[target, attacker], markets=markets, ma_large_threshold=1300, price_power=8)
        first, last = before["companies"][1], after["companies"][1]
        self.assertLess(last["sold_units"]["A"], first["sold_units"]["A"])
        self.assertGreater(last["sold_units"]["B"], first["sold_units"]["B"])
        self.assertEqual(last["sold_total"], first["sold_total"])
        pressure = self.bots._super_target_pressure(before, after, [target])
        self.assertEqual(pressure["targeted_damage"], 0)
        self.assertEqual(pressure["target_surplus"], 0)

    def test_target_tier_precedes_larger_damage_to_lower_priority_company(self):
        markets = [{"city": "A", "market_size": 100_000, "max_price": 25_000, "base_average_price": 10_000}]
        targets = [
            {
                "company_id": company_id, "available": stock, "ma_index": 0, "qi_index": 0,
                "attack_weight": tier,
                "cities": {"A": {"agents": 1, "marketing": 0, "price": price}},
            }
            for company_id, stock, price, tier in ((1, 20_000, 8_000, 4), (2, 10_000, 9_000, 5))
        ]
        attacker = {
            "company_id": 3, "available": 40_000, "ma_index": 0, "qi_index": 0,
            "cities": {"A": {"agents": 1, "marketing": 0, "price": 6_000}},
        }
        before = forecast_market_sales(players=targets, markets=markets, ma_large_threshold=1300, price_power=8)
        after = forecast_market_sales(players=[*targets, attacker], markets=markets, ma_large_threshold=1300, price_power=8)
        first = self.bots._super_target_pressure(before, after, [targets[0]])
        second = self.bots._super_target_pressure(before, after, [targets[1]])
        self.assertGreater(first["targeted_damage"], second["targeted_damage"])
        self.assertGreater(second["targeted_damage"], 0)
        chosen = self.bots._super_target_pressure(before, after, targets)
        self.assertEqual(chosen["target_company_id"], 2)
        self.assertEqual(chosen["target_tier"], 5)


if __name__ == "__main__":
    unittest.main()
