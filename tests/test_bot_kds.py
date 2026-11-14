from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class BotKDSRegressionTest(unittest.TestCase):
    """Exercise the real planner against small, isolated custom KDS games."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "bot-kds.db"
        # sim.db initializes on import, so isolate its environment first too.
        environment = patch.dict(os.environ, {"SIM_DB_PATH": str(path)})
        environment.start()
        self.addCleanup(environment.stop)
        from sim import bots, db

        database_path = patch.object(db, "DB_PATH", path)
        database_path.start()
        self.addCleanup(database_path.stop)
        db.init_db()
        self.db, self.bots = db, bots
        self.connection_context = db.connect()
        self.conn = self.connection_context.__enter__()
        self.addCleanup(self.connection_context.__exit__, None, None, None)
        self.conn.execute("DELETE FROM companies")
        self.conn.execute("DELETE FROM market_config WHERE city NOT IN ('广州','深圳')")

    def company(self, code, *, super_mode=False, ordinary=False, cash=15_000_000):
        cursor = self.conn.execute(
            "INSERT INTO companies(code,name,password_hash,home_city,cash,is_bot,"
            "is_super_bot,bot_profile,setup_submitted_at,created_at) "
            "VALUES(?,?,?,'广州',?,?,?,0,?,?)",
            (code, code, "unused-test-password", cash, int(super_mode or ordinary),
             int(super_mode), self.db.now_iso(), self.db.now_iso()),
        )
        return int(cursor.lastrowid)

    def test_super_selector_maximizes_relative_advantage(self):
        def candidate(name, *, profit, risk, damage=0, ma=1, qi=1, mi=0, price=.9, sold=100, coverage=.8):
            return {
                "name": name,
                "predicted_profit": profit,
                "risk_profit": risk,
                "ma": ma,
                "qi": qi,
                "marketing_total": mi,
                "price_ratio": price,
                "predicted_sold": sold,
                "sell_ratio": min(1.0, sold / 200),
                "coverage": coverage,
                "rival_damage": damage,
                "starting_assets": 10_000,
                "ending_assets": 10_000 + profit,
                "score": (1, profit, risk),
            }

        choices = [
            candidate("high", profit=1000, risk=900, ma=10, price=.97, coverage=.7),
            candidate("ma", profit=980, risk=875, ma=100, price=.9, coverage=.8),
            candidate("qi", profit=980, risk=875, qi=100, price=.9, coverage=.8),
            candidate("mi", profit=980, risk=875, mi=100, price=.9, coverage=.8),
            candidate("balanced", profit=975, risk=870, ma=50, qi=50, mi=50, coverage=.82),
            candidate("fit", profit=970, risk=865, ma=20, qi=20, mi=20, coverage=1.0),
            candidate("low", profit=970, risk=865, price=.60, sold=300, coverage=1.2),
            candidate("too-costly", profit=900, risk=850, ma=1000, qi=1000, mi=1000, sold=1000),
            candidate("unsafe", profit=1100, risk=-1, price=.99, sold=1000),
        ]
        for profile in range(7):
            with self.subTest(profile=profile):
                selected = self.bots._select_empirical_super_candidate(
                    choices,
                    profile,
                    tactical_price_allowed=True,
                )
                self.assertEqual(selected["name"], "unsafe")


        early_price_profile = self.bots._select_empirical_super_candidate(
            choices,
            6,
            tactical_price_allowed=False,
        )
        self.assertEqual(early_price_profile["name"], "unsafe")

        # A style may diversify only after clearing the growth benchmark. If
        # the target is unreachable, it makes the highest-profit safe attempt.
        targeted = self.bots._select_empirical_super_candidate(
            choices,
            1,
            tactical_price_allowed=True,
            profit_target=985,
        )
        unreachable = self.bots._select_empirical_super_candidate(
            choices,
            6,
            tactical_price_allowed=True,
            profit_target=2_000,
        )
        self.assertEqual(targeted["name"], "unsafe")
        self.assertEqual(unreachable["name"], "unsafe")

        destructive = candidate(
            "destructive", profit=-1_900, risk=-2_100,
            damage=3_600, price=.45, sold=500,
        )
        strategic = self.bots._select_empirical_super_candidate(
            [*choices, destructive],
            0,
            tactical_price_allowed=True,
        )
        # Protected/nonattacking bots never sacrifice profit for a cheap
        # pressure proxy. Only designated attackers use verified target loss.
        self.assertEqual(strategic["name"], "unsafe")

        targeted_attack = candidate(
            "targeted-attack", profit=-400, risk=-600,
            damage=0, price=.52, sold=450,
        )
        targeted_attack.update({
            "targeted_damage": 2_000,
            "target_tier": 5,
            "target_cpi_drop": 12,
            "target_surplus": 300,
        })
        selected_attack = self.bots._select_empirical_super_candidate(
            [*choices, targeted_attack],
            0,
            tactical_price_allowed=True,
            competitive_mode=True,
            sacrifice_allowed=True,
        )
        self.assertEqual(selected_attack["name"], "targeted-attack")
        stable_attack = self.bots._select_empirical_super_candidate(
            [*choices, targeted_attack],
            0,
            tactical_price_allowed=True,
            competitive_mode=True,
            sacrifice_allowed=False,
        )
        self.assertEqual(stable_attack["name"], "unsafe")

    def test_early_secondary_sales_fund_research_without_forcing_a_price_cut(self):
        from sim.engine import settle_round
        self.conn.execute("DELETE FROM market_config WHERE city<>'广州'")
        self.db.set_setting(self.conn, "total_rounds", 7)
        human = self.company("HUMAN")
        super_id = self.company("EARLY-SUPER", super_mode=True)
        for company_id in (human, super_id):
            self.conn.execute("INSERT INTO agents(company_id,city,count) VALUES(?,'广州',1)", (company_id,))
        self.conn.execute("UPDATE companies SET product_inventory=100,product_storage_capacity=100 WHERE id=?", (human,))
        self.conn.execute("UPDATE rounds SET status='open' WHERE round_no=1")
        self.conn.execute(
            "INSERT INTO decisions(company_id,round_no,worker_salary,engineer_salary,management_investment,submitted_at) VALUES(?,1,3300,6400,260000,?)",
            (human, self.db.now_iso()),
        )
        self.conn.execute("INSERT INTO city_decisions(company_id,round_no,city,price) VALUES(?,1,'广州',25000)", (human,))
        self.bots.submit_super_bot_decisions(self.conn, 1, defer_rebalance=True)
        decision = self.db.one(self.conn, "SELECT * FROM decisions WHERE company_id=? AND round_no=1", (super_id,))
        self.assertGreater(decision["research_investment"], 0)
        self.bots.finalize_super_bot_decisions(self.conn, 1)
        settle_round(self.conn, 1)
        result = self.db.one(self.conn, "SELECT * FROM results WHERE company_id=? AND round_no=1", (super_id,))
        import json
        report = json.loads(result["report_json"])
        self.assertGreater(result["net_profit"], 0)
        self.assertEqual(result["sold"], result["produced"])
        self.assertEqual(report["research"]["investment"], decision["research_investment"])
        self.assertGreater(sum(item["secondary_units"] for item in report["sales"]), result["sold"] * 0.50)
        self.assertGreaterEqual(report["sales"][0]["price"], 24750)

    def test_patent_budget_uses_remaining_rounds_accumulated_research_and_cash(self):
        plan = dict(balance=3_000_000, remaining_rounds=2, material_spend=80_000_000,
                    sold=9000, available=10000, projected_cash=35_000_000,
                    reserve=15_000_000, risk_profit=12_000_000, amount_25=1_500_000,
                    amount_75=6_000_000, hidden_cap=.43, hidden_multiplier=4/3,
                    buffer=150_000, patent_factor=.7, tax_rate=.2)
        invest = self.bots._profitable_patent_budget
        self.assertEqual(invest(**plan), 5_150_000)
        self.assertEqual(invest(**{**plan, "remaining_rounds": 0}), 0)
        self.assertEqual(invest(**{**plan, "material_spend": 500_000}), 0)
        self.assertEqual(invest(**{**plan, "projected_cash": 16_000_000}), 0)

    def test_super_expansion_never_goes_below_the_unexpanded_profit(self):
        normal = dict(predicted_profit=100., predicted_sold=100., sell_ratio=1.)
        wider = dict(predicted_profit=100., predicted_sold=130., sell_ratio=.97, expansion_probe=True)
        loss = dict(predicted_profit=99., predicted_sold=180., sell_ratio=1., expansion_probe=True)
        chosen = self.bots._select_empirical_super_candidate(
            [normal, wider, loss], 0, tactical_price_allowed=True, expand_at_equal_profit=True,
        )
        self.assertIs(chosen, wider)

    def test_super_drops_a_human_counter_that_improves_the_rival_even_more(self):
        ordinary = dict(predicted_profit=100., predicted_sold=100., projected_rival_profit=200.)
        copied = dict(predicted_profit=120., predicted_sold=150., projected_rival_profit=240., rival_seed=True)
        counter = dict(predicted_profit=121., predicted_sold=155., projected_rival_profit=180., rival_seed=True)
        for plans, expected in (([ordinary, copied], ordinary), ([ordinary, copied, counter], counter)):
            chosen = self.bots._select_empirical_super_candidate(
                plans, 0, tactical_price_allowed=True, rival_benchmark=200.,
            )
            self.assertIs(chosen, expected)

    def test_first_place_bot_prioritizes_the_final_gap_without_taking_a_stress_loss(self):
        high_profit = dict(predicted_profit=180., predicted_sold=150., risk_profit=130., first_place_gap=-120.)
        closer = dict(predicted_profit=100., predicted_sold=125., risk_profit=70., first_place_gap=-15.)
        speculative_win = dict(predicted_profit=200., predicted_sold=155., risk_profit=-1., first_place_gap=80.)
        chosen = self.bots._select_empirical_super_candidate(
            [high_profit, closer, speculative_win], 0, tactical_price_allowed=True, race_for_first=True,
        )
        self.assertIs(chosen, closer)

    def test_first_place_recheck_keeps_the_round_open_on_a_worker_snapshot(self):
        import json
        import sqlite3
        from contextlib import closing
        from sim.remote_worker import _snapshot, load_snapshot

        human = self.company("HUMAN", cash=50_000_000)
        contender = self.company("CONTENDER", cash=20_000_000, super_mode=True)
        trailing = self.company("TRAILING", cash=10_000_000, super_mode=True)
        self.conn.execute("INSERT INTO rounds(round_no,status) VALUES(4,'open')")
        for company_id in (human, contender, trailing):
            self.conn.execute(
                "INSERT INTO decisions(company_id,round_no,worker_salary,engineer_salary,submitted_at,is_draft) "
                "VALUES(?,4,3300,6400,?,?)", (company_id, self.db.now_iso(), int(company_id != human)),
            )
        raw, _ = _snapshot(self.conn)
        with closing(sqlite3.connect(":memory:")) as worker:
            worker.row_factory = sqlite3.Row
            load_snapshot(worker, json.loads(raw))
            with patch.object(self.bots, "_submit_bots") as submit:
                self.bots._recheck_first_place_draft(worker, 4)
            self.assertEqual(submit.call_args.kwargs["target_ids"], {contender})
            self.assertIsNone(self.db.one(worker, "SELECT 1 FROM results WHERE round_no=4"))
            self.assertEqual(self.db.one(worker, "SELECT status FROM rounds WHERE round_no=4")["status"], "open")
            self.assertEqual(self.db.one(worker, "SELECT SUM(is_draft) AS n FROM decisions WHERE round_no=4")["n"], 2)

    def test_top_three_copy_current_human_prices_even_with_no_prior_reports(self):
        humans = [self.company("HUMAN-A"), self.company("HUMAN-B")]
        supers = [self.company(f"SUPER-{i}", super_mode=True, cash=30_000_000) for i in range(4)]
        self.conn.execute("DELETE FROM market_config WHERE city<>'广州'")
        self.conn.execute("UPDATE rounds SET status='open' WHERE round_no=1")
        for company_id in humans + supers:
            self.conn.execute("INSERT INTO agents(company_id,city,count) VALUES(?,'广州',1)", (company_id,))
        for i, company_id in enumerate(humans):
            self.conn.execute("UPDATE companies SET product_inventory=100 WHERE id=?", (company_id,))
            self.conn.execute("INSERT INTO decisions(company_id,round_no,worker_salary,engineer_salary,"
                              "management_investment,submitted_at) VALUES(?,1,3300,6400,?,?)",
                              (company_id, (i+1)*1500, self.db.now_iso()))
            self.conn.execute("INSERT INTO city_decisions(company_id,round_no,city,price) VALUES(?,1,'广州',?)",
                              (company_id, 10000 if i==0 else 24671))
        seen = {}
        original = self.bots._forecast_candidates
        def capture(payloads):
            for item in payloads:
                player = item["players"][-1]
                own_prices = seen.setdefault(player["company_id"], set())
                own_prices.update(route["price"] for route in player["cities"].values())
            return original(payloads)
        with patch.object(self.bots, "_forecast_candidates", side_effect=capture):
            self.assertEqual(self.bots.submit_super_bot_decisions(self.conn, 1, defer_rebalance=True), 4)
        for company_id in supers[:3]:
            self.assertIn(24671, seen[company_id])
        self.assertNotIn(24671, seen[supers[3]])

    def test_bot_only_analysis_uses_earlier_saved_peer_as_a_live_benchmark(self):
        first = self.company("FIRST", super_mode=True, cash=30_000_000)
        second = self.company("SECOND", super_mode=True, cash=30_000_000)
        later = self.company("LATER", super_mode=True, cash=30_000_000)
        self.conn.execute("DELETE FROM market_config WHERE city<>'广州'")
        self.conn.execute("UPDATE rounds SET status='open' WHERE round_no=1")
        self.conn.executemany("INSERT INTO agents(company_id,city,count) VALUES(?,'广州',1)",
                              [(first,), (second,), (later,)])
        self.conn.execute("UPDATE companies SET product_inventory=100 WHERE id IN (?,?)", (first, later))
        for company_id, price in ((first, 24671), (later, 23417)):
            self.conn.execute(
                "INSERT INTO decisions(company_id,round_no,worker_salary,engineer_salary,management_investment,"
                "submitted_at,is_draft) VALUES(?,1,3300,6400,1500,?,1)",
                (company_id, self.db.now_iso()),
            )
            self.conn.execute("INSERT INTO city_decisions(company_id,round_no,city,price) VALUES(?,1,'广州',?)",
                              (company_id, price))
        original = self.bots._forecast_candidates
        seen = set()

        def capture(payloads):
            for item in payloads:
                seen.update(route["price"] for route in item["players"][-1]["cities"].values())
            return original(payloads)

        with patch.object(self.bots, "_forecast_candidates", side_effect=capture):
            self.assertEqual(self.bots.submit_super_bot_decisions(
                self.conn, 1, replace_existing=True, target_ids={second}, defer_rebalance=True,
            ), 1)
        self.assertIn(24671, seen)
        self.assertNotIn(23417, seen)
        self.assertEqual(self.db.one(self.conn, "SELECT price FROM city_decisions WHERE company_id=?",
                                    (first,))["price"], 24671)

    def test_bot_only_rebalance_checks_the_challenger_against_the_bot_leader(self):
        leader = self.company("LEADER", cash=50_000_000, super_mode=True)
        challenger = self.company("CHALLENGER", cash=20_000_000, super_mode=True)
        self.db.set_setting(self.conn, "total_rounds", 3)
        self.conn.execute("INSERT INTO rounds(round_no,status) VALUES(3,'open')")
        for company_id in (leader, challenger):
            self.conn.execute(
                "INSERT INTO decisions(company_id,round_no,worker_salary,engineer_salary,submitted_at,is_draft) "
                "VALUES(?,3,3300,6400,?,1)", (company_id, self.db.now_iso()),
            )
        with patch.object(self.bots, "_submit_bots") as submit:
            self.assertEqual(self.bots.rebalance_super_bot_decisions(self.conn, 3), 2)
        self.assertEqual(submit.call_args.kwargs["target_ids"], {challenger})
        self.assertIsNone(self.db.one(self.conn, "SELECT 1 FROM results WHERE round_no=3"))

    def test_competitive_super_bot_count_is_ranked_and_bounded(self):
        ranked = list(range(1, 10))
        # Penultimate/final rounds use one third, always taken from the lowest
        # eligible ranks while the top three remain protected.
        self.assertEqual(
            self.bots._competitive_super_attackers(
                ranked, official_round=6, total_rounds=7,
                markets_saturated=False,
            ),
            {7, 8, 9},
        )
        self.assertEqual(
            self.bots._competitive_super_attackers(
                ranked, official_round=3, total_rounds=7,
                markets_saturated=True,
            ),
            {8, 9},
        )
        # With fewer than three Super Bots only first place is protected.
        self.assertEqual(
            self.bots._competitive_super_attackers(
                [11, 12], official_round=4, total_rounds=5,
                markets_saturated=False,
            ),
            {12},
        )
        self.assertEqual(
            self.bots._competitive_super_attackers(
                [21, 22, 23], official_round=4, total_rounds=5,
                markets_saturated=True,
            ),
            set(),
        )

    def test_super_profit_target_uses_self_higher_rank_and_previous_bot(self):
        leader_id = self.company("HUMAN")
        previous_bot_id = self.company("SUPER-A", super_mode=True)
        target_bot_id = self.company("SUPER-B", super_mode=True)

        def result(company_id, assets, profit):
            self.conn.execute(
                "INSERT INTO results(company_id,round_no,total_assets,debt,net_assets,cash,"
                "sales_revenue,total_cost,net_profit,produced,sold,inventory,ma_index,qi_index,"
                "research_success,report_json) VALUES(?,1,?,0,?,?,?,0,?,0,0,0,0,0,0,'{}')",
                (company_id, assets, assets, assets, profit, profit),
            )

        result(leader_id, 300, 150)
        result(previous_bot_id, 100, 175)
        result(target_bot_id, 200, 100)
        target = self.bots._super_profit_target(
            self.conn,
            target_bot_id,
            2,
            previous_bot_id,
        )
        self.assertAlmostEqual(target, 176.75)

    def submitted_seller(self, code, price):
        company_id = self.company(code)
        self.conn.execute(
            "INSERT INTO decisions(company_id,round_no,production_volume,submitted_at) "
            "VALUES(?,1,100000,?)", (company_id, self.db.now_iso()),
        )
        self.conn.execute(
            "INSERT INTO agents(company_id,city,count) VALUES(?,'广州',1)",
            (company_id,),
        )
        self.conn.execute(
            "INSERT INTO city_decisions(company_id,round_no,city,price) "
            "VALUES(?,1,'广州',?)", (company_id, price),
        )
        return company_id

    def test_final_capacity_forecast_uses_current_kds_price_power(self):
        self.conn.execute("DELETE FROM market_config WHERE city<>'广州'")
        self.conn.execute(
            "UPDATE market_config SET population=1000000,penetration=0.1,"
            "initial_avg_price=10000"
        )
        lower_price = self.submitted_seller("LOW", 6000)
        higher_price = self.submitted_seller("HIGH", 8000)
        markets = [dict(row) for row in self.db.all_rows(self.conn, "SELECT * FROM market_config")]
        # No MA/QI/MI: the 40% price pool is an independent arithmetic oracle.
        for power in (2, 11):
            with self.subTest(power=power):
                self.db.set_setting(self.conn, "cpi_price_power", power)
                capacities = self.bots._forecast_submitted_cpi_capacity(self.conn, 1, markets)
                expected_low = 40000 * 4000 ** power / (4000 ** power + 2000 ** power)
                self.assertAlmostEqual(capacities[lower_price], expected_low, places=7)
                self.assertAlmostEqual(capacities[higher_price], 40000 - expected_low, places=7)

    def test_super_bot_candidate_search_and_decision_forecast_use_current_power(self):
        self.conn.execute("DELETE FROM market_config WHERE city<>'广州'")
        self.conn.execute("UPDATE market_config SET initial_avg_price=20000")
        self.submitted_seller("RIVAL", 8000)
        super_id = self.company("SUPER", super_mode=True)
        self.conn.execute(
            "INSERT INTO agents(company_id,city,count) VALUES(?,'广州',1)", (super_id,),
        )
        # Change KDS in the same process: neither evaluator may cache default 8
        # or the previous round's exponent. Wrapping retains real calculations.
        for power in (2, 11):
            with self.subTest(power=power):
                self.db.set_setting(self.conn, "cpi_price_power", power)
                with (
                    patch.object(
                        self.bots, "prepare_city_cpi_for_company",
                        wraps=self.bots.prepare_city_cpi_for_company,
                    ) as candidate_cpi,
                    patch.object(
                        self.bots, "allocate_city_cpi_for_company",
                        wraps=self.bots.allocate_city_cpi_for_company,
                    ) as decision_forecast,
                ):
                    self.assertEqual(self.bots.submit_super_bot_decisions(
                        self.conn, 1, replace_existing=True, defer_rebalance=True,
                    ), 1)
                for allocator in (candidate_cpi, decision_forecast):
                    self.assertGreater(allocator.call_count, 0)
                    self.assertEqual(
                        {call.kwargs.get("price_power") for call in allocator.call_args_list},
                        {power},
                    )
                decision = self.db.one(
                    self.conn, "SELECT * FROM decisions WHERE company_id=? AND round_no=1", (super_id,),
                )
                self.assertIsNotNone(decision["submitted_at"])
                self.assertGreater(decision["production_volume"], 0)

    def test_ordinary_and_super_bot_agent_additions_obey_each_city_kds_limit(self):
        bot_id = self.company("BOT", ordinary=True, cash=300_000_000)
        self.db.set_setting(self.conn, "total_rounds", 10)
        self.conn.execute("INSERT INTO rounds(round_no,status) VALUES(7,'open')")
        # R7 wants at least four agents in each initially empty city. This
        # catches both an old hard-coded 3 and a mistaken global per-round cap.
        for super_mode in (False, True):
            self.conn.execute(
                "UPDATE companies SET is_super_bot=? WHERE id=?", (int(super_mode), bot_id),
            )
            for limit in (0, 1, 6):
                with self.subTest(super_mode=super_mode, limit=limit):
                    self.conn.execute("DELETE FROM decisions")
                    self.conn.execute("DELETE FROM city_decisions")
                    self.db.set_setting(self.conn, "max_agent_add_per_city_round", limit)
                    if super_mode:
                        count = self.bots.submit_super_bot_decisions(self.conn, 7, defer_rebalance=True)
                    else:
                        count = self.bots.submit_bot_decisions(self.conn, 7)
                    self.assertEqual(count, 1)
                    additions = [int(row["agent_delta"]) for row in self.db.all_rows(
                        self.conn, "SELECT agent_delta FROM city_decisions WHERE company_id=? "
                        "AND round_no=7 ORDER BY city", (bot_id,),
                    )]
                    self.assertEqual(len(additions), 2)
                    self.assertTrue(all(0 <= delta <= limit for delta in additions), additions)
                    if limit <= 1:
                        self.assertEqual(additions, [limit, limit])
                    else:
                        self.assertTrue(all(delta > 3 for delta in additions), additions)

    def test_late_super_bot_opens_every_city_never_removes_agents_and_has_no_partial_mi(self):
        super_id = self.company("SUPER-LATE", super_mode=True, cash=1_000_000_000)
        self.db.set_setting(self.conn, "total_rounds", 7)
        self.conn.execute("INSERT INTO rounds(round_no,status) VALUES(5,'open')")
        for city in ("广州", "深圳"):
            self.conn.execute(
                "INSERT INTO agents(company_id,city,count) VALUES(?,?,10)",
                (super_id, city),
            )

        self.assertEqual(
            self.bots.submit_super_bot_decisions(
                self.conn, 5, replace_existing=True, defer_rebalance=True,
            ),
            1,
        )
        rows = [dict(row) for row in self.db.all_rows(
            self.conn,
            "SELECT city,agent_delta,marketing_investment FROM city_decisions "
            "WHERE company_id=? AND round_no=5 ORDER BY city",
            (super_id,),
        )]
        self.assertEqual({row["city"] for row in rows}, {"广州", "深圳"})
        self.assertTrue(all(int(row["agent_delta"]) == 0 for row in rows), rows)
        funded = [float(row["marketing_investment"] or 0) > 0 for row in rows]
        self.assertEqual(sum(funded), len(rows), rows)

    def test_global_saturation_uses_one_latest_round_not_historical_peaks(self):
        markets = [dict(row) for row in self.db.all_rows(
            self.conn, "SELECT * FROM market_config ORDER BY city",
        )]
        for round_no, utilization in ((1, 0.96), (2, 0.55)):
            for market in markets:
                size = float(market["population"]) * float(market["penetration"])
                self.conn.execute(
                    "INSERT INTO market_round_stats(city,round_no,base_average_price,"
                    "average_price,market_size,player_total_volume) VALUES(?,?,?,?,?,?)",
                    (market["city"], round_no, market["initial_avg_price"],
                     market["initial_avg_price"], size, size * utilization),
                )
        self.assertTrue(self.bots._all_markets_near_capacity(self.conn, 2, markets))
        self.assertFalse(self.bots._all_markets_near_capacity(self.conn, 3, markets))

    def test_joint_price_pass_preserves_profit_selected_investments(self):
        self.conn.execute("DELETE FROM market_config WHERE city<>'广州'")
        super_id = self.company("SUPER", super_mode=True, cash=1_000_000_000)
        self.conn.execute("INSERT INTO rounds(round_no,status) VALUES(5,'open')")
        for role, count in (("worker", 300), ("engineer", 120)):
            self.conn.execute("INSERT INTO employee_cohorts(company_id,role,count,hire_round) VALUES(?,?,?,4)", (super_id, role, count))
        self.conn.execute("UPDATE companies SET bot_profile=2 WHERE id=?", (super_id,))
        self.conn.execute(
            "INSERT INTO agents(company_id,city,count) VALUES(?,'广州',1)", (super_id,),
        )
        self.conn.execute(
            "INSERT INTO decisions(company_id,round_no,worker_salary,engineer_salary,"
            "production_volume,submitted_at) VALUES(?,5,3300,6400,1000,?)",
            (super_id, self.db.now_iso()),
        )
        self.conn.execute(
            "INSERT INTO city_decisions(company_id,round_no,city,price) "
            "VALUES(?,5,'广州',6000)", (super_id,),
        )
        markets = [dict(row) for row in self.db.all_rows(
            self.conn, "SELECT * FROM market_config",
        )]
        quality_values = []
        for multiplier in (1.20, 1.80):
            self.db.set_setting(self.conn, "qi_safe_multiplier", multiplier)
            self.conn.execute(
                "UPDATE decisions SET management_investment=654321,quality_investment=123456 "
                "WHERE company_id=? AND round_no=5", (super_id,),
            )
            self.conn.execute(
                "UPDATE city_decisions SET price=6000,marketing_investment=98765 "
                "WHERE company_id=? AND round_no=5", (super_id,),
            )
            self.assertTrue(self.bots._coordinate_super_bot_prices(self.conn, 5, markets))
            decision = self.db.one(
                self.conn, "SELECT management_investment,quality_investment FROM decisions "
                "WHERE company_id=? AND round_no=5", (super_id,),
            )
            marketing = self.db.one(
                self.conn, "SELECT marketing_investment FROM city_decisions "
                "WHERE company_id=? AND round_no=5", (super_id,),
            )
            self.assertEqual(float(decision["management_investment"]), 654321.0)
            quality_values.append(float(decision["quality_investment"]))
            self.assertEqual(float(marketing["marketing_investment"]), 98765.0)
        self.assertEqual(quality_values, [123456.0, 123456.0])

        # With production costs already above available cash, the final joint
        # pass must request no CPI investment instead of relying on settlement
        # to truncate MI/QI/MA silently.
        self.conn.execute("UPDATE companies SET cash=1000 WHERE id=?", (super_id,))
        self.conn.execute(
            "UPDATE decisions SET production_volume=100000,management_investment=0,"
            "quality_investment=0 WHERE company_id=? AND round_no=5", (super_id,),
        )
        self.conn.execute(
            "UPDATE city_decisions SET price=6000,marketing_investment=0 "
            "WHERE company_id=? AND round_no=5", (super_id,),
        )
        self.assertTrue(self.bots._coordinate_super_bot_prices(self.conn, 5, markets))
        decision = self.db.one(
            self.conn, "SELECT management_investment,quality_investment FROM decisions "
            "WHERE company_id=? AND round_no=5", (super_id,),
        )
        marketing = self.db.one(
            self.conn, "SELECT SUM(marketing_investment) AS total FROM city_decisions "
            "WHERE company_id=? AND round_no=5", (super_id,),
        )
        self.assertEqual(float(decision["management_investment"]), 0.0)
        self.assertEqual(float(decision["quality_investment"]), 0.0)
        self.assertEqual(float(marketing["total"] or 0), 0.0)
        price = self.db.one(
            self.conn, "SELECT price FROM city_decisions WHERE company_id=? "
            "AND round_no=5 AND city='广州'", (super_id,),
        )
        self.assertEqual(float(price["price"]), 6000.0)


if __name__ == "__main__":
    unittest.main()
