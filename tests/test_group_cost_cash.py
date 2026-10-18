from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class GroupCostCashTests(unittest.TestCase):
    def setUp(self):
        from sim import db
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / "groups.db"
        self.env = patch.dict(os.environ, {"SIM_DB_PATH": str(path)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.db_patch = patch.object(db, "DB_PATH", path)
        self.db_patch.start()
        self.addCleanup(self.db_patch.stop)
        db.init_db()
        self.db = db
        self.ctx = db.connect()
        self.conn = self.ctx.__enter__()
        self.addCleanup(self.ctx.__exit__, None, None, None)
        self.id = int(db.one(self.conn, "SELECT id FROM companies ORDER BY id LIMIT 1")["id"])
        self.conn.execute("DELETE FROM companies WHERE id<>?", (self.id,))
        self.conn.execute("DELETE FROM market_config WHERE city<>'广州'")
        self.conn.execute("UPDATE market_config SET worker_initial_salary=3000,engineer_initial_salary=6000,component_material=100,product_material=200,component_storage=10,product_storage=20,interest_rate=0 WHERE city='广州'")
        self.conn.execute("UPDATE companies SET home_city='广州',setup_submitted_at=? WHERE id=?", (db.now_iso(), self.id))

    def settle(self, round_no, workers, engineers, products, ma, qi, mi, *, worker_salary=3000, engineer_salary=6000, loan=0, research=0):
        from sim.engine import settle_round
        self.conn.execute("INSERT OR REPLACE INTO rounds(round_no,status) VALUES(?,'open')", (round_no,))
        self.conn.execute(
            "INSERT INTO decisions(company_id,round_no,worker_delta,worker_salary,engineer_delta,engineer_salary,production_volume,management_investment,quality_investment,loan_change,research_investment,submitted_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.id, round_no, workers, worker_salary, engineers, engineer_salary, products, ma, qi, loan, research, self.db.now_iso()),
        )
        self.conn.execute("INSERT INTO city_decisions(company_id,round_no,city,agent_delta,marketing_investment,price) VALUES(?,?,'广州',1,?,25000)", (self.id, round_no, mi))
        settle_round(self.conn, round_no)
        result = self.db.one(self.conn, "SELECT * FROM results WHERE company_id=? AND round_no=?", (self.id, round_no))
        report = json.loads(result["report_json"])
        self.assertAlmostEqual(result["cash"], result["total_assets"])
        self.assertAlmostEqual(result["net_assets"], result["cash"] - result["debt"])
        self.assertAlmostEqual(report["finance"]["round_ends"], result["cash"])
        return result, report

    def test_user_group_formula_exactly_matches_paid_cost_across_kds(self):
        from sim.bots import _balanced_production_group
        for rule in ((6, 4, 3, 24, 5), (3, 7, 4, 14, 7), (2, 2, 3, 2, 4)):
            with self.subTest(kds=rule):
                # Independent rounds/companies reset all mutable resources.
                for table in ("city_results", "market_round_stats", "results", "city_decisions", "decisions", "employee_cohorts", "agents"):
                    self.conn.execute(f"DELETE FROM {table}")
                self.conn.execute("UPDATE companies SET cash=50000000,debt=0,patents=0,component_inventory=0,product_inventory=0,component_storage_capacity=0,product_storage_capacity=0 WHERE id=?", (self.id,))
                for key, value in zip(("component_workers", "component_hours", "product_engineers", "product_hours", "components_per_product"), rule):
                    self.db.set_setting(self.conn, key, value)
                self.db.set_setting(self.conn, "worker_training_cost", 5)
                self.db.set_setting(self.conn, "engineer_training_cost", 7)
                group = _balanced_production_group(*rule)
                # The independent formula below is the user's: fixed MI and
                # Agent first, then whole production groups including MA/QI.
                group_cost = group["workers"] * (3000 * 3 + 5) + group["engineers"] * (6000 * 3 + 7)
                group_cost += group["components"] * (100 + 10) + group["products"] * (200 + 20)
                group_cost += (group["workers"] + group["engineers"]) * 1400 + group["products"] * 600
                mi, agent = 77777, 300000
                groups = math.floor((50000000 - mi - agent) / group_cost)
                workers, engineers = int(groups * group["workers"]), int(groups * group["engineers"])
                products = int(groups * group["products"])
                result, report = self.settle(1, workers, engineers, products, (workers + engineers) * 1400, products * 600, mi)
                f = report["finance"]
                self.assertEqual(result["produced"], products)
                self.assertAlmostEqual(f["cash_before_sales"], 50000000 - agent - mi - groups * group_cost, places=5)
                self.assertGreaterEqual(f["cash_before_sales"], -1e-6)
                self.assertLess(f["cash_before_sales"], group_cost)
                self.assertEqual(f["component_material"], groups * group["components"] * 100)
                self.assertEqual(f["product_material"], products * 200)
                self.assertAlmostEqual(report["production"]["ma_index"], 1400)
                self.assertAlmostEqual(report["production"]["qi_index"], 600)

    def test_three_rounds_reuse_storage_and_apply_patent_only_to_materials(self):
        for key, value in zip(("component_workers", "component_hours", "product_engineers", "product_hours", "components_per_product"), (6, 4, 3, 24, 5)):
            self.db.set_setting(self.conn, key, value)
        for round_no in range(1, 4):
            self.conn.execute("UPDATE companies SET cash=1000000,patents=?,component_inventory=0,product_inventory=0 WHERE id=?", (round_no - 1, self.id))
            salary_w, salary_e = 3000 + (round_no - 1) * 100, 6000 + (round_no - 1) * 100
            _, report = self.settle(round_no, 5 if round_no == 1 else 0, 3 if round_no == 1 else 0, 21, 11200, 12600, 0, worker_salary=salary_w, engineer_salary=salary_e)
            f = report["finance"]
            self.assertEqual(report["production"]["produced"], 21)
            self.assertAlmostEqual(f["worker_wages"], 5 * salary_w * 3)
            self.assertAlmostEqual(f["engineer_wages"], 3 * salary_e * 3)
            self.assertAlmostEqual(f["component_material"], 105 * 100 * 0.7 ** (round_no - 1))
            self.assertAlmostEqual(f["product_material"], 21 * 200 * 0.7 ** (round_no - 1))
            self.assertEqual(f["storage"], 1470 if round_no == 1 else 0)
            expected = 1000000 - f["wages"] - f["materials"] - f["storage"] - f["agents"] - f["management"] - f["quality"]
            self.assertAlmostEqual(f["cash_before_sales"], expected)

    def test_excess_requested_investment_is_paid_mi_then_qi_then_ma(self):
        self.conn.execute("UPDATE companies SET cash=1000,product_inventory=100,component_inventory=50 WHERE id=?", (self.id,))
        self.conn.execute("INSERT INTO agents(company_id,city,count) VALUES(?,'广州',1)", (self.id,))
        self.db.set_setting(self.conn, "agent_add_cost", 0)
        _, report = self.settle(1, 0, 0, 0, 700, 600, 800)
        f = report["finance"]
        # Old stock from an import must not be charged for storage again when
        # no new material or products are being made.
        self.assertEqual(f["materials"], 0)
        self.assertEqual(f["storage"], 0)
        self.assertEqual((f["marketing"], f["quality"], f["management"], f["cash_before_sales"]), (800, 200, 0, 0))
        self.assertEqual((f["marketing_requested"], f["quality_requested"], f["management_requested"]), (800, 600, 700))

    def test_one_hundred_varied_settlements_reconcile_cash_and_debt(self):
        from sim.finance import asset_reconciliation, build_finance_ledger
        rng = random.Random(20261018)
        for trial in range(100):
            with self.subTest(trial=trial):
                for table in ("city_results", "market_round_stats", "results", "city_decisions", "decisions", "employee_cohorts", "agents"):
                    self.conn.execute(f"DELETE FROM {table}")
                workers, engineers = rng.randrange(500), rng.randrange(150)
                for role, count in (("worker", workers), ("engineer", engineers)):
                    self.conn.execute("INSERT INTO employee_cohorts(company_id,role,count,hire_round) VALUES(?,?,?,0)", (self.id, role, count))
                cash = rng.randrange(20000000)
                self.conn.execute(
                    "UPDATE companies SET cash=?,debt=?,patents=?,component_inventory=?,product_inventory=?,component_storage_capacity=?,product_storage_capacity=? WHERE id=?",
                    (cash, rng.randrange(10000000), rng.randrange(4), rng.randrange(2000), rng.randrange(2000), rng.randrange(5000), rng.randrange(5000), self.id),
                )
                self.conn.execute("UPDATE market_config SET interest_rate=0.03")
                result, report = self.settle(
                    1, rng.randrange(-workers - 100, 300), rng.randrange(-engineers - 100, 100),
                    rng.randrange(10000), rng.randrange(10000000), rng.randrange(10000000), rng.randrange(10000000),
                    worker_salary=rng.randrange(2000, 4001), engineer_salary=rng.randrange(5000, 7001),
                    loan=rng.randrange(-5000000, 8000000), research=rng.randrange(8000000),
                )
                ledger = build_finance_ledger(report)
                self.assertTrue(all(row["cash"] >= -1e-6 for row in ledger))
                self.assertTrue(all(row["debt"] >= -1e-6 for row in ledger))
                recon = asset_reconciliation(report)
                for key in ("cash_difference", "debt_difference", "asset_difference", "net_asset_difference"):
                    self.assertAlmostEqual(recon[key], 0, places=6)
                self.assertEqual(result["total_assets"], result["cash"])
