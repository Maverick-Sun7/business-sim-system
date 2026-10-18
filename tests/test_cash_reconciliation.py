from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sim.finance import asset_reconciliation, build_finance_ledger


class CashLedgerTests(unittest.TestCase):
    def test_bonus_can_be_spent_before_sales_in_old_and_new_reports(self):
        report = {
            "finance": {"round_begins": 100, "project_bonus": 90,
                        "bonus_in_round_begins": True, "worker_wages": 95,
                        "round_ends": 5},
            "key_metrics": {"total_assets": 5, "debt": 0, "net_assets": 5},
        }
        old = copy.deepcopy(report)
        old["finance"]["round_begins"] = 10
        old["finance"].pop("bonus_in_round_begins")
        for value in (old, report):
            rows = build_finance_ledger(value)
            self.assertEqual(rows[0]["cash"], 100)
            self.assertEqual(rows[-1]["cash"], 5)
            self.assertTrue(all(row["cash"] >= 0 for row in rows))
            bonus = next(row for row in rows if row["key"] == "project_bonus")
            self.assertEqual(bonus["cash_flow"], 0)
            self.assertEqual(asset_reconciliation(value)["asset_difference"], 0)

    def test_debt_only_compensation_and_interest_do_not_deduct_cash(self):
        report = {
            "finance": {"round_begins": 100, "loan_change": -20,
                        "layoff": 50, "layoff_cash": 10, "layoff_debt": 40,
                        "quit_penalty": 90, "quit_penalty_cash": 0,
                        "quit_penalty_debt": 90, "interest": 13,
                        "round_ends": 70},
            # Deliberately omit starting_debt to exercise old-report fallback.
            "key_metrics": {"total_assets": 70, "debt": 323, "net_assets": -253},
        }
        rows = build_finance_ledger(report)
        self.assertEqual(rows[0]["debt"], 200)
        self.assertEqual(rows[-1]["cash"], 70)
        self.assertEqual(rows[-1]["debt"], 323)
        self.assertTrue(all(abs(v) < 1e-9 for k, v in asset_reconciliation(report).items() if k.endswith("difference")))

    def test_ledger_exposes_mismatch_instead_of_overwriting_the_last_balance(self):
        report = {"finance": {"round_begins": 100, "marketing": 30, "round_ends": 99},
                  "key_metrics": {"total_assets": 120, "debt": 0, "net_assets": 120,
                                  "inventory_book_value": 50}}
        self.assertEqual(build_finance_ledger(report)[-1]["cash"], 70)
        reconciliation = asset_reconciliation(report)
        self.assertEqual(reconciliation["cash_difference"], -29)
        self.assertEqual(reconciliation["asset_difference"], 50)


class CashAccountingMigrationTests(unittest.TestCase):
    def test_existing_results_repair_assets_without_altering_actual_cash_or_stock(self):
        from sim import db
        from sim.engine import current_company_net_assets

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"SIM_DB_PATH": str(Path(directory) / "legacy.db")}), patch.object(db, "DB_PATH", Path(directory) / "legacy.db"):
            db.init_db()
            with db.connect() as conn:
                company_id = int(db.one(conn, "SELECT id FROM companies ORDER BY id LIMIT 1")["id"])
                conn.execute("UPDATE companies SET cash=750,debt=200,product_inventory=7,component_inventory=11 WHERE id=?", (company_id,))
                report = {"key_metrics": {"total_assets": 1550, "net_assets": 1350, "inventory_book_value": 800},
                          "finance": {"round_begins": 1000, "round_ends": 750}}
                conn.execute(
                    "INSERT INTO results(company_id,round_no,total_assets,debt,net_assets,cash,sales_revenue,total_cost,net_profit,produced,sold,inventory,ma_index,qi_index,research_success,report_json) VALUES(?,1,1550,200,1350,750,0,250,-250,0,0,7,0,0,0,?)",
                    (company_id, json.dumps(report)),
                )
                conn.execute("DELETE FROM settings WHERE key='asset_accounting_basis'")
            db.init_db()
            db.init_db()
            with db.connect() as conn:
                company = db.one(conn, "SELECT * FROM companies WHERE id=?", (company_id,))
                result = db.one(conn, "SELECT * FROM results WHERE company_id=?", (company_id,))
                self.assertEqual((company["cash"], company["debt"], company["product_inventory"], company["component_inventory"]), (750, 200, 7, 11))
                self.assertEqual((result["cash"], result["total_assets"], result["net_assets"]), (750, 750, 550))
                metrics = json.loads(result["report_json"])["key_metrics"]
                self.assertEqual((metrics["total_assets"], metrics["net_assets"], metrics["inventory_book_value"]), (750, 550, 0))
                self.assertEqual(current_company_net_assets(conn, company), 550)
                # A round-opening bonus is money now; previous result and stock
                # must not replace the authoritative usable cash balance.
                changed = dict(company)
                changed["cash"] += 100
                self.assertEqual(current_company_net_assets(conn, changed), 650)
