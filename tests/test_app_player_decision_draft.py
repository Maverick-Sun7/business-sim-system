from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class PlayerDecisionDraftAppTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="business-sim-player-draft-")
        self.addCleanup(self.temp.cleanup)
        self.environment = patch.dict(
            os.environ,
            {"SIM_DB_PATH": str(Path(self.temp.name) / "player-draft.db")},
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

        from sim import db

        self.db = db
        previous_path = db.DB_PATH
        self.addCleanup(setattr, db, "DB_PATH", previous_path)
        db.DB_PATH = Path(os.environ["SIM_DB_PATH"])
        db.init_db()
        with db.connect() as conn:
            company = db.one(conn, "SELECT id FROM companies ORDER BY id LIMIT 1")
            self.company_id = int(company["id"])
            conn.execute(
                "UPDATE companies SET home_city='广州',setup_submitted_at=? WHERE id=?",
                (db.now_iso(), self.company_id),
            )
            conn.execute(
                "UPDATE rounds SET status='open',starts_at=?,ends_at=NULL WHERE round_no=1",
                (db.now_iso(),),
            )
        self.app_path = str(Path(__file__).resolve().parents[1] / "app.py")

    def test_unsubmitted_inputs_survive_page_navigation(self) -> None:
        def navigation(app):
            return next(
                item for item in app.get("button_group")
                if item.key == "player_navigation"
            )

        def choose(app, label):
            widget = navigation(app)
            # Newer Streamlit exposes single-select segmented controls as a
            # scalar; older AppTest versions used a one-element list.
            return widget.set_value(label if isinstance(widget.value, str) else [label])

        app = AppTest.from_file(self.app_path, default_timeout=30)
        app.session_state["auth"] = {
            "role": "player",
            "company_id": self.company_id,
        }
        app.run()
        choose(app, "决策").run()
        self.assertFalse(app.exception)

        prefix = f"player_decision_draft:{self.company_id}:1:"
        production = app.number_input(key=f"{prefix}production_volume")
        management = app.number_input(key=f"{prefix}management_investment")
        marketing = app.number_input(key=f"{prefix}city:广州:marketing_investment")
        choose(app, "决策")
        production.set_value(4321).run()
        choose(app, "决策")
        management.set_value(765_432.0).run()
        choose(app, "决策")
        marketing.set_value(234_567.0).run()

        choose(app, "报表").run()
        self.assertFalse(app.exception)
        choose(app, "决策").run()
        self.assertFalse(app.exception)

        self.assertEqual(
            app.number_input(key=f"{prefix}production_volume").value,
            4321,
        )
        self.assertEqual(
            app.number_input(key=f"{prefix}management_investment").value,
            765_432.0,
        )
        self.assertEqual(
            app.number_input(key=f"{prefix}city:广州:marketing_investment").value,
            234_567.0,
        )
        with self.db.connect() as conn:
            self.assertIsNone(
                self.db.one(
                    conn,
                    "SELECT 1 FROM decisions WHERE company_id=? AND round_no=1",
                    (self.company_id,),
                )
            )

    def test_genius_mode_admin_grant_preview_and_submission(self) -> None:
        from sim.engine import preview_round_operations
        from sim.genius import has_genius_access

        with self.db.connect() as conn:
            conn.execute("UPDATE companies SET patents=2 WHERE id=?", (self.company_id,))
            conn.execute("INSERT INTO agents(company_id,city,count) VALUES(?,'广州',1)", (self.company_id,))

        player = AppTest.from_file(self.app_path, default_timeout=30)
        player.session_state["auth"] = {"role": "player", "company_id": self.company_id}
        player.session_state["player_navigation"] = "决策"
        player.run()
        self.assertFalse(player.exception)
        self.assertFalse(any(toggle.label == "使用天才模式" for toggle in player.toggle))

        admin = AppTest.from_file(self.app_path, default_timeout=30)
        admin.session_state["auth"] = {"role": "admin"}
        admin.run()
        admin.sidebar.radio[0].set_value("队伍管理").run()
        grant = admin.checkbox(key=f"genius_grant_{self.company_id}")
        grant.check()
        next(btn for btn in admin.button if btn.label == "保存天才模式权限").click().run()
        self.assertFalse(admin.exception)
        with self.db.connect() as conn:
            self.assertTrue(has_genius_access(conn, self.company_id))

        player = AppTest.from_file(self.app_path, default_timeout=30)
        player.session_state["auth"] = {"role": "player", "company_id": self.company_id}
        player.session_state["player_navigation"] = "决策"
        player.run()
        self.assertFalse(player.exception)
        prefix = f"player_decision_draft:{self.company_id}:1:genius:"
        action = f"player_genius_action:{self.company_id}:1:"
        player.number_input(key=f"{prefix}reserve").set_value(1_000_000.125)
        player.number_input(key=f"{prefix}management_index").set_value(123.456)
        player.number_input(key=f"{prefix}quality_index").set_value(17.25)
        player.number_input(key=f"{prefix}深圳:mi_index").set_value(300.125)
        player.button(key=f"{action}calculate").click().run()
        self.assertFalse(player.exception)
        self.assertTrue(any("49.000%" in c.value and "广州" in c.value for c in player.caption))
        finance = player.dataframe[0].value
        pre_sales = finance.loc[finance["项目"] == "售前剩余现金", "预计余额"].iloc[0]
        self.assertGreaterEqual(pre_sales, 1_000_000.125)
        with self.db.connect() as conn:
            self.assertIsNone(self.db.one(conn, "SELECT 1 FROM decisions WHERE company_id=?", (self.company_id,)))

        player.number_input(key=f"{prefix}reserve").set_value(1_200_000.125).run()
        self.assertFalse(any(btn.key == f"{action}submit" for btn in player.button))
        player.button(key=f"{action}calculate").click().run()
        latest_finance = player.dataframe[0].value
        confirmed_balance = latest_finance.loc[latest_finance["项目"] == "售前剩余现金", "预计余额"].iloc[0]
        player.button(key=f"{action}submit").click().run()
        self.assertFalse(player.exception)
        with self.db.connect() as conn:
            submitted = self.db.one(conn, "SELECT * FROM decisions WHERE company_id=? AND round_no=1",
                                    (self.company_id,))
            self.assertIsNotNone(submitted["submitted_at"])
            self.assertGreater(submitted["production_volume"], 0)
            conn.execute("UPDATE companies SET home_city='广州' WHERE id<>?", (self.company_id,))
            operations = preview_round_operations(conn, 1)[self.company_id]
            old_products = self.db.one(conn, "SELECT product_inventory FROM companies WHERE id=?",
                                       (self.company_id,))["product_inventory"]
            self.assertAlmostEqual(submitted["management_investment"],
                                   123.456 * (operations["workers"] + operations["engineers"]), places=3)
            self.assertAlmostEqual(submitted["quality_investment"],
                                   17.25 * (old_products * 1.2 + submitted["production_volume"]), places=3)
            shenzhen = self.db.one(conn, "SELECT * FROM city_decisions WHERE company_id=? AND city='深圳'",
                                   (self.company_id,))
            self.assertEqual(shenzhen["agent_delta"], 1)
            self.assertAlmostEqual(shenzhen["marketing_investment"],
                                   300.125 * (operations["workers"] + operations["engineers"]), places=3)
            operations = preview_round_operations(conn, 1)[self.company_id]
            self.assertEqual(operations["produced"], submitted["production_volume"])
            self.assertAlmostEqual(operations["cash_pre_sales"], confirmed_balance, places=3)


if __name__ == "__main__":
    unittest.main()
