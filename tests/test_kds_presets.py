import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sim import db


@pytest.fixture
def database(tmp_path):
    with patch.object(db, "DB_PATH", tmp_path / "presets.db"):
        db.init_db()
        yield


def test_preset_roundtrip_only_updates_kds(database):
    with db.connect() as conn:
        assert [row[0] for row in db.all_rows(conn, "SELECT name FROM kds_presets")] == ["25人OBOS"]
        payload = db.kds_snapshot(conn)
        payload["settings"].update(initial_cash=18_000_000, cpi_price_power=12)
        payload["markets"] = [r for r in payload["markets"] if r["city"] in {"广州", "深圳"}]
        payload["markets"][0].update(transport_cost=123, worker_training_cost=987)
        db.save_kds_preset(conn, "两城场", payload)
        db.set_setting(conn, "test_round_enabled", 1)
        db.set_setting(conn, "total_rounds", 8)
        db.set_setting(conn, "round_duration_minutes", 17)
        conn.execute("UPDATE companies SET home_city='广州',cash=123 WHERE id=1")
        conn.execute("INSERT INTO agents(company_id,city,count) VALUES(1,'广州',1)")
        assert len(db.kds_snapshot(conn)["markets"]) == 10
    with db.connect() as conn:
        db.apply_kds_preset(conn, "两城场")
        assert db.kds_snapshot(conn) == payload
        assert db.get_setting(conn, "test_round_enabled") == 1
        assert db.get_setting(conn, "total_rounds") == 8
        assert db.get_setting(conn, "round_duration_minutes") == 17
        assert db.one(conn, "SELECT home_city,cash FROM companies WHERE id=1")[:] == ("广州", 123)
        assert db.one(conn, "SELECT cash FROM companies WHERE id=2")[0] == 18_000_000
        assert db.one(conn, "SELECT city,count FROM agents WHERE company_id=1")[:] == ("广州", 1)


def test_blocks_conflicts_started_rounds_and_invalid_or_duplicate_import(database):
    with db.connect() as conn:
        payload = db.kds_snapshot(conn)
        payload["markets"] = [r for r in payload["markets"] if r["city"] != "广州"]
        db.save_kds_preset(conn, "备用", payload)
        conn.execute("UPDATE companies SET home_city='广州' WHERE id=1")
    with pytest.raises(ValueError, match="不支持的主场"), db.connect() as conn:
        db.apply_kds_preset(conn, "备用")
    with db.connect() as conn:
        assert len(db.kds_snapshot(conn)["markets"]) == 10
        conn.execute("UPDATE rounds SET status='settled'")
    with pytest.raises(ValueError, match="只能在开赛前"), db.connect() as conn:
        db.apply_kds_preset(conn, "备用")
    with pytest.raises(ValueError, match="已经存在"), db.connect() as conn:
        db.save_kds_preset(conn, "备用")
    with pytest.raises(ValueError, match="有效的非负数字"), db.connect() as conn:
        payload["markets"][0]["population"] = float("nan")
        db.save_kds_preset(conn, "非法", payload)


def test_upgrade_archives_exact_small_field_snapshot_once_and_allows_switching(database):
    with db.connect() as conn:
        conn.execute("DROP TABLE kds_presets")  # Simulate a pre-library database.
        conn.execute("DELETE FROM market_config WHERE city NOT IN ('深圳','苏州')")
        conn.execute("UPDATE market_config SET population=1500000,initial_avg_price=840,max_loan=2400000 WHERE city='深圳'")
        conn.execute("UPDATE market_config SET population=6000000,initial_avg_price=710,max_loan=1500000 WHERE city='苏州'")
        for key, value in dict(component_hours=2, product_engineers=1, product_hours=4,
                               components_per_product=2, price_min=320, price_max=2000, cpi_price_power=17).items():
            db.set_setting(conn, key, value)
        exact = db.kds_snapshot(conn)
    db.init_db()
    with db.connect() as conn:
        assert db.kds_snapshot(conn) == exact
        db.set_setting(conn, "cpi_price_power", 2)
    db.init_db()
    with db.connect() as conn:
        archived = db.one(conn, "SELECT payload_json FROM kds_presets WHERE name='3人2000场'")
        assert json.loads(archived[0]) == exact
        assert db.get_setting(conn, "cpi_price_power") == 2
        db.apply_kds_preset(conn, "25人OBOS")
        obos = db.kds_snapshot(conn)
        assert obos["settings"]["price_max"] == 25_000
        assert obos["settings"]["price_min"] == 3_500
        assert len(obos["markets"]) == 10
    with db.connect() as conn:
        db.apply_kds_preset(conn, "3人2000场")
        assert db.kds_snapshot(conn) == exact


def test_admin_save_and_apply_refresh_the_fields(database):
    from streamlit.testing.v1 import AppTest

    with db.connect() as conn:
        db.set_setting(conn, "initial_cash", 18_000_000)
        db.save_kds_preset(conn, "另一个城市版本")
        db.set_setting(conn, "initial_cash", 15_000_000)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=25)
    app.session_state["auth"] = {"role": "admin"}
    app.run()
    app.sidebar.radio[0].set_value("KDS 设置").run()
    assert not app.exception
    next(widget for widget in app.text_input if widget.label == "新预设名称").set_value("手动备份")
    next(widget for widget in app.button if widget.label == "将当前 KDS 存为预设").click().run()
    assert {"25人OBOS", "手动备份"} <= set(app.selectbox(key="kds_preset_selection").options)
    app.selectbox(key="kds_preset_selection").set_value("另一个城市版本").run()
    app.button(key="kds_preset_apply").click().run()
    assert not app.exception
    assert next(widget for widget in app.number_input if widget.label == "初始现金").value == 18_000_000
