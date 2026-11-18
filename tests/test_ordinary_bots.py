import json
from unittest.mock import patch

import pytest

from sim import db
from sim.bots import submit_bot_decisions
from sim.engine import settle_round
from sim.ordinary_bots import (
    _group, _market_pressure, _new_production_target, _production_sale_credit, forecast_market_sales,
)


def _seed(tmp_path, monkeypatch, count):
    monkeypatch.setattr(db, 'DB_PATH', tmp_path / 'ordinary-bots.db')
    db.init_db()
    with db.connect() as conn:
        conn.execute('DELETE FROM companies')
        db.set_setting(conn, 'total_rounds', 7)
        cities = [row[0] for row in conn.execute('SELECT city FROM market_config ORDER BY city')]
        stamp = db.now_iso()
        for i in range(count):
            city = cities[i % len(cities)]
            company_id = conn.execute(
                'INSERT INTO companies(code,name,password_hash,home_city,cash,is_bot,bot_profile,setup_submitted_at,created_at) '
                'VALUES(?,?,?,?,15000000,1,?,?,?)',
                (f'BOT{i:03}', f'Bot {i}', 'not-used', city, i % 12, stamp, stamp),
            ).lastrowid
            conn.execute('INSERT INTO agents(company_id,city,count) VALUES(?,?,1)', (company_id, city))
        db.start_competition(conn, 30, False)


def test_group_is_derived_from_the_current_recipe():
    case = dict(component_workers=3, component_hours=7, components_per_product=7, product_engineers=4, product_hours=14)
    parts, workers_per_product, engineers_per_product, products = _group(case)
    assert (round(workers_per_product * products), round(engineers_per_product * products), parts * products, products) == (21, 8, 504, 72)
    case.update(component_workers=6, component_hours=4, components_per_product=5, product_engineers=3, product_hours=24)
    parts, workers_per_product, engineers_per_product, products = _group(case)
    assert (round(workers_per_product * products), round(engineers_per_product * products), parts * products, products) == (5, 3, 105, 21)


def test_recent_capacity_pressure_only_discounts_secondary_sales_in_the_full_city():
    pressure = _market_pressure(.82, 1.)
    assert pressure == pytest.approx(.7)
    assert _market_pressure(.82, .85) == 0.
    assert _market_pressure(1., .85) == pytest.approx(1.)
    markets = {'full': {'production_pressure': pressure}, 'open': {'production_pressure': 0.}}
    forecast = {'sold_units': {'full': 100, 'open': 100}, 'primary': {'full': 50., 'open': 30.}}
    sale_credit = _production_sale_credit(forecast, ('full', 'open'), markets)
    assert sale_credit == pytest.approx(186.)
    assert _new_production_target(sale_credit, 100) == 86
    assert _new_production_target(sale_credit, 190) == 0


def test_ordinary_investments_do_not_depend_on_random_match_seed(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, 12)
    with db.connect() as conn:
        ids = [row[0] for row in conn.execute('SELECT id FROM companies ORDER BY id DESC LIMIT 2')]
        conn.execute('UPDATE companies SET is_bot=0 WHERE id=?', (ids[0],))
        conn.execute('UPDATE companies SET is_super_bot=1 WHERE id=?', (ids[1],))
        outputs = []
        for seed in ('first', 'second'):
            db.set_setting(conn, 'ordinary_bot_match_seed', seed)
            assert submit_bot_decisions(conn, 1) == 10
            assert submit_bot_decisions(conn, 1) == 0
            outputs.append((
                [tuple(r) for r in conn.execute('SELECT company_id,worker_delta,engineer_delta,management_investment,'
                                                 'quality_investment,production_volume FROM decisions ORDER BY company_id')],
                [tuple(r) for r in conn.execute('SELECT company_id,city,price,agent_delta,marketing_investment FROM '
                                                 'city_decisions ORDER BY company_id,city')],
            ))
            conn.execute('DELETE FROM city_decisions'); conn.execute('DELETE FROM decisions')
        assert outputs[0] == outputs[1]
        assert len({r[1:] for r in outputs[0][0]}) > 4


def test_plain_bot_field_keeps_growing_without_heavy_search(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, 120)
    with db.connect() as conn:
        assets = []
        with patch('sim.ordinary_bots.forecast_market_sales', wraps=forecast_market_sales) as forecast:
            for round_no in range(1, 8):
                assert submit_bot_decisions(conn, round_no) == 120
                settle_round(conn, round_no)
                assets.append(dict(conn.execute('SELECT company_id,net_assets FROM results WHERE round_no=?', (round_no,))))
                if round_no < 7:
                    db.open_next_round(conn, round_no, 30)
            assert forecast.call_count == 7
        assert all(len(round_assets) == 120 for round_assets in assets)
        assert sum(assets[4][k] < assets[5][k] < v for k, v in assets[6].items()) >= 10
        assert min(assets[6].values()) > 0
        for row in conn.execute('SELECT report_json FROM results'):
            finance = json.loads(row[0])['finance']
            spending = sum(finance[key] for key in ('wages', 'layoff', 'quit_penalty', 'training', 'materials',
                                                     'storage', 'agents', 'marketing', 'quality', 'management'))
            assert spending <= finance['round_begins'] + 1e-5
