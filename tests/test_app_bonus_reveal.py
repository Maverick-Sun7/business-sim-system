from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app
from sim import db
from sim.engine import settle_round


def submit_no_operations(conn, round_no):
    conn.execute(
        "INSERT INTO decisions(company_id,round_no,worker_salary,engineer_salary,submitted_at) "
        "SELECT id,?,3300,6400,? FROM companies",
        (round_no, db.now_iso()),
    )


@pytest.fixture
def settled_round(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "bonus.db")
    db.init_db()
    with db.connect() as conn:
        conn.execute("UPDATE companies SET home_city='广州',setup_submitted_at=?", (db.now_iso(),))
        conn.execute("UPDATE rounds SET status='open',starts_at=? WHERE round_no=1", (db.now_iso(),))
        teams = db.all_rows(conn, "SELECT id FROM companies ORDER BY id")
        leader, recipient = [int(row["id"]) for row in teams[:2]]
        conn.execute("UPDATE companies SET cash=cash+100000 WHERE id=?", (leader,))
        submit_no_operations(conn, 1)
        settle_round(conn, 1)
    return leader, recipient


def make_app(auth):
    instance = AppTest.from_file(str(Path(app.__file__)), default_timeout=25)
    instance.session_state["auth"] = auth
    return instance.run()


def test_opening_bonus_reveals_new_rank_and_assets_after_countdown(settled_round):
    leader, recipient = settled_round
    player = make_app({"role": "player", "company_id": recipient})
    assert not player.exception
    assert any("Initial Assets" in item.value for item in player.markdown)

    admin = make_app({"role": "admin"})
    admin.sidebar.radio[0].set_value("回合控制").run()
    admin.number_input(key=f"bonus_2_{recipient}").set_value(250_000)
    next(button for button in admin.button if "发放 Bonus 并开启第 2 轮" == button.label).click().run()
    assert not admin.exception
    with db.connect() as conn:
        assert app.rank_rows(conn, 1)[0]["id"] == recipient
        original = app.rank_rows(conn, 1, settled_only=True)
        assert original[0]["id"] == leader
        original_cash = next(row["cash"] for row in original if row["id"] == recipient)
        assert db.one(conn, "SELECT cash FROM companies WHERE id=?", (recipient,))[0] == original_cash + 250_000
        assert [row["net_assets"] for row in app.wealth_result_rows(conn, company_id=recipient)] == [original_cash + 250_000]

    player.session_state["player_round_countdown_pending_round"] = 2
    player.run()
    assert not player.exception
    assert any("Initial Assets" in item.value for item in player.markdown)
    del player.session_state["player_round_countdown_pending_round"]
    player.run()
    overview = next(item.value for item in player.markdown if "Round 1 Summary" in item.value)
    assert 'class="position">#1' in overview
    assert app.money(original_cash + 250_000) in overview
    assert "Bonus" not in overview

    navigation = next(item for item in player.get("button_group") if item.key == "player_navigation")
    navigation.set_value("排名" if isinstance(navigation.value, str) else ["排名"]).run()
    assert not player.exception
    standings = next(item.value for item in player.markdown if 'class="ranking-card"' in item.value)
    assert 'class="ranking-row is-me"><div class="ranking-avatar">●</div><div class="ranking-position">1</div>' in standings
    assert "Bonus" not in standings and "250000" not in standings and "¥" not in standings


@pytest.mark.parametrize("legacy", [False, True])
def test_bonus_not_paid_twice_and_rollback_removes_only_discarded_credit(settled_round, legacy):
    _, recipient = settled_round
    with db.connect() as conn:
        before = db.one(conn, "SELECT cash FROM companies WHERE id=?", (recipient,))[0]
        db.open_next_round(conn, 1, 15, {recipient: 250_000})
        db.open_next_round(conn, 1, 15, {recipient: 250_000})
        submit_no_operations(conn, 2)
        settle_round(conn, 2)
        displayed = app.wealth_result_rows(conn, company_id=recipient)
        assert [(r["round_no"], r["net_assets"]) for r in displayed] == [(1, before + 250_000), (2, before + 250_000)]
        db.open_next_round(conn, 2, 15, {recipient: 500_000})
        if legacy:
            conn.execute("DELETE FROM round_snapshots WHERE round_no=2")
        assert db.rollback_latest_settled_round(conn) == 2
        assert [r["net_assets"] for r in app.wealth_result_rows(conn, company_id=recipient)] == [before + 250_000]
        assert db.one(conn, "SELECT cash FROM companies WHERE id=?", (recipient,))[0] == before + 250_000
