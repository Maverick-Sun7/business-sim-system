"""Replay selected Bot decisions against an immutable competition backup.

Each case starts from its saved pre-settlement snapshot. Other teams keep
their recorded decisions; only the requested Bot is re-planned. The backup
is opened read-only and every simulated payment happens in memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import bots, db, engine


def run_case(source: sqlite3.Connection, round_no: int, company_id: int | None) -> dict:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    source.backup(conn)
    try:
        snapshot = json.loads(db.one(conn, "SELECT snapshot_json FROM round_snapshots WHERE round_no=?", (round_no,))["snapshot_json"])
        db._restore_snapshot_state(conn, snapshot)
        for table in ("city_results", "market_round_stats", "results"):
            conn.execute(f"DELETE FROM {table} WHERE round_no>=?", (round_no,))
        for table in ("city_decisions", "decisions", "rounds", "round_snapshots", "round_bonuses"):
            conn.execute(f"DELETE FROM {table} WHERE round_no>?", (round_no,))
        conn.execute("UPDATE results SET total_assets=cash,net_assets=cash-debt")
        conn.execute("UPDATE rounds SET status='open',settled_at=NULL WHERE round_no=?", (round_no,))
        target_ids = (
            {company_id} if company_id is not None else {
                int(row["id"]) for row in db.all_rows(conn, "SELECT id FROM companies WHERE is_bot=1 AND is_super_bot=1")
            }
        )
        for target_id in target_ids:
            conn.execute("DELETE FROM city_decisions WHERE company_id=? AND round_no=?", (target_id, round_no))
            conn.execute("DELETE FROM decisions WHERE company_id=? AND round_no=?", (target_id, round_no))
        conn.commit()
        started = time.perf_counter()
        bots.submit_super_bot_decisions(conn, round_no, target_ids=target_ids, defer_rebalance=True)
        if company_id is None:
            bots.rebalance_super_bot_decisions(conn, round_no)
        elapsed = time.perf_counter() - started
        bots.finalize_super_bot_decisions(conn, round_no)
        engine.settle_round(conn, round_no)
        if company_id is None:
            rows = db.all_rows(conn, "SELECT r.*,c.is_super_bot,c.is_disruptor_bot FROM results r JOIN companies c ON c.id=r.company_id WHERE round_no=? ORDER BY net_assets DESC", (round_no,))
            return {
                "round": round_no, "mode": "whole-field", "seconds": elapsed,
                "teams": [{
                    "company_id": row["company_id"], "is_super_bot": row["is_super_bot"],
                    "is_disruptor_bot": row["is_disruptor_bot"], "profit": row["net_profit"],
                    "cash": row["cash"], "sold": row["sold"], "produced": row["produced"],
                    "stock": row["inventory"],
                    "research": json.loads(row["report_json"])["finance"]["research"],
                } for row in rows],
            }
        result = db.one(conn, "SELECT * FROM results WHERE company_id=? AND round_no=?", (company_id, round_no))
        report = json.loads(result["report_json"])
        routes = [item for item in report["sales"] if item["agents"] > 0]
        return {
            "round": round_no, "company_id": company_id,
            "profit": result["net_profit"], "ending_cash": result["cash"],
            "produced": result["produced"], "sold": result["sold"], "stock": result["inventory"],
            "ma": result["ma_index"], "qi": result["qi_index"],
            "mi": report["finance"]["marketing"], "research": report["finance"]["research"],
            "price_min": min((item["price"] for item in routes), default=0),
            "price_max": max((item["price"] for item in routes), default=0),
            "secondary_units": sum(item.get("secondary_units", 0) for item in routes),
            "seconds": elapsed,
        }
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--rounds", default="1,2,3,4,7")
    parser.add_argument("--companies", default="15,20")
    parser.add_argument("--whole-field", action="store_true", help="Replan every Super Bot together; keep human/ordinary Bot decisions recorded")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    with sqlite3.connect(args.source.resolve().as_uri() + "?mode=ro", uri=True) as source:
        source.row_factory = sqlite3.Row
        results = []
        for round_no in map(int, args.rounds.split(",")):
            for company_id in ([None] if args.whole_field else map(int, args.companies.split(","))):
                result = run_case(source, round_no, company_id)
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != original_hash:
        raise RuntimeError("Source backup changed during read-only replay")
    args.output.write_text(json.dumps({"source_sha256": original_hash, "cases": results}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
