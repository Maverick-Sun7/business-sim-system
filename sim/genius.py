"""Player-only production planning against the current KDS."""

import math
import sqlite3
from contextlib import closing
from fractions import Fraction
from math import gcd, lcm

from .db import all_rows, one, settings_dict, now_iso
from .engine import (
    available_loan_limit, current_company_net_assets, loan_ceiling_for_round,
    market_size, proportional_quits, settle_round, spend, weighted_salary_average,
)

DECISION_FIELDS = (
    "loan_change", "worker_delta", "worker_salary", "engineer_delta", "engineer_salary",
    "management_investment", "production_volume", "quality_investment", "research_investment",
)


def has_genius_access(conn: sqlite3.Connection, company_id: int) -> bool:
    return one(conn, "SELECT 1 FROM genius_mode_grants g JOIN companies c ON c.id=g.company_id "
               "WHERE g.company_id=? AND c.is_bot=0", (company_id,)) is not None


def set_genius_access(conn: sqlite3.Connection, company_id: int, enabled: bool) -> None:
    if not one(conn, "SELECT 1 FROM companies WHERE id=? AND is_bot=0", (company_id,)):
        raise ValueError("天才模式只能授予人类玩家。")
    if enabled:
        conn.execute("INSERT OR IGNORE INTO genius_mode_grants(company_id) VALUES(?)", (company_id,))
    else:
        conn.execute("DELETE FROM genius_mode_grants WHERE company_id=?", (company_id,))


def write_decision(conn, company_id, round_no, decision, cities):
    columns = ", ".join(DECISION_FIELDS)
    updates = ", ".join(f"{field}=excluded.{field}" for field in DECISION_FIELDS)
    placeholders = ", ".join("?" for _ in DECISION_FIELDS)
    conn.execute(
        f"INSERT INTO decisions(company_id, round_no, {columns}, submitted_at) "
        f"VALUES(?, ?, {placeholders}, ?) ON CONFLICT(company_id, round_no) DO UPDATE SET "
        f"{updates}, submitted_at=excluded.submitted_at, is_draft=0",
        (company_id, round_no, *(decision[field] for field in DECISION_FIELDS), now_iso()),
    )
    conn.executemany(
        "INSERT INTO city_decisions(company_id,round_no,city,agent_delta,marketing_investment,price,order_report) "
        "VALUES(?,?,?,?,?,?,?) ON CONFLICT(company_id,round_no,city) DO UPDATE SET "
        "agent_delta=excluded.agent_delta,marketing_investment=excluded.marketing_investment,"
        "price=excluded.price,order_report=excluded.order_report",
        [(company_id, round_no, city, values["agent_delta"], values["marketing_investment"],
          values["price"], values["order_report"]) for city, values in cities.items()],
    )


def _staff_planner(conn, company_id, round_no, role, wage, initial_wage):
    cohorts = all_rows(conn, "SELECT count, hire_round FROM employee_cohorts WHERE company_id=? AND role=? "
                       "ORDER BY hire_round, id", (company_id, role))
    current = sum(row["count"] for row in cohorts)
    experienced = sum(row["count"] for row in cohorts if round_no - row["hire_round"] >= 2)
    average = weighted_salary_average([(1, wage, initial_wage)], initial_wage)
    multiplier = min(wage / max(average, 1.), 1.10)

    def staff_delta(effective_required):
        if effective_required <= 0 or multiplier <= 0:
            return -current
        low = 0
        high = math.ceil(effective_required / (multiplier * min(1., multiplier))) + 2
        while low < high:
            count = (low + high) // 2
            retained = count - proportional_quits(count, wage, average)
            effective = (retained + .10 * min(retained, experienced)) * multiplier
            if effective >= effective_required:
                high = count
            else:
                low = count + 1
        return low - current

    return staff_delta


def _production_group(settings):
    """Return the smallest whole KDS production group as (workers, engineers, products)."""
    component_need = max(1, int(round(float(settings["components_per_product"]))))
    worker_rate = (Fraction(str(settings["component_workers"]))
                   * Fraction(str(settings["component_hours"])) * component_need / 504)
    engineer_rate = (Fraction(str(settings["product_engineers"]))
                     * Fraction(str(settings["product_hours"])) / 504)
    denominator = lcm(worker_rate.denominator, engineer_rate.denominator)
    worker_units = worker_rate.numerator * (denominator // worker_rate.denominator)
    engineer_units = engineer_rate.numerator * (denominator // engineer_rate.denominator)
    divisor = gcd(worker_units, engineer_units)
    if divisor == 0:
        raise ValueError("KDS 每件产品至少需要一种员工工时，无法计算生产组。")
    return worker_units // divisor, engineer_units // divisor, denominator // divisor


def plan_genius_decision(conn, company_id, round_no, parameters, cities):
    """Turn player-entered indices into a whole-group KDS plan on a private copy."""
    if not has_genius_access(conn, company_id):
        raise ValueError("此队伍当前未获授天才模式。")
    company = one(conn, "SELECT * FROM companies WHERE id=?", (company_id,))
    home = one(conn, "SELECT * FROM market_config WHERE city=?", (company["home_city"],))
    if home is None:
        raise ValueError("请先选择主场。")
    markets = {row["city"]: row for row in all_rows(conn, "SELECT * FROM market_config ORDER BY city")}
    if markets.keys() != cities.keys():
        raise ValueError("市场配置已变化，请刷新后重新计算。")
    for city, route in cities.items():
        current = one(conn, "SELECT count FROM agents WHERE company_id=? AND city=?", (company_id, city))
        if route["mi_index"] > 0 and max(0, int(current["count"]) if current else 0) + int(route["agent_delta"]) <= 0:
            raise ValueError(f"{city} 有 MI 指数但没有 Agent，无法产生市场效果。请先在普通决策中设置 Agent。")
    index_values = [parameters[name] for name in ("management_index", "quality_index")]
    index_values.extend(route["mi_index"] for route in cities.values())
    numeric_values = [*index_values, parameters["worker_salary"], parameters["engineer_salary"],
                      parameters["reserve"], parameters["loan_change"],
                      float(parameters.get("research_investment", 0.))]
    if any(not math.isfinite(value) for value in numeric_values):
        raise ValueError("工资、指数、留存现金和贷款必须是有效数字。")
    if any(value < 0 for value in [*index_values, parameters["worker_salary"],
                                   parameters["engineer_salary"], parameters["reserve"]]):
        raise ValueError("指数和预留现金请输入有效的非负数。")
    settings = settings_dict(conn)
    cash, reserve, loan = float(company["cash"]), parameters["reserve"], parameters["loan_change"]
    ceiling = loan_ceiling_for_round(round_no, home, settings["global_max_loan"])
    loan_limit = available_loan_limit(current_company_net_assets(conn, company),
                                     settings["loan_asset_threshold"], home["min_loan"], ceiling)
    if loan < -min(company["debt"], cash) or loan > loan_limit or 0 < loan < home["min_loan"]:
        raise ValueError("贷款或还款金额超出主场与本轮允许的范围。")
    staff = {
        role: _staff_planner(conn, company_id, round_no, role,
                             parameters[f"{role}_salary"], home[f"{role}_initial_salary"])
        for role in ("worker", "engineer")
    }
    component_count = max(1, int(round(float(settings["components_per_product"]))))
    workers_per_group, engineers_per_group, products_per_group = _production_group(settings)
    group = (workers_per_group, engineers_per_group, products_per_group)
    image = bytearray(conn.serialize())
    image[18:20] = b"\x01\x01"
    with closing(sqlite3.connect(":memory:")) as sandbox:
        sandbox.deserialize(bytes(image))
        sandbox.row_factory = sqlite3.Row
        # Scope labour and production estimates to the player's own team.
        # No rival bids, forecasts, or current-round wage data reach the player.
        sandbox.execute("DELETE FROM companies WHERE id<>?", (company_id,))
        sandbox.execute("DELETE FROM results WHERE round_no=?", (round_no,))

        def simulate(decision, routes):
            sandbox.execute("SAVEPOINT genius_probe")
            try:
                write_decision(sandbox, company_id, round_no, decision, routes)
                result = settle_round(sandbox, round_no, _operations_only=True)[company_id]
            finally:
                sandbox.execute("ROLLBACK TO genius_probe")
                sandbox.execute("RELEASE genius_probe")
            return result

        def evaluate(units):
            components = max(0, units * component_count - company["component_inventory"])
            decision = {
                "loan_change": loan, "worker_salary": parameters["worker_salary"],
                "engineer_salary": parameters["engineer_salary"],
                "management_investment": 0., "quality_investment": 0.,
                "research_investment": float(parameters.get("research_investment", 0.)),
                "production_volume": units,
                "worker_delta": staff["worker"](
                    components * settings["component_workers"] * settings["component_hours"] / 504.),
                "engineer_delta": staff["engineer"](
                    units * settings["product_engineers"] * settings["product_hours"] / 504.),
            }
            base_routes = {city: {**route, "marketing_investment": 0.}
                           for city, route in cities.items()}
            base_state = simulate(decision, base_routes)
            people = max(0, int(base_state["workers"]) + int(base_state["engineers"]))
            decision["management_investment"] = parameters["management_index"] * people
            decision["quality_investment"] = parameters["quality_index"] * (
                max(0, float(company["product_inventory"])) * 1.2 + units
            )
            planned_routes = {
                city: {**route, "marketing_investment": route["mi_index"] * people}
                for city, route in cities.items()
            }
            state = simulate(decision, planned_routes)
            routes = state["city_decisions"]
            fits = (
                state["produced"] == units and state["cash_pre_sales"] + 1e-6 >= reserve
                and state["layoff_debt"] + state["quit_penalty_debt"] < 1e-6
                and math.isclose(state["management"], decision["management_investment"], rel_tol=0, abs_tol=1e-6)
                and math.isclose(state["quality"], decision["quality_investment"], rel_tol=0, abs_tol=1e-6)
                and all(
                    routes[city]["agent_delta_actual"] == route["agent_delta"]
                    and (route["marketing_investment"] == 0 or routes[city]["agents_after"] > 0)
                    and math.isclose(routes[city]["marketing_investment"], route["marketing_investment"],
                                     rel_tol=0, abs_tol=1e-6)
                    for city, route in planned_routes.items()
                )
            )
            return fits, decision, state, planned_routes

        low_groups, high_groups = 0, 1
        low = 0
        while True:
            high = high_groups * products_per_group
            fits, proposal, estimate, routes = evaluate(high)
            if not fits:
                break
            low_groups = high_groups
            low, decision, state, planned_cities = high, proposal, estimate, routes
            if high_groups >= 2**30:
                break
            high_groups *= 2
        while low_groups + 1 < high_groups:
            mid_groups = (low_groups + high_groups) // 2
            mid = mid_groups * products_per_group
            fits, proposal, estimate, routes = evaluate(mid)
            if fits:
                low_groups = mid_groups
                low, decision, state, planned_cities = mid, proposal, estimate, routes
            else:
                high_groups = mid_groups
                high = mid

        if low == 0:
            raise ValueError("按当前工资、指数、贷款和留存现金，资金不足以完成一整组生产。请降低指数或留存金额，或调整工资和贷款。")

    return {
        "decision": decision, "cities": planned_cities, "home": home["city"],
        "patents": int(company["patents"]), "material_factor": settings["patent_factor"] ** company["patents"],
        "workers": state["workers"], "engineers": state["engineers"], "stock": state["available"],
        "group": {"workers": group[0], "engineers": group[1], "products": group[2],
                  "count": decision["production_volume"] // products_per_group},
        "investments": {"ma": decision["management_investment"], "qi": decision["quality_investment"],
                        "mi": {city: route["marketing_investment"] for city, route in planned_cities.items()}},
        "cash_before_sales": state["cash_pre_sales"],
        "cashflow": _estimate_cashflow(company, state, markets, settings, round_no, parameters["sell_through"]),
    }


def _estimate_cashflow(company, state, markets, settings, round_no, sell_through):
    balance = float(company["cash"])
    rows = [("期初现金", None, balance)]
    for label, amount in (
        ("贷款变化", state["loan_change"]), ("工人工资", -state["worker_wage_cost"]),
        ("工程师工资", -state["engineer_wage_cost"]), ("裁员补偿", -state["layoff_cash"]),
        ("离职补偿", -state["quit_penalty_cash"]), ("培训费", -state["training_cost"]),
        ("零件材料", -state["component_material_cost"]), ("零件仓储", -state["component_storage_cost"]),
        ("产品材料", -state["product_material_cost"]), ("产品仓储", -state["product_storage_cost"]),
        ("Agent 变更", -state["agent_cost"]), ("城市营销 MI", -state["marketing_total"]),
        ("品质投入 QI", -state["quality"]), ("管理投入 MA", -state["management"]),
    ):
        balance += amount
        rows.append((label, amount, balance))
    balance = state["cash_pre_sales"]
    rows.append(("售前剩余现金", None, balance))
    capacities = {
        city: max(0, math.floor(market_size(markets[city], round_no, settings["market_growth"])))
        for city, route in state["city_decisions"].items() if route["agents_after"] > 0
    }
    capacity = sum(capacities.values())
    sold = max(0, min(math.floor(state["available"] * sell_through / 100), capacity))
    shipments = {}
    if sold:
        shares = {city: divmod(sold * limit, capacity) for city, limit in capacities.items()}
        shipments = {city: whole for city, (whole, _) in shares.items()}
        remainder = sold - sum(shipments.values())
        order = sorted(shares, key=lambda city: (-shares[city][1], city))
        for city in order[:remainder]:
            shipments[city] += 1
    revenue = sum(units * state["city_decisions"][city]["price"] for city, units in shipments.items())
    transport_request = sum(units for city, units in shipments.items() if city != company["home_city"])
    transport_request *= settings["transport_cost"]
    rows.append(("假设销售收入", revenue, balance + revenue))
    balance, research = spend(balance + revenue, state["research_requested"])
    rows.append(("研发 / 专利投入", -research, balance))
    balance, transport = spend(balance, transport_request)
    rows.append(("运输费", -transport, balance))
    report_cost = settings["report_cost"]
    reports = 0.
    for route in state["city_decisions"].values():
        if route["report_requested"] and balance + 1e-9 >= report_cost:
            balance, paid = spend(balance, report_cost)
            reports += paid
    rows.append(("市场报告", -reports, balance))
    interest = max(0., state["debt_before_interest"]) * state["home"]["interest_rate"]
    operating_cost = company["cash"] + state["loan_change"] - state["cash_pre_sales"] + research + transport + reports
    balance, tax = spend(balance, max(0., (revenue - operating_cost - interest) * settings["tax_rate"]))
    rows.append(("所得税", -tax, balance))
    rows.append(("预计期末现金", None, balance))
    return {"rows": rows, "sold": sold, "debt": state["debt_before_interest"] + interest,
            "cash_after_sales": balance}
