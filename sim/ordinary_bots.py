from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction

from .bot_market_forecast import forecast_market_sales
from .cpi import agent_mi_benefit, investment_price_factor
from .db import now_iso
from .defaults import DEFAULT_SETTINGS
from .engine import market_size, proportional_quits


@dataclass(frozen=True)
class Route:
    city: str
    price: float
    agents: int
    added: int
    marketing: float
    demand: float
    net_price: float


@dataclass(frozen=True)
class Budget:
    units: int
    workers: int
    engineers: int
    ma: float
    qi: float
    cost: float
    headcount: int
    materials: float


def _group(settings):
    parts = max(1, round(float(settings['components_per_product'])))
    worker_rate = Fraction(str(settings['component_workers'])) * Fraction(str(settings['component_hours'])) * parts / 504
    engineer_rate = Fraction(str(settings['product_engineers'])) * Fraction(str(settings['product_hours'])) / 504
    products = math.lcm(worker_rate.denominator, engineer_rate.denominator)
    return parts, float(worker_rate), float(engineer_rate), products


def _wage(bot, home, history, report, role, settings):
    initial = float(home[f'{role}_initial_salary'])
    previous = float(history.get(f'{role}_salary', initial))
    minimum, maximum = float(settings['salary_min']), float(settings['salary_max'])
    previous = min(maximum, max(minimum, previous))
    step = max(0.0, float(settings['salary_change_limit']))
    average = float(report.get('human_resources', {}).get(f'average_{role}_salary') or initial)
    premium = 0.05 + 0.045 * (int(bot['bot_profile']) % 7) / 6
    salary = min(maximum, previous + step, max(minimum, previous - step, average * (1 + premium)))
    return salary


def _staff(required, cohorts, multiplier):
    current = sum(count for count, _ in cohorts)
    effective = sum(count * factor for count, factor in cohorts)
    if required <= 0:
        return 0
    required = required / max(0.01, multiplier)
    if effective < required - 1e-9:
        return current + math.ceil(required - effective - 1e-9)
    removable = 0
    for count, factor in cohorts:
        n = min(count, max(0, math.floor((effective - required) / factor + 1e-9)))
        effective -= n * factor
        removable += n
    return current - removable


def _workforce(required, cohorts, salary, average):
    ratio = min(1., salary / max(1., average))
    retained = _staff(required, cohorts, ratio)
    if ratio >= 1 or not retained:
        return retained, 0
    low, high = retained, math.ceil(retained / max(.01, ratio))
    while low < high:
        mid = (low + high) // 2
        if mid - proportional_quits(mid, salary, average) >= retained:
            high = mid
        else:
            low = mid + 1
    return low, proportional_quits(low, salary, average)


def _fund(batch, cash, target, budget):
    if target < batch:
        return budget(0)
    high = target // batch
    low = 0
    while low < high:
        mid = (low + high + 1) // 2
        if budget(mid * batch).cost <= cash + 1e-7:
            low = mid
        else:
            high = mid - 1
    return budget(low * batch)


def _market_pressure(latest_fill, older_fill):
    # A recently full city does not become an expansion market after one soft round.
    return max(0., min(1., (max(latest_fill, older_fill * .97) - .90) / .10))


def _production_sale_credit(prediction, routes, markets):
    credit = 0.
    for city in routes:
        sold = prediction['sold_units'][city]
        primary = min(sold, prediction['primary'][city])
        # Forecast secondary orders still count, but when a city is full they
        # can disappear as competitors liquidate stock or change their bids.
        credit += primary + (sold - primary) * (1. - .40 * markets[city]['production_pressure'])
    return credit


def _new_production_target(sale_credit, old_stock):
    return max(0, math.floor(sale_credit) - old_stock)


def _build_plan(bot, slot, settings, group, markets, previous, last_routes, agents, cohorts, traffic, field_size, round_no):
    key = int(bot['id'])
    home_city = str(bot['home_city'] or next(iter(markets)))
    home = markets.get(home_city, next(iter(markets.values())))
    cash = max(0., float(bot['cash']))
    old_products, old_parts = int(bot['product_inventory']), int(bot['component_inventory'])
    report = json.loads(previous['report_json']) if 'report_json' in previous else {}
    cw = cohorts.get((key, 'worker'), ())
    ce = cohorts.get((key, 'engineer'), ())
    workers_before, engineers_before = (sum(n for n, _ in cohort) for cohort in (cw, ce))
    worker_wage = _wage(bot, home, previous, report, 'worker', settings)
    engineer_wage = _wage(bot, home, previous, report, 'engineer', settings)
    parts, worker_rate, engineer_rate, batch = group
    factor = float(settings['patent_factor']) ** int(bot['patents'])
    part_material = factor * float(home['component_material'])
    product_material = factor * float(home['product_material'])
    part_storage, product_storage = float(home['component_storage']), float(home['product_storage'])
    worker_training = float(settings['worker_training_cost'])
    engineer_training = float(settings['engineer_training_cost'])
    base_unit_cost = (
        worker_rate * (3 * worker_wage + worker_training)
        + engineer_rate * (3 * engineer_wage + engineer_training)
        + parts * (part_material + part_storage) + product_material + product_storage
    )
    before = report.get('production', {})
    previous_stock = float(before.get('old_products', 0)) + float(before.get('produced', previous.get('produced', 0)))
    sell_through = float(before.get('sold', previous.get('sold', 0))) / previous_stock if previous_stock else 1.
    late = round_no > 1 and round_no >= int(settings['total_rounds']) - 1
    distressed = old_products >= max(1, min(batch, 72)) and sell_through < .5
    ma_large = max(0., float(settings['cpi_ma_large_threshold']))
    transport = max(0., float(settings['transport_cost']))
    agent_cost = max(0., float(settings['agent_add_cost']))
    max_add = max(0, int(settings['max_agent_add_per_city_round']))
    breadth = min(len(markets), 1 + min(3, max(0, round_no - 1)) + int(slot % 3 == 0))
    price_power = max(1, int(settings['cpi_price_power']))
    spread = ((slot % 9) + (slot // 9 % 7) / 8) / 9
    has_route = any(agents.get((key, city), 0) for city in markets)
    choices = []
    for city, market in markets.items():
        existing = agents.get((key, city), 0)
        if not existing and not max_add:
            continue
        size, avg, fill = market['market_size'], market['base_average_price'], market['fill']
        if size <= 0:
            continue
        city_fee = 0. if city == home_city else transport
        top = min(float(market['max_price']), float(settings['price_max']))
        floor = min(top, float(settings['price_min']))
        previous_route = last_routes.get((key, city), {})
        fair = size * (.52 + .42 * fill) / max(1., traffic[city] + (not existing))
        last_sold = float(previous_route.get('sold', 0)) * float(settings['market_growth'])
        demand = max(fair * .70, min(fair * 2.35, .55 * fair + .78 * last_sold))
        saturated = fill >= .82
        value_lane = (late or saturated or avg >= .75 * top) and (slot % 4 != 0 or sell_through < .72 or field_size <= 2)
        premium = top * (.965 + .028 * spread)
        if saturated or late:
            premium = min(premium, max(avg * (1.08 + .12 * spread), (base_unit_cost + city_fee) * 1.28))
        gap = (.15 + .12 * spread) ** (8. / price_power)
        value = avg * (1 - gap)
        normal_floor = (base_unit_cost + city_fee) * (1.16 + .08 * spread)
        can_discount = value >= normal_floor
        price = value if value_lane and can_discount else premium
        if distressed and existing:
            price = min(price, avg * (1 - max(gap, .25)))
        price = min(top, max(floor, round(price)))
        net_price = price - city_fee
        if net_price <= 0 or (net_price <= base_unit_cost * 1.02 and not distressed):
            continue
        expected_margin = demand * max(0., net_price - base_unit_cost)
        opening_cost = 0. if existing else agent_cost / max(1, int(settings['total_rounds']) - round_no + 1)
        route_rank = expected_margin - opening_cost
        route_rank *= 1. + (.14 if city == home_city else 0.) + (.10 if existing else 0.)
        if existing or route_rank > 0:
            choices.append((route_rank, city, price, existing, demand, net_price, fill))

    # Choose routes by expected contribution before committing to production.
    choices.sort(key=lambda item: (-item[0], item[1]))
    active: dict[str, Route] = {}
    route_fills = []
    fixed = 0.
    total_planned_demand = 0.
    for _, city, price, existing, demand, net_price, fill in choices:
        if len(active) >= breadth and not existing:
            continue
        added = 0 if existing else 1
        if added and (fixed + agent_cost > cash * .16 or (distressed and has_route)):
            continue
        if added and total_planned_demand * base_unit_cost >= cash * 1.05 and not old_products and round_no < 3:
            continue
        fixed += added * agent_cost
        total_planned_demand += demand
        active[city] = Route(city, price, existing + added, added, 0., demand, net_price)
        route_fills.append((city, fill))

    primary = max(active.values(), key=lambda route: route.demand * route.net_price, default=None)
    qi_large = float(markets[primary.city]['max_price']) / 50. if primary else 0.
    typical_profit = max(0., primary.net_price - base_unit_cost) if primary else 0.
    contest = max((fill for _, fill in route_fills), default=0.)
    ma_index = 0.
    qi_index = 0.
    if primary and not distressed and typical_profit > 0:
        average = markets[primary.city]['base_average_price']
        maximum = markets[primary.city]['max_price']
        observed = markets[primary.city]['pool_indices']
        current_factor = investment_price_factor(primary.price, average, maximum)
        ma_entry = min(2.15, max(.80, 1.06 / max(.15, current_factor)))
        qi_entry = ma_entry
        intensity = min(1.75, 1. + .28 * contest + .20 * (1 - sell_through) + .18 * spread)
        if contest < .70 and slot % 5 == 0:
            ma_index = ma_large * (.38 + .12 * spread)
        elif slot % 5 == 1:
            qi_index = qi_large * qi_entry * intensity
        else:
            ma_index = min(ma_large * 3.3, max(ma_large * ma_entry, observed[0] * .76) * intensity)
            if contest >= .72 and slot % 5 in (2, 4):
                qi_index = min(qi_large * 3., max(qi_large * qi_entry, observed[1] * .76) * intensity)
        if ma_index * (worker_rate + engineer_rate) > typical_profit * .29:
            ma_index = 0.
        if qi_index > typical_profit * .24:
            qi_index = 0.

    marketing_spend = 0.
    for city, fill in route_fills:
        route = active[city]
        market = markets[city]
        if distressed or round_no < 3 or (slot % 5 not in (2, 3, 4) and fill < .84):
            continue
        if not market['mi_active'] and fill < .52:
            continue
        new_agent = int(route.added < max_add and route.agents < 5 and round_no < int(settings['total_rounds']))
        before_threshold = float(market['max_price']) / 50. * float(market['market_size']) / 15.
        if before_threshold <= 0:
            continue
        saving = before_threshold * .10 / (agent_mi_benefit(route.agents) * agent_mi_benefit(route.agents + 1))
        new_agent = new_agent if saving > agent_cost * .80 else 0
        entry = before_threshold / agent_mi_benefit(route.agents + new_agent)
        relative = investment_price_factor(route.price, market['base_average_price'], market['max_price'])
        threshold = entry * max(1., min(2.2, 1.06 / max(.15, relative)))
        contest_margin = route.demand * max(0., route.net_price - base_unit_cost)
        bid = max(threshold, min(entry * 1.65, market['pool_indices'][2] / agent_mi_benefit(route.agents + new_agent)))
        invest = round(bid * (1.02 + .21 * fill + .14 * spread), 2)
        if invest + new_agent * agent_cost > min(contest_margin * .37, cash * .20 - marketing_spend - fixed):
            continue
        fixed += new_agent * agent_cost
        marketing_spend += invest
        active[city] = Route(city, route.price, route.agents + new_agent, route.added + new_agent, invest, route.demand, route.net_price)

    stock_ma = ma_large * .38 if old_products and any(
        route.price >= markets[route.city]['base_average_price'] for route in active.values()
    ) else 0.
    old_component_capacity = max(old_parts, int(bot['component_storage_capacity']))
    old_product_capacity = max(old_products, int(bot['product_storage_capacity']))

    def planning_average(role, wage):
        initial = float(home[f'{role}_initial_salary'])
        avg = float(report.get('human_resources', {}).get(f'average_{role}_salary') or initial)
        return max((wage + 2 * initial) / 3., avg * 1.045)

    worker_average = planning_average('worker', worker_wage)
    engineer_average = planning_average('engineer', engineer_wage)

    def build_budget(units):
        # MA and QI grow with the group; Agent and MI costs stay fixed while
        # finding the largest affordable production plan for this portfolio.
        components = max(0, units * parts - old_parts)
        workers, worker_quits = _workforce(components * worker_rate / parts, cw, worker_wage, worker_average)
        engineers, engineer_quits = _workforce(units * engineer_rate, ce, engineer_wage, engineer_average)
        net_workers, net_engineers = workers - workers_before, engineers - engineers_before
        wages = (workers * worker_wage + engineers * engineer_wage) * 3
        layoff = max(0, -net_workers) * worker_wage + max(0, -net_engineers) * engineer_wage
        quit_cost = 2 * (worker_quits * worker_wage + engineer_quits * engineer_wage)
        training = max(0, net_workers) * worker_training + max(0, net_engineers) * engineer_training
        materials = components * part_material + units * product_material
        storage = max(0, old_parts + components - old_component_capacity) * part_storage
        storage += max(0, old_products + units - old_product_capacity) * product_storage
        ma = round(ma_index * max(1, workers + engineers - worker_quits - engineer_quits), 2)
        qi = round(qi_index * (old_products * 1.2 + units), 2)
        if not units:
            ma = stock_ma
            qi = 0.
        return Budget(units, net_workers, net_engineers, ma, qi,
                      wages + layoff + quit_cost + training + materials + storage + fixed + marketing_spend + ma + qi,
                      workers + engineers - worker_quits - engineer_quits, materials)

    demand = sum(route.demand for route in active.values())
    headroom = 1.65 if contest < .78 else 1.32
    supply_goal = sum(
        route.demand * (headroom - (headroom - 1.05) * markets[city]['production_pressure'])
        for city, route in active.items()
    )
    group_limit = _new_production_target(supply_goal, old_products)
    if field_size <= 2 and active and not (contest >= .82 or distressed):
        group_limit = max(group_limit, min(int(demand * 2), int(cash / max(1., base_unit_cost))))
    if distressed:
        group_limit = min(group_limit, max(0, int(max(demand * .50, float(previous.get('sold', 0))) - old_products)))
    if not active or worker_rate <= 0 or engineer_rate <= 0:
        group_limit = 0
    if batch > max(1., demand * 2):
        batch = 1
    if 0 < group_limit < batch and not old_products and active and demand >= batch * .5:
        group_limit = batch
    budget = _fund(batch, cash, group_limit, build_budget)
    if budget.cost > cash + 1e-7:
        active = {city: Route(city, route.price, route.agents - route.added, 0, 0., route.demand, route.net_price)
                  for city, route in active.items() if route.agents > route.added}
        fixed = marketing_spend = ma_index = qi_index = stock_ma = 0.
        budget = build_budget(0)
    return active, budget, build_budget, (worker_wage, engineer_wage), batch, contest, distressed


def submit_ordinary_decisions(conn: sqlite3.Connection, round_no: int) -> int:
    """Write a deterministic group-cost plan with one shared, short sales forecast."""
    from .bots import _profitable_patent_budget

    bots = [dict(row) for row in conn.execute(
        'SELECT c.* FROM companies c WHERE c.is_bot=1 AND c.is_super_bot=0 AND NOT EXISTS '
        '(SELECT 1 FROM decisions d WHERE d.company_id=c.id AND d.round_no=?) ORDER BY c.id', (round_no,)
    )]
    if not bots:
        return 0
    official_round = max(1, round_no)
    configured = dict(conn.execute('SELECT key,value FROM settings'))
    settings = {key: (float(configured.get(key, default)) if isinstance(default, (float, int))
                      else configured.get(key, default)) for key, default in DEFAULT_SETTINGS.items()}
    group = _group(settings)
    markets = {str(row['city']): dict(row) for row in conn.execute('SELECT * FROM market_config ORDER BY city')}
    if not markets:
        return 0
    prior = conn.execute('SELECT MAX(round_no) FROM results WHERE round_no>=1 AND round_no<?', (round_no,)).fetchone()[0]
    previous = {int(row['company_id']): dict(row) for row in conn.execute(
        'SELECT r.*, d.worker_salary,d.engineer_salary FROM results r JOIN decisions d USING(company_id,round_no) '
        'WHERE r.round_no=?', (prior,)
    )}
    last_routes = {(int(row['company_id']), str(row['city'])): dict(row) for row in conn.execute(
        'SELECT company_id,city,sold,price,marketing FROM city_results WHERE round_no=?', (prior,)
    )}
    stats = defaultdict(list)
    for row in conn.execute(
        'SELECT city,market_size,player_total_volume,average_price FROM market_round_stats WHERE round_no IN '
        '(SELECT DISTINCT round_no FROM market_round_stats WHERE round_no>=1 AND round_no<? ORDER BY round_no DESC LIMIT 2) '
        'ORDER BY round_no DESC',
        (round_no,),
    ):
        stats[str(row['city'])].append(row)
    pool_totals = defaultdict(lambda: [0., 0., 0., 0])
    agents = {(int(row['company_id']), str(row['city'])): int(row['count']) for row in conn.execute('SELECT * FROM agents')}
    homes = Counter(str(row['home_city']) for row in conn.execute('SELECT home_city FROM companies'))
    traffic = Counter(city for (_, city), count in agents.items() if count > 0)
    for (company_id, city), route in last_routes.items():
        old = previous.get(company_id, {})
        pools = pool_totals[city]
        weight = float(route['sold'])
        pools[0] += weight * float(old.get('ma_index', 0.))
        pools[1] += weight * float(old.get('qi_index', 0.))
        pools[2] += weight * route['marketing'] * agent_mi_benefit(agents.get((company_id, city), 0))
        pools[3] += weight
    for city, market in markets.items():
        history = stats[city]
        stat = history[0] if history else None
        fills = [min(1., row['player_total_volume'] / max(1., row['market_size'])) for row in history]
        fill, older_fill = (fills + [0., 0.])[:2]
        size = market_size(market, round_no, float(settings['market_growth']))
        ma, qi, mi, weight = pool_totals[city]
        market.update(market_size=size, base_average_price=float(stat['average_price'] if stat else market['initial_avg_price']),
                      fill=fill, production_pressure=_market_pressure(fill, older_fill),
                      pool_indices=(ma / max(1., weight), qi / max(1., weight), mi / max(1., weight)), mi_active=mi > 0.)
        traffic[city] = max(traffic[city], homes[city])
    cohorts = defaultdict(list)
    for row in conn.execute('SELECT company_id,role,count,hire_round FROM employee_cohorts ORDER BY hire_round DESC,id DESC'):
        cohorts[(int(row['company_id']), str(row['role']))].append((int(row['count']), 1.1 if round_no - int(row['hire_round']) >= 2 else 1.))
    slots = {int(row['id']): slot for slot, row in enumerate(conn.execute(
        'SELECT id FROM companies WHERE is_bot=1 AND is_super_bot=0 ORDER BY id'
    ))}
    field_size = sum(homes.values())
    plans = {}
    players = []
    for bot in bots:
        key = int(bot['id'])
        plan = _build_plan(bot, slots[key], settings, group, markets, previous.get(key, {}), last_routes, agents, cohorts, traffic, field_size, official_round)
        plans[key] = plan
        routes, budget, *_ = plan
        old_stock = int(bot['product_inventory'])
        players.append({'company_id': key, 'available': old_stock + budget.units,
                        'ma_index': budget.ma / max(1, budget.headcount), 'qi_index': budget.qi / max(1., old_stock * 1.2 + budget.units),
                        'cities': {city: {'agents': routes[city].agents, 'marketing': routes[city].marketing, 'price': routes[city].price}
                                   if city in routes else {'agents': agents.get((key, city), 0), 'marketing': 0.,
                                                           'price': min(float(settings['price_max']), float(market['max_price']))}
                                   for city, market in markets.items()}})
    other_ids = set(previous) - plans.keys()
    other_routes = defaultdict(dict)
    for (key, city), route in last_routes.items():
        if key in other_ids and city in markets:
            other_routes[key][city] = {'agents': agents.get((key, city), 0), 'marketing': route['marketing'], 'price': route['price']}
    for key in sorted(other_routes):
        old = previous[key]
        players.append({'company_id': key, 'available': int(old['sold']) + int(old['inventory']),
                        'ma_index': old['ma_index'], 'qi_index': old['qi_index'], 'cities': other_routes[key]})
    predictions = forecast_market_sales(players=players, markets=list(markets.values()),
                                        ma_large_threshold=float(settings['cpi_ma_large_threshold']),
                                        price_power=int(settings['cpi_price_power']), max_iterations=1)['companies']
    submitted = now_iso()
    for bot in bots:
        key = int(bot['id'])
        routes, budget, build_budget, wages, batch, contest, distressed = plans[key]
        cash, stock = max(0., float(bot['cash'])), int(bot['product_inventory'])
        estimate = predictions[key]
        sold = estimate['sold_total']
        expected_revenue = sum(count * routes[city].net_price for city, count in estimate['sold_units'].items() if city in routes)
        capacity = estimate['visible_total']
        target = budget.units
        if budget.units and (field_size > 2 or contest >= .82 or stock) and sold < (stock + budget.units) * .90:
            times = 1.10 if contest < .78 and expected_revenue >= budget.cost else 1.02
            if expected_revenue < budget.cost:
                times = .92
            target = max(0, min(budget.units, math.ceil(max(sold, min(capacity, sold * 1.24)) * times - stock)))
        if budget.units and any(markets[city]['production_pressure'] > 0 for city in routes):
            target = min(target, _new_production_target(_production_sale_credit(estimate, routes, markets), stock))
        if target < budget.units:
            budget = _fund(batch, cash, target, build_budget)
        available = stock + budget.units
        revenue = expected_revenue * min(1., available / max(1., sold))
        interest = float(bot['debt']) * float(markets.get(str(bot['home_city']), next(iter(markets.values())))['interest_rate'])
        conservative_profit = .82 * revenue - budget.cost - interest
        research = 0.
        if not distressed and contest < .78:
            research = _profitable_patent_budget(
                balance=float(bot['research_balance']), remaining_rounds=int(settings['total_rounds']) - official_round,
                material_spend=budget.materials, sold=min(available, sold), available=available,
                projected_cash=max(0., cash + conservative_profit), reserve=max(float(settings['initial_cash']) * .35, budget.cost * .4),
                risk_profit=conservative_profit, amount_25=float(settings['research_25']), amount_75=float(settings['research_75']),
                hidden_cap=float(settings['research_probability_cap']), hidden_multiplier=float(settings['research_hidden_threshold_multiplier']),
                buffer=float(settings['research_buffer']), patent_factor=float(settings['patent_factor']), tax_rate=float(settings['tax_rate']),
            )
        conn.execute(
            'INSERT INTO decisions(company_id,round_no,loan_change,worker_delta,worker_salary,engineer_delta,'
            'engineer_salary,management_investment,production_volume,quality_investment,research_investment,submitted_at,is_draft) '
            'VALUES(?,?,0,?,?,?,?,?,?,?,?,?,0)',
            (key, round_no, budget.workers, wages[0], budget.engineers, wages[1], budget.ma, budget.units, budget.qi, research, submitted),
        )
        conn.executemany(
            'INSERT INTO city_decisions(company_id,round_no,city,agent_delta,marketing_investment,price,order_report) VALUES(?,?,?,?,?,?,0)',
            [(key, round_no, city, routes[city].added if available else 0,
              routes[city].marketing if available else 0., routes[city].price) if city in routes else
             (key, round_no, city, 0, 0., min(float(settings['price_max']), float(market['max_price']))) for city, market in markets.items()],
        )
    return len(bots)
