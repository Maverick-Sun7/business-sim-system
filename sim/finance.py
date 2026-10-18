"""One cash/debt ledger for the on-screen report and its PDF export."""
from __future__ import annotations

from typing import Any


def _amount(values: dict[str, Any], key: str, fallback: str | None = None) -> float:
    value = values.get(key)
    if value is None and fallback is not None:
        value = values.get(fallback)
    return float(value or 0.0)


def build_finance_ledger(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct actual payments, without concealing a stored balance mismatch.

    Bonuses are usable before production. Historical reports stored their
    opening balance excluding that bonus; new reports mark it as already
    included. Both become the same usable opening balance here. The final
    Bonus row is informational, so it never credits the cash a second time.
    """
    finance = report.get("finance") or {}
    metrics = report.get("key_metrics") or {}
    bonus = _amount(finance, "project_bonus")
    cash = _amount(finance, "round_begins")
    if not finance.get("bonus_in_round_begins", False):
        cash += bonus

    loan = _amount(finance, "loan_change")
    layoff_debt = _amount(finance, "layoff_debt")
    quit_debt = _amount(finance, "quit_penalty_debt")
    interest = _amount(finance, "interest")
    debt = float(finance.get(
        "starting_debt",
        _amount(metrics, "debt") - loan - layoff_debt - quit_debt - interest,
    ) or 0.0)

    rows: list[dict[str, Any]] = []

    def append(key: str, en: str, zh: str, cash_flow: float = 0.0, debt_change: float = 0.0) -> None:
        nonlocal cash, debt
        cash += cash_flow
        debt += debt_change
        rows.append({
            "key": key, "label_en": en, "label_zh": zh,
            "cash_flow": cash_flow, "cash": cash,
            "debt_change": debt_change, "debt": debt,
        })

    append("round_begins", "Round begins", "期初")
    append("loan_change", "Bank loan", "银行贷款", loan, loan)
    append("worker_wages", "Workers salary cost", "工人工资", -_amount(finance, "worker_wages", "wages"))
    append("engineer_wages", "Engineers salary cost", "工程师工资", -_amount(finance, "engineer_wages"))
    append("layoff", "Layoff compensation", "裁员费用", -_amount(finance, "layoff_cash", "layoff"), layoff_debt)
    append("quit_penalty", "Quit compensation", "离职补偿", -_amount(finance, "quit_penalty_cash", "quit_penalty"), quit_debt)
    append("training", "Employee training cost", "培训费用", -_amount(finance, "training"))
    append("component_material", "Components material cost", "零件材料", -_amount(finance, "component_material", "materials"))
    append("component_storage", "Components storage cost", "零件仓储", -_amount(finance, "component_storage", "storage"))
    append("product_material", "Products material cost", "产品材料", -_amount(finance, "product_material"))
    append("product_storage", "Products storage cost", "产品仓储", -_amount(finance, "product_storage"))
    append("agents", "Change sales agents cost", "Agent 变更", -_amount(finance, "agents"))
    append("marketing", "Marketing investment", "营销投入", -_amount(finance, "marketing"))
    append("quality", "Quality investment", "品质投入", -_amount(finance, "quality"))
    append("management", "Management investment", "管理投入", -_amount(finance, "management"))
    revenue = _amount(metrics, "sales_revenue") if "sales_revenue" in metrics else _amount(finance, "sales_revenue")
    append("sales_revenue", "Sales revenue", "销售收入", revenue)
    # Match settle_round: revenue, then research, transport, reports, and tax.
    append("research", "Research investment", "研发投入", -_amount(finance, "research"))
    append("transport", "Transportation cost", "跨城运输", -_amount(finance, "transport"))
    append("market_reports", "Market report cost", "市场报告", -_amount(finance, "market_reports"))
    append("interest", "Debt interest", "贷款利息", 0.0, interest)
    append("tax", "Tax deduction", "税费", -_amount(finance, "tax"))
    append("project_bonus", f"Project bonus ({bonus:,.0f}, included at round start)", f"奖励（¥{bonus:,.0f}，已计入期初可用现金）")
    append("round_ends", "Round ends", "期末")
    return rows


def asset_reconciliation(report: dict[str, Any]) -> dict[str, float]:
    """Expose signed cash/debt mismatches. Total assets always equal cash."""
    finance = report.get("finance") or {}
    metrics = report.get("key_metrics") or {}
    closing = build_finance_ledger(report)[-1]
    ending_cash = float(closing["cash"])
    ending_debt = float(closing["debt"])

    total_assets = float(metrics.get("total_assets", ending_cash))
    net_assets = float(metrics.get("net_assets", total_assets - ending_debt))
    return {
        "ending_cash": ending_cash,
        "ending_debt": ending_debt,
        "total_assets": total_assets,
        "net_assets": net_assets,
        "cash_difference": ending_cash - float(finance.get("round_ends", ending_cash)),
        "debt_difference": ending_debt - float(metrics.get("debt", ending_debt)),
        "asset_difference": total_assets - ending_cash,
        "net_asset_difference": net_assets - (ending_cash - ending_debt),
    }
