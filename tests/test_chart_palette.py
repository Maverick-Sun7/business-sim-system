import re

from sim.chart_palette import WEALTH_COLORS, wealth_color_range


def test_wealth_chart_uses_distinct_colors_for_large_matches():
    teams = [f"BOT{number:02d}" for number in range(160)]
    colors = wealth_color_range(teams, teams)
    assert len(WEALTH_COLORS) == len(colors) == len(set(colors)) == 160
    assert all(re.fullmatch(r"#[0-9a-f]{6}", color) for color in colors)


def test_wealth_chart_colors_do_not_change_with_ranking():
    teams = ["C01", "BOT01", "BOT02"]
    colors = wealth_color_range(teams, teams)
    assert wealth_color_range(teams, ["BOT02", "C01", "BOT01"]) == [colors[2], colors[0], colors[1]]
