import re
from colorsys import rgb_to_hls

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


def test_wealth_chart_keeps_the_original_light_theme_colors_first():
    assert WEALTH_COLORS[:10] == (
        "#0068c9", "#83c9ff", "#ff2b2b", "#ffabab", "#29b09d",
        "#7defa1", "#ff8700", "#ffd16a", "#6d3fc0", "#d5dae5",
    )


def test_wealth_chart_extension_uses_colorful_midtones():
    for color in WEALTH_COLORS[10:]:
        rgb = (channel / 255 for channel in bytes.fromhex(color[1:]))
        _, lightness, saturation = rgb_to_hls(*rgb)
        assert 0.25 < lightness < 0.85
        assert 0.23 < saturation < 0.93
