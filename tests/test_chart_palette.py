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


def test_wealth_chart_colors_stay_muted_without_fading_into_light_background():
    def channels(color):
        return tuple(channel / 255 for channel in bytes.fromhex(color[1:]))

    def luminance(rgb):
        linear = (v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb)
        return sum(weight * v for weight, v in zip((0.2126, 0.7152, 0.0722), linear))

    background = luminance(channels("#f4f5fa"))
    for color in WEALTH_COLORS:
        rgb = channels(color)
        assert rgb_to_hls(*rgb)[2] <= 0.50
        assert (background + 0.05) / (luminance(rgb) + 0.05) >= 2.0
