"""Muted cream-tone team colors for the light wealth chart."""

from itertools import cycle


# Blue and teal anchor the palette; sage, apricot, dusty rose and lilac add
# separation without neon accents. Each 16-color band mixes hue families and
# varies lightness, so small matches get a balanced mix too.
WEALTH_COLORS = (
    "#598daf", "#b17a64", "#639a7a", "#9d77a1", "#a18552", "#56999a", "#af737a", "#7987b8",
    "#829460", "#ab738a", "#5594a2", "#9780b1", "#908851", "#58978b", "#ae815c", "#6589b4",
    "#b38eb0", "#b09c70", "#75acb1", "#c18c8c", "#949bc7", "#94aa7f", "#be8b9a", "#75a7b8",
    "#af96bf", "#a09f70", "#74aba4", "#bd9a76", "#849dc4", "#79a1c1", "#c1937b", "#7bae96",
    "#92656e", "#66769a", "#757e56", "#8e667a", "#4f7f87", "#7e7197", "#7e744b", "#548174",
    "#957057", "#577795", "#4f7a90", "#956a5e", "#5f8368", "#82698b", "#8b714f", "#528380",
    "#b17e99", "#619fa6", "#9b8cbd", "#9f8f5c", "#68a08e", "#b98a6c", "#6a95ba", "#6099b4",
    "#ba8375", "#76a27e", "#a183af", "#ae8c61", "#65a39e", "#b67e8b", "#7c93c0", "#939b68",
    "#aaa77f", "#85b2aa", "#c4a387", "#8ea7c7", "#86aac4", "#c69d8b", "#8db59e", "#b799b8",
    "#b8a580", "#86b4b6", "#c5979a", "#9ca5cb", "#a1b18b", "#c297a6", "#84b0bb", "#b4a1c5",
    "#607dae", "#5181ab", "#a77151", "#4c9277", "#996a93", "#937d42", "#448f97", "#a96868",
    "#757ab0", "#6f8d58", "#a66779", "#48899f", "#9473a5", "#808144", "#448e88", "#a37949",
    "#84a58f", "#a48da8", "#a99677", "#7ca4a4", "#b18a8f", "#8d98b7", "#96a080", "#ae8a99",
    "#7aa0a8", "#a294b4", "#9d9775", "#7ca298", "#b2957e", "#8098b3", "#7a9bb0", "#b38f82",
    "#68928d", "#9e7680", "#7586a5", "#878c69", "#9a778a", "#658e92", "#8b82a4", "#8e8260",
    "#6a8f81", "#a2806d", "#68879f", "#62899b", "#a27b72", "#749177", "#8e7a99", "#998164",
    "#9fac73", "#c58ba7", "#6aafba", "#ad9acf", "#aea065", "#71b1a0", "#cc9874", "#78a4cf",
    "#6ba8c9", "#ce917e", "#7eb48e", "#b490c0", "#bf9c69", "#6db3b0", "#cb8b97", "#8ca1d4",
    "#a188ab", "#8e9168", "#6a9b97", "#aa8d6c", "#7b8eb1", "#7291af", "#ad876f", "#709f8c",
    "#a3819d", "#9d8f67", "#6c9ca3", "#ae807d", "#8a8cb3", "#849b77", "#ad7f89", "#6d97a8",
)


def wealth_color_range(team_order, ranking_order):
    """Assign colors in roster order, then emit them in legend order."""
    by_team = dict(zip(team_order, cycle(WEALTH_COLORS)))
    return [by_team[team] for team in ranking_order]
