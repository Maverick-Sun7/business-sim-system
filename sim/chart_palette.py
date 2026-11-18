"""Perceptually spaced team colors for the light wealth chart."""

from itertools import cycle


# Saturated, dark enough for the chart background, and ordered to spread out
# perceptually similar colors when a smaller match only uses the first entries.
WEALTH_COLORS = (
    "#0069d2", "#d23228", "#00826e", "#8c32af", "#be6900", "#374b1e", "#af236e", "#283278",
    "#ff00e1", "#5a001e", "#9687b4", "#00a500", "#0000e1", "#b45aff", "#691eff", "#873c1e",
    "#ff3c87", "#690069", "#5a5a87", "#87964b", "#786900", "#0096d2", "#5a00b4", "#c35aa5",
    "#6978ff", "#c300d2", "#007800", "#965a69", "#87003c", "#000096", "#ff4b00", "#c37878",
    "#005a69", "#5a2d00", "#e100a5", "#3c0069", "#a500ff", "#693c5a", "#2d3cb4", "#004bff",
    "#7869c3", "#4b9696", "#2d78a5", "#5a4bd2", "#0f2d4b", "#b4002d", "#e10069", "#5a871e",
    "#4b1e4b", "#5a3c96", "#3c693c", "#003c00", "#872d78", "#a578e1", "#874bf0", "#780096",
    "#d24bd2", "#005aa5", "#b42da5", "#3c0096", "#965a2d", "#e100ff", "#780fd2", "#780000",
    "#4b00e1", "#c34b69", "#96784b", "#ff003c", "#e15a5a", "#875a96", "#694b00", "#b48700",
    "#a55ac3", "#5a5aff", "#0078ff", "#005a00", "#ff2db4", "#3c9669", "#6987d2", "#3c4b78",
    "#960000", "#963c5a", "#0f1eb4", "#a56996", "#e1691e", "#691e4b", "#7878a5", "#2d0fff",
    "#b43c00", "#b45a4b", "#5a0087", "#a500c3", "#5a784b", "#e15a87", "#a54b87", "#4b5ab4",
    "#a50087", "#a53ce1", "#692d78", "#784b78", "#0f6987", "#693c2d", "#1e1e69", "#e100d2",
    "#004bd2", "#003c96", "#c300ff", "#4b2d78", "#e15ab4", "#4b1e0f", "#c33c87", "#b478b4",
    "#c369d2", "#00963c", "#8700ff", "#784ba5", "#008796", "#4b78e1", "#005a3c", "#3c2d96",
    "#4b5a00", "#692db4", "#960f5a", "#5a69a5", "#d24bff", "#4b3cf0", "#ff4b5a", "#78875a",
    "#9669ff", "#3c00c3", "#782d3c", "#4b87b4", "#e13c96", "#e14b0f", "#789600", "#b43c4b",
    "#785a2d", "#c30087", "#00874b", "#5a2d3c", "#874b4b", "#874bc3", "#5a003c", "#967800",
    "#c3784b", "#9600e1", "#c31e5a", "#963c96", "#1e1e87", "#2d2dd2", "#004b3c", "#0f3c5a",
    "#5a2dd2", "#00695a", "#e13c5a", "#9678c3", "#004b0f", "#8769e1", "#4b781e", "#0f691e",
)


def wealth_color_range(team_order, ranking_order):
    """Assign colors in roster order, then emit them in legend order."""
    by_team = dict(zip(team_order, cycle(WEALTH_COLORS)))
    return [by_team[team] for team in ranking_order]
