"""The original wealth-chart colors extended for large fields."""

from itertools import cycle


# Exact Streamlit light chart colors used before the custom scale was added.
_ORIGINAL_WEALTH_COLORS = (
    "#0068c9", "#83c9ff", "#ff2b2b", "#ffabab", "#29b09d",
    "#7defa1", "#ff8700", "#ffd16a", "#6d3fc0", "#d5dae5",
)

# Extend those blues, reds, teals, greens, oranges and purples with alternating
# light and richer tones. Hue families are interleaved instead of grouped.
WEALTH_COLORS = _ORIGINAL_WEALTH_COLORS + (
    "#2c7ed0", "#f8a088", "#46aea1", "#cf913e", "#9581e0", "#3cb4dd", "#b3648c", "#73c37f",
    "#d56622", "#60c7b5", "#a460a8", "#bcc376", "#8391d9", "#e67a94", "#59a96f",
    "#cda9f8", "#50a2c5", "#d681b3", "#64a954", "#e6862d", "#3c948f", "#d795d4", "#99871f",
    "#88b6f8", "#c8526b", "#78cca7", "#78b0f7", "#e05559", "#5fcaa1", "#a5731e",
    "#cd8641", "#58b5aa", "#b376c1", "#a8aa40", "#6883c5", "#f38693", "#429e5c", "#3d9be7",
    "#f7755a", "#349681", "#d9a54e", "#8972dc", "#49c7dd", "#b85a8e", "#7ace94",
    "#779fe7", "#db6887", "#4ebe8b", "#93b9f7", "#dc785b", "#54b89a", "#c4801e", "#b58bf0",
    "#3692ad", "#e48bb5", "#519e39", "#f89b73", "#1d8e88", "#e3a0d7", "#99a153",
    "#f79799", "#1f9079", "#ddb360", "#ac86d4", "#45afc7", "#c970a8", "#55bf71", "#b86c34",
    "#62bfbc", "#af6bc1", "#c9b24e", "#5a7aca", "#f7a1a3", "#51aa87", "#238fdc",
    "#1f8ba7", "#f59ebe", "#54ac74", "#e18653", "#3ea89b", "#d180ca", "#8c8738", "#92acf7",
    "#d4537b", "#7ec986", "#6da7f8", "#c36053", "#5bc3ad", "#bb7a23", "#afa6f7",
    "#74c9ca", "#c17eb4", "#b2a54d", "#6590dd", "#f67483", "#4c976d", "#6a96e3", "#ec7a75",
    "#3aab8a", "#cf961e", "#8c73c1", "#60bddd", "#c15ea0", "#8ac96a", "#bd6521",
    "#dd787b", "#67b37d", "#6ebdf8", "#cc5347", "#6accbe", "#ba9239", "#a693eb", "#20a1bb",
    "#e67bab", "#529859", "#e89556", "#219d8e", "#d69cee", "#867f21", "#a7b6f8",
    "#249f7c", "#f1a34c", "#8667c5", "#76c6e9", "#d27b9b", "#70b366", "#d7753e", "#3fbab9",
    "#a36aa4", "#b4b35e", "#5589df", "#f78cad", "#31935f", "#4b88cd", "#f68875",
    "#ea8fcf", "#3d9344", "#f3a969", "#4ea7a9", "#c288cd", "#a69633", "#839bf5", "#c26071",
    "#6fc497", "#54a5ed", "#e16766", "#2cbea8", "#a87723", "#ba9af1", "#239ab0",
)


def wealth_color_range(team_order, ranking_order):
    """Assign colors in roster order, then emit them in legend order."""
    by_team = dict(zip(team_order, cycle(WEALTH_COLORS)))
    return [by_team[team] for team in ranking_order]
