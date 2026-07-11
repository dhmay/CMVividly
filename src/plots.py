from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_values_over_time(df: pd.DataFrame):
    """Plot monthly values for each group."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for group, group_df in df.groupby("group"):
        ax.plot(group_df["month"], group_df["value"], marker="o", label=f"Group {group}")

    ax.set_title("Example observations over time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Value")
    ax.legend(title="Group")
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig, ax
