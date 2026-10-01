from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE_ORDER = ["Stage II", "Stage III", "Stage IV"]
STAGE_LABELS = ["II", "III", "IV"]
MODULE_COLORS = {
    "Structural": "#1b9e77",
    "Signaling": "#7570b3",
    "Metabolic": "#e6ab02",
    "ECM": "#d95f02",
}
GENE_COLORS = {
    "ACTN1": "#d95f02",
    "ABI3BP": "#1b9e77",
    "ADAM33": "#7570b3",
}
SURVIVAL_COLORS = {
    "Low ACTN1": "#2c7fb8",
    "High ACTN1": "#d95f02",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 17,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "figure.titlesize": 20,
            "axes.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file was not found: {path}")
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="left",
    )


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def plot_stage_series(
    ax: plt.Axes,
    data: pd.DataFrame,
    names: list[str],
    colors: dict[str, str],
    ylabel: str,
    title: str,
) -> None:
    x_positions = np.arange(len(STAGE_ORDER))
    for name in names:
        subset = data[data["name"] == name].copy()
        if subset.empty:
            continue
        subset["stage"] = pd.Categorical(subset["stage"], STAGE_ORDER, ordered=True)
        subset = subset.sort_values("stage")
        means = [subset.loc[subset["stage"] == stage, "mean"].mean() for stage in STAGE_ORDER]
        errors = [subset.loc[subset["stage"] == stage, "ci95"].mean() for stage in STAGE_ORDER]
        ax.errorbar(
            x_positions,
            means,
            yerr=errors,
            marker="o",
            markersize=7,
            linewidth=2.2,
            capsize=5,
            label=name,
            color=colors.get(name, "#333333"),
        )

    ax.axhline(0, color="#999999", linewidth=1, linestyle="--", zorder=0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(STAGE_LABELS)
    ax.set_xlabel("Pathologic stage")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)


def make_figure3(data_dir: Path, output_dir: Path) -> None:
    data = read_csv(data_dir / "Figure3_module_activity_stage_summary.csv")
    for column in ["mean", "ci95", "n"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    module_data = data[data["type"] == "module"]
    gene_data = data[(data["type"] == "gene") & data["name"].isin(GENE_COLORS)]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    fig.suptitle("Stage-supported module activity in TCGA BLCA primary tumors", y=1.03)

    plot_stage_series(
        axes[0],
        module_data,
        list(MODULE_COLORS),
        MODULE_COLORS,
        "Mean z-scored module activity",
        "Module activity",
    )
    plot_stage_series(
        axes[1],
        gene_data,
        list(GENE_COLORS),
        GENE_COLORS,
        "Mean z-scored expression",
        "Transition-associated genes",
    )
    add_panel_label(axes[0], "(a)")
    add_panel_label(axes[1], "(b)")
    save_figure(fig, output_dir, "Figure3_Functional_Module_Activity_Evolution")


def kaplan_meier_curve(times: pd.Series, events: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    clean = pd.DataFrame({"time": times, "event": events}).dropna()
    clean = clean[clean["time"] >= 0].sort_values("time")
    if clean.empty:
        return np.array([0.0]), np.array([1.0])

    survival = 1.0
    x_values = [0.0]
    y_values = [1.0]
    event_times = np.sort(clean.loc[clean["event"] == 1, "time"].unique())

    for event_time in event_times:
        at_risk = int((clean["time"] >= event_time).sum())
        deaths = int(((clean["time"] == event_time) & (clean["event"] == 1)).sum())
        if at_risk <= 0:
            continue
        survival *= 1.0 - deaths / at_risk
        x_values.append(float(event_time))
        y_values.append(float(survival))

    max_time = float(clean["time"].max())
    if x_values[-1] < max_time:
        x_values.append(max_time)
        y_values.append(y_values[-1])

    return np.asarray(x_values), np.asarray(y_values)


def survival_at_times(curve_x: np.ndarray, curve_y: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(curve_x, query_times, side="right") - 1
    indices = np.clip(indices, 0, len(curve_y) - 1)
    return curve_y[indices]


def make_figure4(data_dir: Path, output_dir: Path) -> None:
    survival_data = read_csv(data_dir / "Figure4_ACTN1_survival_data.csv")
    stats = read_csv(data_dir / "Figure4_ACTN1_survival_statistics.csv").iloc[0]
    risk_table = read_csv(data_dir / "Figure4_ACTN1_risk_table.csv")

    required = {"OS_MONTHS_NUM", "OS_EVENT", "ACTN1_GROUP"}
    missing = required.difference(survival_data.columns)
    if missing:
        raise ValueError(f"Survival data are missing required columns: {sorted(missing)}")

    survival_data["OS_MONTHS_NUM"] = pd.to_numeric(survival_data["OS_MONTHS_NUM"], errors="coerce")
    survival_data["OS_EVENT"] = pd.to_numeric(survival_data["OS_EVENT"], errors="coerce")
    survival_data = survival_data.dropna(subset=["OS_MONTHS_NUM", "OS_EVENT", "ACTN1_GROUP"])

    fig = plt.figure(figsize=(10.8, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[4.7, 1.35])
    ax = fig.add_subplot(grid[0])
    table_ax = fig.add_subplot(grid[1])

    for group in ["Low ACTN1", "High ACTN1"]:
        group_data = survival_data[survival_data["ACTN1_GROUP"] == group]
        curve_x, curve_y = kaplan_meier_curve(group_data["OS_MONTHS_NUM"], group_data["OS_EVENT"])
        ax.step(
            curve_x,
            curve_y,
            where="post",
            linewidth=2.6,
            color=SURVIVAL_COLORS[group],
            label=f"{group} (n={len(group_data)})",
        )

        censored = group_data.loc[group_data["OS_EVENT"] == 0, "OS_MONTHS_NUM"].to_numpy(dtype=float)
        if censored.size:
            ax.plot(
                censored,
                survival_at_times(curve_x, curve_y, censored),
                linestyle="none",
                marker="|",
                markersize=9,
                markeredgewidth=1.5,
                color=SURVIVAL_COLORS[group],
                alpha=0.85,
            )

    hr = float(stats.get("univ_hr_high_vs_low", np.nan))
    ci_low = float(stats.get("univ_ci_low", np.nan))
    ci_high = float(stats.get("univ_ci_high", np.nan))
    logrank_p = float(stats.get("logrank_p", np.nan))
    high_5y = float(stats.get("high_5y", np.nan)) * 100.0
    low_5y = float(stats.get("low_5y", np.nan)) * 100.0

    annotation = (
        f"Log-rank P = {format_p(logrank_p)}\n"
        f"HR = {hr:.2f} (95% CI {ci_low:.2f}-{ci_high:.2f})\n"
        f"5-year OS: high {high_5y:.1f}%, low {low_5y:.1f}%"
    )
    ax.text(
        0.98,
        0.98,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"},
    )

    ax.set_title("ACTN1 expression and overall survival in TCGA BLCA", pad=10)
    ax.set_xlabel("Overall survival time (months)")
    ax.set_ylabel("Survival probability")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower left")
    add_panel_label(ax, "(a)")

    table_ax.axis("off")
    add_panel_label(table_ax, "(b)")
    month_columns = [column for column in risk_table.columns if column != "group"]
    display_columns = [column.replace("m", "") for column in month_columns]
    cell_text = risk_table[month_columns].astype(int).astype(str).values.tolist()
    row_labels = risk_table["group"].tolist()
    table = table_ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=display_columns,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12.5)
    table.scale(1.0, 1.45)
    table_ax.set_title("Number at risk by month", pad=12, fontsize=15)

    save_figure(fig, output_dir, "Figure4_ACTN1_Survival_Analysis")


def make_figure7(data_dir: Path, output_dir: Path) -> None:
    curves = read_csv(data_dir / "Figure7_stage_ordered_robustness_curves.csv")
    summary = read_csv(data_dir / "Figure7_robustness_summary.csv").iloc[0]
    for column in curves.columns:
        curves[column] = pd.to_numeric(curves[column], errors="coerce")

    x_values = curves["center_index"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    fig.suptitle("Stage-ordered robustness checks for early-warning signals", y=1.03)

    axes[0].fill_between(
        x_values,
        curves["null_variance_low"],
        curves["null_variance_high"],
        color="#cccccc",
        alpha=0.5,
        label="Permutation interval",
    )
    axes[0].plot(x_values, curves["observed_variance"], color="#1b9e77", linewidth=2.5, label="Observed")
    axes[0].set_title("Windowed variance")
    axes[0].set_xlabel("Stage-ordered sample index")
    axes[0].set_ylabel("Variance")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    axes[0].text(
        0.04,
        0.95,
        f"Permutation P = {format_p(float(summary.get('variance_permutation_p', np.nan)))}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=13,
    )

    axes[1].fill_between(
        x_values,
        curves["null_ordered_ac1_low"],
        curves["null_ordered_ac1_high"],
        color="#cccccc",
        alpha=0.5,
        label="Permutation interval",
    )
    axes[1].plot(x_values, curves["observed_ordered_ac1"], color="#7570b3", linewidth=2.5, label="Observed")
    axes[1].axhline(0, color="#999999", linewidth=1, linestyle="--")
    axes[1].set_title("Ordered lag-1 autocorrelation")
    axes[1].set_xlabel("Stage-ordered sample index")
    axes[1].set_ylabel("Ordered AC1")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)
    axes[1].text(
        0.04,
        0.95,
        f"Permutation P = {format_p(float(summary.get('ordered_ac1_permutation_p', np.nan)))}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=13,
    )

    add_panel_label(axes[0], "(a)")
    add_panel_label(axes[1], "(b)")
    save_figure(fig, output_dir, "Figure7_Early_Warning_Signals")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Reproduce revised TID-framework figures.")
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data", help="Input CSV directory.")
    parser.add_argument("--figures-dir", type=Path, default=repo_root / "figures", help="Output figure directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    make_figure3(args.data_dir, args.figures_dir)
    make_figure4(args.data_dir, args.figures_dir)
    make_figure7(args.data_dir, args.figures_dir)
    print(f"Figures written to: {args.figures_dir.resolve()}")


if __name__ == "__main__":
    main()
