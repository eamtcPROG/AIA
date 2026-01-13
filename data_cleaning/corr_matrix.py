"""
Correlation Matrix Visualization
===============================

Loads the ML dataset from `cleared_data/ml_dataset.csv`, computes the correlation
matrix over numeric columns, and displays it as a heatmap.

Run:
  python3 data_cleaning/corr_matrix.py

Optional:
  python3 data_cleaning/corr_matrix.py --max-cols 40 --method spearman --save cleared_data/plots/corr_matrix.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    # This file lives at <repo>/data_cleaning/corr_matrix.py
    return Path(__file__).resolve().parents[1]


def _load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at: {csv_path}\n"
            f"Tip: create it first (or pass --csv to point to it)."
        )
    return pd.read_csv(csv_path)


def _auto_figsize(n_cols: int) -> tuple[float, float]:
    # Large matrices need more space; cap size so it doesn't get ridiculous.
    side = min(40.0, max(10.0, n_cols * 0.35))
    return (side, side)


def _plot_corr_heatmap(
    corr: pd.DataFrame,
    *,
    title: str,
    annot: bool,
    save_path: Path | None,
    show: bool,
) -> None:
    # Prefer seaborn if available; fall back to pure matplotlib if not.
    import matplotlib.pyplot as plt

    figsize = _auto_figsize(len(corr.columns))
    fig, ax = plt.subplots(figsize=figsize)

    try:
        import seaborn as sns  # type: ignore

        sns.heatmap(
            corr,
            ax=ax,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.25,
            linecolor="white",
            cbar_kws={"shrink": 0.8},
            annot=annot,
            fmt=".1f",
            annot_kws={"size": 6} if annot else None,
        )
    except Exception:
        # Matplotlib fallback (no seaborn dependency)
        import numpy as np

        data = corr.to_numpy(dtype=float)
        # Make NaNs transparent (e.g., constant columns produce NaNs)
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad(color=(1, 1, 1, 0))
        im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, shrink=0.8)

        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(list(corr.columns))
        ax.set_yticklabels(list(corr.index))

        if annot:
            # Write values into cells (e.g. 0.2). Choose text color for contrast.
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    val = data[i, j]
                    if np.isnan(val):
                        continue
                    text_color = "white" if abs(val) > 0.6 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.1f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color=text_color,
                    )

    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelrotation=0, labelsize=7)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Display correlation matrix heatmap for ml_dataset.csv")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to the dataset CSV (default: <repo>/cleared_data/ml_dataset.csv).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pearson",
        choices=["pearson", "spearman", "kendall"],
        help="Correlation method (default: pearson).",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=10,
        help=(
            "How many numeric features to include (default: 10). "
            "If match_outcome exists, selects match_outcome + the most-correlated features."
        ),
    )
    parser.add_argument(
        "--max-cols",
        type=int,
        default=None,
        help="Back-compat: same as --n-features, but selects the first N numeric columns.",
    )
    annot_group = parser.add_mutually_exclusive_group()
    annot_group.add_argument(
        "--annot",
        dest="annot",
        action="store_true",
        help="Annotate heatmap cells with correlation values (default).",
    )
    annot_group.add_argument(
        "--no-annot",
        dest="annot",
        action="store_false",
        help="Do not annotate heatmap cells.",
    )
    parser.set_defaults(annot=True)
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the plot (e.g. cleared_data/plots/corr_matrix.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Compute/render but don't open a window (useful when only saving).",
    )
    args = parser.parse_args()

    # In restricted environments (e.g. sandbox), Matplotlib cannot write to ~/.matplotlib.
    # Keep its config/cache inside the repo so plotting works reliably.
    mpl_config_dir = _repo_root() / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    default_csv = _repo_root() / "cleared_data" / "ml_dataset.csv"
    csv_path = Path(args.csv).expanduser() if args.csv else default_csv
    save_path = Path(args.save).expanduser() if args.save else None

    df = _load_dataset(csv_path)
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] == 0:
        raise ValueError("No numeric columns found in the dataset; cannot compute correlations.")

    # Default: select only 10 features to keep the plot readable.
    # If match_outcome exists, pick match_outcome + the most-correlated numeric features.
    n_features = args.n_features
    if args.max_cols is not None and args.max_cols > 0:
        n_features = args.max_cols

    if n_features is not None and n_features > 0 and numeric.shape[1] > n_features:
        if "match_outcome" in numeric.columns:
            target = numeric["match_outcome"]
            corrs = numeric.drop(columns=["match_outcome"], errors="ignore").corrwith(target).abs()
            top_cols = corrs.sort_values(ascending=False).head(max(0, n_features - 1)).index.tolist()
            numeric = numeric.loc[:, ["match_outcome", *top_cols]]
        else:
            numeric = numeric.iloc[:, :n_features]

    corr = numeric.corr(method=args.method)

    title = f"Correlation matrix ({args.method}) — {numeric.shape[1]} selected features"

    # For headless runs (e.g. CI / just saving), avoid GUI backends.
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    _plot_corr_heatmap(
        corr,
        title=title,
        annot=bool(args.annot),
        save_path=save_path,
        show=not bool(args.no_show),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
