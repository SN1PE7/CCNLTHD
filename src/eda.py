from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("dataset/housing.csv")
OUTPUT_DIR = Path("output")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    OUTPUT_DIR.mkdir(exist_ok=True)

    numeric_columns = [
        "median_house_value",
        "median_income",
        "housing_median_age",
        "total_rooms",
        "population",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for index, column in enumerate(numeric_columns):
        df[column].dropna().hist(ax=axes[index], bins=30, color="#2F6B9A", edgecolor="white")
        axes[index].set_title(f"Histogram: {column}")
        axes[index].set_xlabel(column)
        axes[index].set_ylabel("Frequency")

    axes[-1].axis("off")
    fig.suptitle("Housing Dataset EDA", fontsize=16)
    fig.tight_layout()
    histogram_path = OUTPUT_DIR / "eda_histograms.png"
    fig.savefig(histogram_path, dpi=150)
    plt.close(fig)

    scatter_fig, scatter_ax = plt.subplots(figsize=(8, 6))
    scatter = scatter_ax.scatter(
        df["median_income"],
        df["median_house_value"],
        alpha=0.25,
        s=12,
        c=df["latitude"],
        cmap="viridis",
    )
    scatter_ax.set_title("Median Income vs House Value")
    scatter_ax.set_xlabel("median_income")
    scatter_ax.set_ylabel("median_house_value")
    scatter_fig.colorbar(scatter, ax=scatter_ax, label="latitude")
    scatter_fig.tight_layout()
    scatter_path = OUTPUT_DIR / "eda_scatter_income_value.png"
    scatter_fig.savefig(scatter_path, dpi=150)
    plt.close(scatter_fig)

    # EDA-style plot to show that standardization changes scale but keeps correlation pattern.
    pair_df = df[["median_income", "median_house_value"]].dropna().copy()
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(pair_df)
    pair_df["median_income_z"] = scaled_values[:, 0]
    pair_df["median_house_value_z"] = scaled_values[:, 1]

    raw_corr = float(np.corrcoef(pair_df["median_income"], pair_df["median_house_value"])[0, 1])
    scaled_corr = float(
        np.corrcoef(pair_df["median_income_z"], pair_df["median_house_value_z"])[0, 1]
    )

    corr_fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(pair_df["median_income"], pair_df["median_house_value"], alpha=0.22, s=10)
    axes[0].set_title(f"Before Standardization (r = {raw_corr:.3f})")
    axes[0].set_xlabel("median_income")
    axes[0].set_ylabel("median_house_value")

    axes[1].scatter(pair_df["median_income_z"], pair_df["median_house_value_z"], alpha=0.22, s=10)
    axes[1].set_title(f"After Standardization (r = {scaled_corr:.3f})")
    axes[1].set_xlabel("z-score(median_income)")
    axes[1].set_ylabel("z-score(median_house_value)")

    corr_fig.suptitle("Correlation Pattern Before vs After Standardization", fontsize=14)
    corr_fig.tight_layout()
    corr_plot_path = OUTPUT_DIR / "eda_corr_before_after_standardization.png"
    corr_fig.savefig(corr_plot_path, dpi=150)
    plt.close(corr_fig)

    print(f"Saved histogram grid to: {histogram_path}")
    print(f"Saved scatter plot to: {scatter_path}")
    print(f"Saved correlation standardization plot to: {corr_plot_path}")


if __name__ == "__main__":
    main()