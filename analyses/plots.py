import pandas as pd
from matplotlib import pyplot as plt


def load_df(name):
    df = pd.read_csv(name, names=["n_samples", "accuracy", "number"])
    return df


adaptive = load_df("output/utility_adaptive.csv")
static = load_df("output/utility_static.csv")

stats_adaptive = adaptive.groupby("n_samples").agg(
    min=("accuracy", "min"), max=("accuracy", "max"),
)
stats_static = static.groupby("n_samples").agg(
    min=("accuracy", "min"), max=("accuracy", "max"),
)

plt.fill_between(
    stats_adaptive.index, stats_adaptive["min"], stats_adaptive["max"],
    alpha=0.5, label="Bayesian active learning",
)
plt.fill_between(
    stats_static.index, stats_static["min"], stats_static["max"], alpha=0.5,
    label="Random sample selection",
)
plt.xlabel("Number of trials")
plt.ylabel("Accuracy")
plt.legend(loc="upper center", ncol=2, frameon=False)
plt.savefig("output/active_vs_static.pdf", bbox_inches="tight")
plt.savefig("output/active_vs_static.png", bbox_inches="tight", dpi=144)
plt.show()
