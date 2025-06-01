from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt

# Define groups
hc_subjects = {
    "sub-142", "sub-152", "sub-260", "sub-325", "sub-400", "sub-422", "sub-562",
    "sub-777", "sub-817", "sub-843", "sub-869", "sub-906", "sub-916", "sub-957"
}

ad_subjects = {
    "sub-98", "sub-263", "sub-298", "sub-360", "sub-443", "sub-462", "sub-538",
    "sub-549", "sub-582", "sub-689", "sub-749", "sub-826", "sub-852", "sub-856",
    "sub-913", "sub-958", "sub-962", "sub-971", "sub-800"
}

dlb_subjects = {
    "sub-398", "sub-485", "sub-815", "sub-849", "sub-959", "sub-3127",
    "sub-3422", "sub-3906", "sub-3958", "sub-3971"
}

subject_group = {}
for sub in hc_subjects:
    subject_group[sub] = "HC"
for sub in ad_subjects:
    subject_group[sub] = "AD"
for sub in dlb_subjects:
    subject_group[sub] = "DLB"

# Load CSV files
base_path = Path(r"O:\Tech_NeuroData\Code\New_1CH_AUHCologne\OUS_files\sleepstages_scalp")
csv_files = sorted(base_path.glob("*.csv"))

channels = []
paths = []
for file in csv_files:
    channel = file.name.split("_")[0]
    channels.append(channel)
    paths.append(file)

dfs = {}
for ch, p in zip(channels, paths):
    df = pd.read_csv(p, usecols=["Subject", "Session", "Epoch", "PredictedStage"])
    df = df[df["PredictedStage"].isin([0, 2])].copy()
    df["Session"] = df["Session"].astype(str).str.zfill(2)
    df["Epoch"] = df["Epoch"].astype(int)
    df["StageGroup"] = df["PredictedStage"].map({0: 0, 2: 1})
    df.set_index(["Subject", "Session", "Epoch"], inplace=True)
    df = df.sort_index()
    dfs[ch] = df[["StageGroup"]].rename(columns={"StageGroup": f"SG_{ch}"})

all_subjects = set(subject_group.keys())

# Calculate per-subject consensus kappa for each derivation
kappa_data = []
for held in channels:
    held_df = dfs[held].copy()
    other_channels = [c for c in channels if c != held]
    merged = held_df.copy()
    for c in other_channels:
        merged = merged.join(dfs[c], how="inner")
    merged = merged.loc[merged.index.get_level_values("Subject").isin(all_subjects)]
    subjects_in_merged = merged.index.get_level_values("Subject").unique()
    for subj in subjects_in_merged:
        idx = merged.index.get_level_values("Subject") == subj
        if not np.any(idx):
            continue
        y_true = merged.loc[idx, f"SG_{held}"].values
        votes = merged.loc[idx, [f"SG_{c}" for c in other_channels]].values
        pred = []
        for row in votes:
            vote_count = {0: 0, 1: 0}
            for v in row:
                vote_count[v] += 1
            pred.append(0 if vote_count[0] >= vote_count[1] else 1)
        kappa = cohen_kappa_score(y_true, pred, labels=[0, 1])
        kappa_data.append({
            "Subject": subj,
            "Group": subject_group.get(subj, "Other"),
            "Derivation": held,
            "Kappa": kappa
        })

df_kappa = pd.DataFrame(kappa_data)

# Compute median for each (derivation, group)
medians = df_kappa.groupby(["Derivation", "Group"])["Kappa"].median().reset_index()

derivation_order = channels
group_markers = {"HC": "o", "AD": "x", "DLB": "d"}
group_colors = {"HC": "tab:blue", "AD": "tab:orange", "DLB": "tab:green"}

# Define x-offsets for each group so the points don't overlap
group_offsets = {"HC": -0.10, "AD": 0.00, "DLB": 0.10}
derivation_order = channels  # Use your original channel order


fig, ax = plt.subplots(figsize=(8, 5))

for group in group_markers:
    df_grp = medians[medians["Group"] == group]
    # For each row in group, plot at derivation x-location plus group offset
    xs = [derivation_order.index(d) + group_offsets[group] for d in df_grp["Derivation"]]
    ys = df_grp["Kappa"].values
    ax.scatter(
        xs,
        ys,
        marker=group_markers[group],
        color=group_colors[group],
        s=150,
        label=group,
        edgecolor="black",
        zorder=3
    )

print(medians.pivot(index="Group", columns="Derivation", values="Kappa").round(3))


ax.set_xticks(range(len(derivation_order)))
ax.set_xticklabels(derivation_order, rotation=25)
ax.set_xlabel("Held-out derivation")
ax.set_ylabel("Median consensus $\kappa$")
ax.set_ylim(-0.2, 1.05)
ax.set_title("Median consensus $\kappa$ by derivation and group")
ax.legend(title="Group", loc="lower right", frameon=True)
plt.tight_layout()
plt.show()