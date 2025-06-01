from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt

# Define subject groups
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

# Set data directory
base_path = Path(r"O:\Tech_NeuroData\Code\New_1CH_AUHCologne\OUS_files\sleepstages_ear\m1m2_auhcologne_onechannel_lightning")
csv_files = sorted(base_path.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {base_path}")

channels = [f.stem for f in csv_files]
# For pretty x-tick labels: "ERA_ELA" etc.
def short_label(fullname):
    parts = fullname.split("_")
    if len(parts) >= 2:
        return parts[0] + "_" + parts[1]
    return fullname
derivation_labels = [short_label(ch) for ch in channels]

# 1. Load all CSVs, filter to binary, set index
dfs = {}
for ch, p in zip(channels, csv_files):
    df = pd.read_csv(p, usecols=["Subject", "Session", "Epoch", "PredictedStage"])
    df = df[df["PredictedStage"].isin([0, 2])].copy()
    df["Session"] = df["Session"].astype(str).str.zfill(2)
    df["Epoch"] = df["Epoch"].astype(int)
    df["StageGroup"] = df["PredictedStage"].map({0: 0, 2: 1})
    df.set_index(["Subject", "Session", "Epoch"], inplace=True)
    df = df.sort_index()
    dfs[ch] = df[["StageGroup"]].rename(columns={"StageGroup": ch})

# 2. Inner-join on full index to get only shared (Subject,Session,Epoch)
merged_all = None
for df in dfs.values():
    if merged_all is None:
        merged_all = df.copy()
    else:
        merged_all = merged_all.join(df, how="inner")

# Only use subjects with group label
valid_subjects = set(subject_group.keys())
merged_all = merged_all[merged_all.index.get_level_values("Subject").isin(valid_subjects)]

print(f"Number of rows after join: {merged_all.shape[0]}")
print("Subjects in joined data:", merged_all.index.get_level_values("Subject").unique().tolist())


# 3. Compute per-subject consensus-vs-heldout kappa, by group and derivation
kappa_data = []
for held in channels:
    peer_cols = [c for c in channels if c != held]
    for subj in valid_subjects:
        subj_idx = merged_all.index.get_level_values("Subject") == subj
        if not np.any(subj_idx):
            continue
        y_true = merged_all.loc[subj_idx, held].values
        votes = merged_all.loc[subj_idx, peer_cols].values
        pred = []
        for row in votes:
            v0 = np.sum(row == 0)
            v1 = np.sum(row == 1)
            pred.append(0 if v0 >= v1 else 1)
        kappa = cohen_kappa_score(y_true, pred, labels=[0, 1])
        kappa_data.append({
            "Subject": subj,
            "Group": subject_group[subj],
            "Derivation": held,
            "Kappa": kappa
        })

df_kappa = pd.DataFrame(kappa_data)

# 4. Compute median per (derivation, group)
medians = df_kappa.groupby(["Derivation", "Group"])["Kappa"].median().reset_index()

# 5. Print median kappa per group per derivation in table format
pivot_table = medians.pivot(index="Group", columns="Derivation", values="Kappa")
print(pivot_table.round(3))

# 6. Plot
derivation_order = channels
group_markers = {"HC": "o", "AD": "x", "DLB": "d"}
group_colors = {"HC": "tab:blue", "AD": "tab:orange", "DLB": "tab:green"}
group_offsets = {"HC": -0.20, "AD": 0.00, "DLB": 0.20}

fig, ax = plt.subplots(figsize=(max(12, len(derivation_order)*0.5), 6))
for group in group_markers:
    df_grp = medians[medians["Group"] == group]
    xs = [derivation_order.index(d) + group_offsets[group] for d in df_grp["Derivation"]]
    ys = df_grp["Kappa"].values
    ax.scatter(
        xs, ys,
        marker=group_markers[group],
        color=group_colors[group],
        s=120,
        label=group,
        edgecolor="black",
        zorder=3
    )

ax.set_xticks(range(len(derivation_order)))
ax.set_xticklabels([short_label(d) for d in derivation_order], rotation=90, fontsize=9)
ax.set_xlabel("Held-out derivation")
ax.set_ylabel("Median consensus $\kappa$")
ax.set_ylim(-0.2, 1.05)
ax.set_title("Median consensus $\kappa$ by ear derivation and group")
ax.legend(title="Group", loc="lower right", frameon=True)
plt.tight_layout()
plt.show()
