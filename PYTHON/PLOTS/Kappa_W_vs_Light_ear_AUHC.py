# Description:
# Computes pairwise Cohen's kappa per subject (using all sessions/epochs per subject)
# Assigns each subject to a clinical group (HC, AD, DLB)
# Plots groupwise distributions as horizontal boxplots with overlayed datapoints
# Uses all CSV files in O:/Tech_NeuroData/Code/New_1CH_AUHCologne/OUS_files/sleepstages_ear/m1m2_auhcologne_onechannel_lightning

import itertools
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from matplotlib.lines import Line2D

# ──────────────────────────────────────────────────────────────────────────────
# 0) Cosmetic rcParams
# ──────────────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 16,
    'axes.unicode_minus': False,
})
sns.set_theme(style="whitegrid", font="serif")

# ──────────────────────────────────────────────────────────────────────────────
# 1) File scan (Ear AUHC Lightning dataset)
# ──────────────────────────────────────────────────────────────────────────────
ear_dir = Path(
    r"O:\Tech_NeuroData\Code\New_1CH_AUHCologne\OUS_files\sleepstages_ear\m1m2_auhcologne_onechannel_lightning"
)
files = sorted(ear_dir.glob("*.csv"))
if not files:
    raise FileNotFoundError("No CSV files found in the specified directory.")

labels = [fp.stem for fp in files]

# ──────────────────────────────────────────────────────────────────────────────
# 2) Load & filter to wake (0) vs light sleep (2), map to binary StageGroup
# ──────────────────────────────────────────────────────────────────────────────
data = {}
for lbl, fp in zip(labels, files):
    df = pd.read_csv(fp)
    df = df[df["PredictedStage"].isin([0, 2])].copy()
    df["Session"] = df["Session"].astype(str).str.zfill(2)
    df["Epoch"] = df["Epoch"].astype(int)
    df["StageGroup"] = df["PredictedStage"].map({0: 0, 2: 1})
    df.sort_values(["Subject", "Session", "Epoch"], inplace=True)
    data[lbl] = df

# ──────────────────────────────────────────────────────────────────────────────
# 3) Hardcoded Subject-to-Group Mapping (same as scalp)
# ──────────────────────────────────────────────────────────────────────────────
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
subject_to_group = {}
for subject in hc_subjects:
    subject_to_group[subject] = "HC"
for subject in ad_subjects:
    subject_to_group[subject] = "AD"
for subject in dlb_subjects:
    subject_to_group[subject] = "DLB"

# ──────────────────────────────────────────────────────────────────────────────
# 4) Compute pairwise Cohen’s κ PER SUBJECT, assign group
# ──────────────────────────────────────────────────────────────────────────────
expanded_pairwise = []
for a, b in itertools.combinations(labels, 2):
    merged = pd.merge(
        data[a], data[b],
        on=["Subject", "Session", "Epoch"],
        suffixes=("_a", "_b")
    )
    for subject_id, subject_df in merged.groupby("Subject"):
        group = subject_to_group.get(subject_id, None)
        if group is None:
            continue
        kappa = cohen_kappa_score(
            subject_df["StageGroup_a"],
            subject_df["StageGroup_b"],
            labels=[0, 1]
        )
        expanded_pairwise.append({
            "Pair": f"{a} vs {b}",
            "Kappa": kappa,
            "Subject": subject_id,
            "Group": group
        })
df_k_grouped = pd.DataFrame(expanded_pairwise)

# ──────────────────────────────────────────────────────────────────────────────
# 5) Plot: Boxplots of pairwise κ by patient group (HC, AD, DLB)
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

palette = {
    "HC": sns.color_palette("pastel")[2],
    "AD": sns.color_palette("pastel")[1],
    "DLB": sns.color_palette("pastel")[0]
}

# Horizontal boxplots by group
sns.boxplot(
    x="Kappa",
    y="Group",
    data=df_k_grouped,
    order=["HC", "AD", "DLB"],
    ax=ax,
    width=0.6,
    showcaps=True,
    showfliers=False,
    boxprops=dict(edgecolor="black", linewidth=1.25, alpha=1),
    whiskerprops=dict(color="black", linewidth=2.5),
    capprops=dict(color="black", linewidth=2.5),
    medianprops=dict(color="black", linewidth=3),
    palette=palette,
    orient='h'
)

# Overlay horizontal stripplot for all points
sns.stripplot(
    x="Kappa",
    y="Group",
    data=df_k_grouped,
    order=["HC", "AD", "DLB"],
    ax=ax,
    size=6,
    jitter=0.07,
    palette=palette,
    edgecolor="gray",
    linewidth=1,
    alpha=1.0,
    orient='h',
    dodge=False
)

# Titles, labels, grid, legend
ax.set_xlabel(r"$\kappa$", fontsize=30, family="serif")
ax.set_ylabel("")  # No y-label at all
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20, left=True, labelleft=True)
ax.yaxis.grid(False)
ax.xaxis.grid(True, color='gray', linestyle='-', linewidth=1, which='major')

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

# Add median for each group in the legend
legend_elems = [
    Line2D([0], [0], color=palette["HC"], linewidth=5, linestyle="-",
           label=rf"$\tilde{{\kappa}}$ = {df_k_grouped.loc[df_k_grouped['Group']=='HC','Kappa'].median():.2f}"),
    Line2D([0], [0], color=palette["AD"], linewidth=5, linestyle="-",
           label=rf"$\tilde{{\kappa}}$ = {df_k_grouped.loc[df_k_grouped['Group']=='AD','Kappa'].median():.2f}"),
    Line2D([0], [0], color=palette["DLB"], linewidth=5, linestyle="-",
           label=rf"$\tilde{{\kappa}}$ = {df_k_grouped.loc[df_k_grouped['Group']=='DLB','Kappa'].median():.2f}")
]

ax.legend(handles=legend_elems, loc="lower left", frameon=True, fontsize=16, handlelength=0.25)

plt.tight_layout()
plt.show()

# Print the number of datapoints for each group
group_counts = df_k_grouped["Group"].value_counts().reindex(["HC", "AD", "DLB"], fill_value=0)
print("Number of datapoints per group:")
for group, count in group_counts.items():
    print(f"{group}: {count}")
