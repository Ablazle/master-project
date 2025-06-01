import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from matplotlib.lines import Line2D

# ──────────────────────────────────────────────────────────────────────────────
# Cosmetic settings
mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 18,
    "axes.unicode_minus": False,
})
sns.set_theme(style="whitegrid", font="serif")

# ──────────────────────────────────────────────────────────────────────────────
# File paths and subject-to-group mapping
scalp_csv = r"O:\Tech_NeuroData\Code\New_1CH_AUHCologne\OUS_files\Majority_scalp\majority_voted_scalp_sleep_stages.csv"
ear_csv   = r"O:\Tech_NeuroData\Code\New_1CH_AUHCologne\OUS_files\Majority_ear\majority_voted_ear_sleep_stages.csv"

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
# Data loading and preprocessing functions
def preprocess_majority_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["MajStage"].isin([0, 2])].copy()
    df["Epoch"] = df["Epoch"].astype(int)
    df["Session"] = df["Session"].astype(str)
    df["Subject"] = df["Subject"].astype(str)
    return df

scalp_df = preprocess_majority_df(pd.read_csv(scalp_csv))
ear_df   = preprocess_majority_df(pd.read_csv(ear_csv))

# ──────────────────────────────────────────────────────────────────────────────
# Compute Cohen's kappa
session_kappas = []
scalp_pairs = set(zip(scalp_df["Subject"], scalp_df["Session"]))
ear_pairs = set(zip(ear_df["Subject"], ear_df["Session"]))
common_pairs = scalp_pairs & ear_pairs

for subject, session in sorted(common_pairs):
    scalp_ss = scalp_df[(scalp_df["Subject"] == subject) & (scalp_df["Session"] == session)]
    ear_ss   = ear_df[(ear_df["Subject"] == subject) & (ear_df["Session"] == session)]
    merged = pd.merge(
        scalp_ss, ear_ss,
        on=["Subject", "Session", "Epoch"],
        suffixes=("_scalp", "_ear")
    )
    if merged.empty:
        continue
    kappa = cohen_kappa_score(
        merged["MajStage_scalp"],
        merged["MajStage_ear"],
        labels=[0, 2]
    )
    group = subject_to_group.get(subject, None)
    session_kappas.append({
        "Subject": subject,
        "Session": session,
        "Kappa": kappa,
        "Group": group
    })

df_kappa = pd.DataFrame(session_kappas)
df_kappa = df_kappa[df_kappa["Group"].notnull()]

# ──────────────────────────────────────────────────────────────────────────────
# Compute Cohen's kappa (all epochs/sessions for each subject)
overall_kappas = []
for subject in sorted(set(scalp_df["Subject"]) & set(ear_df["Subject"])):
    scalp_subj = scalp_df[scalp_df["Subject"] == subject]
    ear_subj = ear_df[ear_df["Subject"] == subject]
    merged_subj = pd.merge(
        scalp_subj, ear_subj,
        on=["Subject", "Session", "Epoch"],
        suffixes=("_scalp", "_ear")
    )
    if merged_subj.empty:
        continue
    kappa = cohen_kappa_score(
        merged_subj["MajStage_scalp"],
        merged_subj["MajStage_ear"],
        labels=[0, 2]
    )
    overall_kappas.append({
        "Subject": subject,
        "Session": "All",
        "Kappa": kappa,
        "Group": "Overall"
    })

df_overall = pd.DataFrame(overall_kappas)

# ──────────────────────────────────────────────────────────────────────────────
# Concatenate for plotting
df_kappa_plot = pd.concat([df_kappa, df_overall], ignore_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# Plot: Four boxplots (HC, AD, DLB, Overall)
palette = {
    "HC": sns.color_palette("pastel")[2],
    "AD": sns.color_palette("pastel")[1],
    "DLB": sns.color_palette("pastel")[0],
    "Overall": sns.color_palette("pastel")[3]
}
order = ["HC", "AD", "DLB", "Overall"]

fig, ax = plt.subplots(figsize=(9, 6))

sns.boxplot(
    x="Kappa",
    y="Group",
    data=df_kappa_plot,
    order=order,
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

sns.stripplot(
    x="Kappa",
    y="Group",
    data=df_kappa_plot,
    order=order,
    ax=ax,
    size=7,
    jitter=0.05,
    palette=palette,
    edgecolor="gray",
    linewidth=1.5,
    alpha=0.9,
    orient='h',
    dodge=False
)

ax.set_xlabel(r"$\kappa$", fontsize=22, family="serif")
ax.set_ylabel("")
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16, left=True, labelleft=True)
ax.yaxis.grid(False)
ax.xaxis.grid(True, color='gray', linestyle='-', linewidth=1, which='major')

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

# Median legend for each group
legend_elems = [
    Line2D([0], [0], color=palette["HC"], linewidth=5, linestyle="-",
           label=rf"$\tilde{{\kappa}}$ HC = {df_kappa.loc[df_kappa['Group']=='HC','Kappa'].median():.2f}"),
    Line2D([0], [0], color=palette["AD"], linewidth=5, linestyle="-",
           label=rf"$\tilde{{\kappa}}$ AD = {df_kappa.loc[df_kappa['Group']=='AD','Kappa'].median():.2f}"),
    Line2D([0], [0], color=palette["DLB"], linewidth=5, linestyle="-",
           label=rf"$\tilde{{\kappa}}$ DLB = {df_kappa.loc[df_kappa['Group']=='DLB','Kappa'].median():.2f}"),
    Line2D([0], [0], color=palette["Overall"], linewidth=5, linestyle="-",
           label=rf"$\tilde{{\kappa}}$ Overall = {df_overall['Kappa'].median():.2f}")
]
ax.legend(handles=legend_elems, loc="lower left", frameon=True, fontsize=14, handlelength=0.8)

plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# Print session count per group and all computed kappas for reporting
group_counts = df_kappa["Group"].value_counts().reindex(["HC", "AD", "DLB"], fill_value=0)
print("Number of subject-sessions per group:")
for group, count in group_counts.items():
    print(f"{group}: {count}")

print("\nSession-wise Cohen's kappa by group:")
for row in df_kappa.itertuples(index=False):
    print(f"Subject: {row.Subject} | Session: {row.Session} | Group: {row.Group} | Kappa: {row.Kappa:.2f}")

print("\nPer-subject Overall kappa values:")
for row in df_overall.itertuples(index=False):
    print(f"Subject: {row.Subject} | Kappa: {row.Kappa:.2f}")
