import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import confusion_matrix

# ──────────────────────────────────────────────────────────────────────────────
# 0) Cosmetic: rcParams
# ──────────────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})
sns.set_theme(style="whitegrid", font="serif")

# ──────────────────────────────────────────────────────────────────────────────
# 1) Data paths and sleep stage names/codes
# ──────────────────────────────────────────────────────────────────────────────
scalp_csv = r"O:\Tech_NeuroData\Code\New_1CH_AUHCologne\OUS_files\Majority_scalp\majority_voted_scalp_sleep_stages.csv"
ear_csv   = r"O:\Tech_NeuroData\Code\New_1CH_AUHCologne\OUS_files\Majority_ear\majority_voted_ear_sleep_stages.csv"

stage_codes = [0, 1, 2, 3, 4]
stage_names = ["W", "N1", "N2", "N3", "REM"]

# ──────────────────────────────────────────────────────────────────────────────
# 2) Load and merge paired scalp & ear epochs
# ──────────────────────────────────────────────────────────────────────────────
scalp_df = pd.read_csv(scalp_csv, usecols=["Subject", "Session", "Epoch", "MajStage"])
ear_df = pd.read_csv(ear_csv, usecols=["Subject", "Session", "Epoch", "MajStage"])

# Ensure consistent types
scalp_df["Subject"] = scalp_df["Subject"].astype(str)
scalp_df["Session"] = scalp_df["Session"].astype(str)
scalp_df["Epoch"] = scalp_df["Epoch"].astype(int)
scalp_df["MajStage"] = scalp_df["MajStage"].astype(int)

ear_df["Subject"] = ear_df["Subject"].astype(str)
ear_df["Session"] = ear_df["Session"].astype(str)
ear_df["Epoch"] = ear_df["Epoch"].astype(int)
ear_df["MajStage"] = ear_df["MajStage"].astype(int)

# Merge on Subject, Session, Epoch
merged = pd.merge(
    scalp_df.rename(columns={"MajStage": "True"}),
    ear_df.rename(columns={"MajStage": "Pred"}),
    on=["Subject", "Session", "Epoch"]
)

if merged.empty:
    raise RuntimeError("No matching (Subject, Session, Epoch) pairs found between scalp and ear CSVs.")

y_true = merged["True"].to_numpy()
y_pred = merged["Pred"].to_numpy()

# ──────────────────────────────────────────────────────────────────────────────
# 3) Compute 5-class confusion matrix
# ──────────────────────────────────────────────────────────────────────────────
cm5 = confusion_matrix(y_true, y_pred, labels=stage_codes)
df5 = pd.DataFrame(cm5, index=stage_names, columns=stage_names)

# ──────────────────────────────────────────────────────────────────────────────
# 4) Function for custom confusion matrix plotting
# ──────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix_rowwise_color(
    df_cm, title, xlabel, ylabel, figsize=(8,6), annot_size=15, title_size=17
):
    data = df_cm.values.astype(float)
    n_rows, n_cols = data.shape

    # Row-wise percent
    row_sums = data.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_percent = np.divide(data, row_sums, where=row_sums != 0) * 100

    # Row-wise normalization for color
    normalized_colors = np.zeros_like(row_percent)
    for i in range(row_percent.shape[0]):
        max_val = row_percent[i, :].max() if row_percent[i, :].max() > 0 else 1
        normalized_colors[i, :] = row_percent[i, :] / max_val

    # colors to choose from, blues, greens, 
    cmap = plt.get_cmap("Blues")  # Use a pastel colormap for better visibility

    fig, ax = plt.subplots(figsize=figsize)
    # Draw each cell manually (no gridlines, no colorbar)
    for i in range(n_rows):
        for j in range(n_cols):
            color = cmap(normalized_colors[i, j])
            ax.add_patch(
                plt.Rectangle(
                    (j, i), 1, 1,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.0
                )
            )
            text_color = "white" if normalized_colors[i, j] > 0.6 else "black"
            annotation = f"{int(data[i, j])}\n{row_percent[i, j]:.1f}%"
            ax.text(
                j + 0.5, i + 0.5,
                annotation,
                ha="center", va="center",
                fontsize=annot_size,
                fontweight="bold" if i == j else "normal",
                color=text_color,
                family="serif"
            )

    # Axis formatting
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(df_cm.columns, fontsize=14, family="serif")
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel(xlabel, labelpad=10, fontsize=16, family="serif")
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', top=True, bottom=False)

    ax.set_xlabel(xlabel, labelpad=10, fontsize=16, family="serif")
    ax.xaxis.set_label_position('bottom')

    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(df_cm.index, fontsize=14, family="serif")
    ax.set_ylabel(ylabel, labelpad=-25, fontsize=16, family="serif")
    ax.tick_params(axis='y', left=True, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    rect = plt.Rectangle(
        (0, 0), n_cols, n_rows,
        fill=False, edgecolor="black", linewidth=1, zorder=3, clip_on=False
    )
    ax.add_patch(rect)
    ax.set_title(title, pad=10, fontsize=title_size, family="serif")
    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 5) Plot 5-class confusion matrix
# ──────────────────────────────────────────────────────────────────────────────
plot_confusion_matrix_rowwise_color(
    df5,
    title="",
    xlabel="Ear",
    ylabel="Scalp"
)

# ──────────────────────────────────────────────────────────────────────────────
# 6) Compute and plot binary (Wake vs Sleep[N2]) confusion matrix
# ──────────────────────────────────────────────────────────────────────────────
mask = np.isin(y_true, [0, 2])
y_true_bin = np.where(y_true[mask] == 2, 1, 0)
y_pred_bin = np.where(y_pred[mask] == 2, 1, 0)

cm2 = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
df2 = pd.DataFrame(cm2, index=["W", "Sleep"], columns=["W", "Sleep"])

plot_confusion_matrix_rowwise_color(
    df2,
    title="",
    xlabel="Ear",
    ylabel="Scalp",
    figsize=(5, 4),
    annot_size=16,
    title_size=15
)
