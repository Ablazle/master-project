# Description:
# Sets Computer Modern (cmr10) font for all matplotlib plot text, then generates hypnogram plots 
# for given subjects/sessions using loaded scalp and ear stage CSVs. No changes to plot logic or color.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

# ──────────────────────────────────────────────────────────────
# Set global Computer Modern font, matching the style of the kappa plots
# ──────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "axes.titlesize": 20,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 18,
    "axes.unicode_minus": False,
})

def plotHypnoGram(labels, axis, title, show_xlabel=False, show_xticks=False, xtick_interval=5):
    """
    Plots a hypnogram (sleep stage time series) for a given set of labels.
    Parameters:
        labels (array-like): Array of integer sleep stage labels.
        axis (matplotlib.axes.Axes): Axis to plot on.
        title (str): Title for the plot.
        show_xlabel (bool): Whether to show the x-axis label.
        show_xticks (bool): Whether to show x-ticks and labels.
        xtick_interval (int): Interval (in minutes) for x-ticks.
    """
    stage_transform = {0: 0, 1: 2, 2: 3, 3: 4, 4: 1, 5: -1}
    y_values = np.array([stage_transform[label] for label in labels])
    num_epochs = len(y_values)
    minutes = np.arange(num_epochs) / 2  # Each epoch = 30 seconds

    valid_mask = y_values != -1
    axis.plot(minutes[valid_mask], y_values[valid_mask], color="k", linewidth=1)
    axis.plot(minutes[(y_values == 1)], y_values[(y_values == 1)], "r.", label="REM")
    axis.invert_yaxis()
    axis.set_yticks([0, 1, 2, 3, 4])
    axis.set_yticklabels(["W", "R", "N1", "N2", "N3"])
    axis.set_title(title, fontsize=15)
    if show_xlabel:
        axis.set_xlabel("Minutes")
    if show_xticks:
        max_min = np.ceil(minutes[-1])
        axis.set_xticks(np.arange(0, max_min + 1, xtick_interval))
        axis.set_xticklabels([f"{int(x):02d}" for x in np.arange(0, max_min + 1, xtick_interval)])
    else:
        axis.set_xticks([])
        axis.set_xticklabels([])

# Specify paths to the single aggregated scalp and ear CSV files
scalp_csv = r"O:\Tech_NeuroData\Code\New_1CH_AUHCologne\OUS_files\Majority_scalp\majority_voted_scalp_sleep_stages.csv"
ear_csv = r"O:\Tech_NeuroData\Code\New_1CH_AUHCologne\OUS_files\Majority_ear\majority_voted_ear_sleep_stages.csv"

# Load the full CSVs once
scalp_df_all = pd.read_csv(scalp_csv)
ear_df_all = pd.read_csv(ear_csv)

# List of (subject, session, caption) to plot
subjects_sessions_titles = [
    ("sub-152", "01", "HC, sub-152 ses-01"),
    ("sub-263", "01", "AD, sub-263 ses-01"),
    ("sub-3906", "01", "DLB, sub-3906 ses-01"),
]

fig = plt.figure(figsize=(15, 12))
outer_gs = gridspec.GridSpec(3, 1, hspace=0.4, wspace=0.05)

for row_index, (subject, session, caption) in enumerate(subjects_sessions_titles):
    # 2-row inner grid for each subject: scalp (top), ear (bottom)
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[row_index], height_ratios=[1, 1], hspace=0.12
    )
    try:
        # Subset rows for this subject and session
        scalp_df = scalp_df_all[(scalp_df_all['Subject'] == subject) & 
                                (scalp_df_all['Session'].str.replace('ses-', '').str.zfill(2) == session)]
        ear_df = ear_df_all[(ear_df_all['Subject'] == subject) & 
                            (ear_df_all['Session'].str.replace('ses-', '').str.zfill(2) == session)]

        # Check if both dataframes are not empty
        if scalp_df.empty or ear_df.empty:
            raise ValueError("Data missing for subject/session")

        merged = pd.merge(
            scalp_df[["Epoch", "MajStage"]],
            ear_df[["Epoch", "MajStage"]],
            on="Epoch",
            suffixes=("_scalp", "_ear")
        )
        scalp_labels = merged["MajStage_scalp"].astype(int).values
        ear_labels = merged["MajStage_ear"].astype(int).values
    except Exception as e:
        ax_scalp = plt.subplot(inner_gs[0])
        ax_ear = plt.subplot(inner_gs[1], sharex=ax_scalp)
        ax_scalp.text(0.5, 0.5, f"{caption}\nData missing", ha='center', va='center', fontsize=12)
        ax_scalp.axis('off')
        ax_ear.axis('off')
        print(f"Warning: {e}")
        continue

    ax_scalp = plt.subplot(inner_gs[0])
    ax_ear = plt.subplot(inner_gs[1], sharex=ax_scalp)
    # Scalp: top, do not show x-ticks/labels
    plotHypnoGram(
        scalp_labels, ax_scalp,
        caption + " (Top: Scalp Bottom: Ear)",
        show_xlabel=False, show_xticks=False
    )
    # Ear: bottom, show x-ticks/labels only here
    plotHypnoGram(
        ear_labels, ax_ear,
        "",
        show_xlabel=True, show_xticks=True, xtick_interval=5
    )
    ax_scalp.set_xlabel("")  # Remove x-label from top
    plt.setp(ax_scalp.get_xticklabels(), visible=False)
    plt.setp(ax_ear.get_xticklabels(), visible=True)

plt.show()
