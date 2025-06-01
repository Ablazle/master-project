import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plotHypnoGram(labels, ax, title, xlabel, xticks):
    """
    Plot a hypnogram with:
      • Stage codes mapped: 0 (W)→4, 4 (R)→3, 1 (N1)→2, 2 (N2)→1, 3 (N3)→0
      • X-axis in elapsed minutes, using provided xticks
      • Y-axis ordered top→bottom: W, R, N1, N2, N3
      • Only the bottom panel shows xlabel
    """
    mapping = {0:4, 4:3, 1:2, 2:1, 3:0}
    positions = [mapping[int(s)] for s in labels if int(s) in mapping]
    times_min = np.arange(len(positions)) * 0.5

    ax.plot(times_min, positions, color="k", linewidth=1)
    rem_mask = np.array(positions) == mapping[4]
    ax.plot(times_min[rem_mask], np.array(positions)[rem_mask], "r.", label="REM")

    ax.set_yticks([0,1,2,3,4])
    ax.set_yticklabels(["N3","N2","N1","R","W"])
    ax.set_title(title)
    ax.set_xticks(xticks)
    if xlabel:
        ax.set_xlabel(xlabel)

# --- Directories ---
scalp_base = r"O:\Tech_NeuroData\Code\1CH_AUHCologne\AUHC_Majority_Scalp_Hypnograms"
ear_base   = r"O:\Tech_NeuroData\Code\1CH_AUHCologne\AUHC_Majority_Ear_Hypnograms"
output_dir = r"O:\Tech_NeuroData\Code\1CH_AUHCologne\Combined_Hypnogram"
os.makedirs(output_dir, exist_ok=True)

# --- Find all scalp CSVs ---
scalp_files = sorted(glob.glob(os.path.join(scalp_base, "sub-*", "ses-*", "*.csv")))

for scalp_path in scalp_files:
    # Parse subject & session
    scalp_fn  = os.path.basename(scalp_path)
    subj, sess_tok = scalp_fn.split("_")[:2]
    sess_str  = sess_tok.split("-")[1]

    # Load scalp labels
    df_scalp       = pd.read_csv(scalp_path, usecols=["MajStage"])
    labels_scalp   = df_scalp["MajStage"].values

    # Compute common xticks (1-minute intervals)
    n_epochs = len(labels_scalp)
    max_min  = n_epochs * 0.5
    xticks   = np.arange(0, max_min + 1e-6, 1.0)

    # Locate matching ear CSV
    ear_dir = os.path.join(ear_base, subj, f"ses-{sess_str}")
    ear_csvs = glob.glob(os.path.join(ear_dir, "*.csv"))
    if not ear_csvs:
        raise FileNotFoundError(f"No ear CSV found in {ear_dir}")
    ear_path     = ear_csvs[0]
    df_ear       = pd.read_csv(ear_path, usecols=["MajStage"])
    labels_ear   = df_ear["MajStage"].values

    # --- Plot 2×1 hypnogram ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
    plotHypnoGram(
        labels_scalp, axes[0],
        title=f"Scalp Majority Hypnogram: {subj} ses-{sess_str}",
        xlabel=None,
        xticks=xticks
    )
    plotHypnoGram(
        labels_ear, axes[1],
        title=f"Ear Majority Hypnogram: {subj} ses-{sess_str}",
        xlabel="Minutes",
        xticks=xticks
    )
    plt.tight_layout()
    
     # --- Save ---
    sub_out = os.path.join(output_dir, subj, f"ses-{sess_str}")
    os.makedirs(sub_out, exist_ok=True)
    out_name = f"{subj}_ses-{sess_str}_Combined_Majority_Hypnogram.png"
    fig.savefig(os.path.join(sub_out, out_name), bbox_inches="tight")
    plt.close(fig)
