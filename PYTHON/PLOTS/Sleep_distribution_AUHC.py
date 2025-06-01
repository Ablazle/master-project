"""
Sleep-Stage Distribution Plot for Scalp Derivatives

This script loads all CSVs from the specified directory,
maps PredictedStage → stage labels, counts epochs per stage
per channel, and renders a grouped bar chart.
"""

import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Parameters and mappings
DATA_DIR = Path(r"O:\Tech_NeuroData\Code\1CH_AUHCologne\Sleepstages\Scalp_Sleepstages")

# Mapping from integer code to stage label
STAGE_MAP = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}

# Desired order on the x-axis
STAGE_ORDER = ["Wake", "N1", "N2", "N3", "REM"]

# Desired channel order in legend and bar grouping
CHANNEL_ORDER = ["C3-C4", "F3-F4", "F7-F8", "P3-P4", "T7-T8"]

# 2. Aggregate counts
records = []

for csv_path in sorted(DATA_DIR.glob("*.csv")):
    # Extract channel code, e.g. "Sleepstages_1CH_C3C4_...csv" → "C3C4"
    raw_channel = csv_path.stem.split("_")[2]
    # Insert hyphen: "C3C4" → "C3-C4"
    channel_label = f"{raw_channel[:2]}-{raw_channel[2:]}"
    
    df = pd.read_csv(csv_path)
    # Filter to valid stage codes only
    df = df[df["PredictedStage"].isin(STAGE_MAP)].copy()
    # Map to descriptive labels
    df["StageLabel"] = df["PredictedStage"].map(STAGE_MAP)
    
    # Count epochs per stage, ensure all five stages appear
    stage_counts = (
        df["StageLabel"]
        .value_counts()
        .reindex(STAGE_ORDER, fill_value=0)
    )
    
    for stage_label, count in stage_counts.items():
        records.append({
            "Channel": channel_label,
            "Stage": stage_label,
            "Count": int(count)
        })

df_counts = pd.DataFrame(records)


# 3. Plotting
sns.set(style="whitegrid", font_scale=1.1)
fig, ax = plt.subplots(figsize=(10, 6))

# Use the first five colors of the 'tab10' palette
palette = sns.color_palette("tab10", n_colors=len(CHANNEL_ORDER))

sns.barplot(
    data=df_counts,
    x="Stage",
    y="Count",
    hue="Channel",
    order=STAGE_ORDER,
    hue_order=CHANNEL_ORDER,
    palette=palette,
    ax=ax
)

# Annotate each bar with its count
for bar in ax.patches:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + max(df_counts["Count"]) * 0.005,
        f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=9
    )

# Titles and labels
#ax.set_title("Sleep-Stage Distribution Across Scalp Derivatives")
ax.set_xlabel("Predicted Sleep Stage")
ax.set_ylabel("Number of Epochs")

# Legend placement
ax.legend(
    title="Channel",
    loc="upper right",
    frameon=True
)

print(df_counts.columns)
print(df_counts.head())


plt.tight_layout()
plt.show()
