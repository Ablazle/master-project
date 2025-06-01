import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from collections import Counter

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

# Subject-to-group mapping (for validation only)
subject_to_group = {
    'sub-24390': 'AD', 'sub-36821': 'DLB', 'sub-18657': 'HC', 'sub-22519': 'AD', 'sub-21502': 'AD',
    'sub-23608': 'AD', 'sub-15749': 'HC', 'sub-22073': 'AD', 'sub-35712': 'DLB', 'sub-20619': 'AD',
    'sub-25813': 'AD', 'sub-29387': 'AD', 'sub-29708': 'AD', 'sub-13167': 'HC', 'sub-17829': 'HC',
    'sub-24263': 'AD', 'sub-22987': 'AD', 'sub-23906': 'AD', 'sub-21870': 'AD', 'sub-20634': 'AD',
    'sub-25394': 'AD', 'sub-34701': 'DLB', 'sub-21388': 'AD', 'sub-20347': 'AD', 'sub-20365': 'AD',
    'sub-30985': 'DLB', 'sub-21753': 'AD', 'sub-32801': 'DLB', 'sub-23842': 'AD', 'sub-29732': 'AD',
    'sub-26530': 'AD', 'sub-23152': 'AD', 'sub-18423': 'HC', 'sub-25634': 'AD', 'sub-29341': 'AD',
    'sub-29827': 'AD', 'sub-14890': 'HC', 'sub-25219': 'AD', 'sub-26374': 'AD', 'sub-17098': 'HC',
    'sub-31976': 'DLB', 'sub-23650': 'AD', 'sub-14360': 'HC', 'sub-25316': 'AD', 'sub-26578': 'AD',
    'sub-37915': 'DLB', 'sub-19742': 'HC', 'sub-39201': 'DLB', 'sub-22176': 'AD', 'sub-21235': 'AD',
    'sub-21847': 'AD', 'sub-37790': 'DLB', 'sub-22964': 'AD', 'sub-25609': 'AD', 'sub-22873': 'AD'
}

# Cohort subject-session explicit ordering (sorted ascending)
ad_order = sorted([
    ('sub-25813', 'ses-01'), ('sub-25813', 'ses-02'),
    ('sub-29708', 'ses-01'), ('sub-29708', 'ses-02'),
    ('sub-20619', 'ses-01'), ('sub-20619', 'ses-02'),
    ('sub-21870', 'ses-01'), ('sub-21870', 'ses-02'), ('sub-21870', 'ses-03'),
    ('sub-22073', 'ses-01'), ('sub-22073', 'ses-02'), ('sub-22073', 'ses-03'),
    ('sub-23842', 'ses-01'), ('sub-23842', 'ses-02'),
    ('sub-23906', 'ses-01'), ('sub-23906', 'ses-02'), ('sub-23906', 'ses-03'),
    ('sub-26578', 'ses-01'), ('sub-26578', 'ses-02'),
    ('sub-25394', 'ses-01'), ('sub-25394', 'ses-02'), ('sub-25394', 'ses-03'),
    ('sub-22987', 'ses-01'), ('sub-22987', 'ses-02'),
    ('sub-21847', 'ses-01'), ('sub-21847', 'ses-02'), ('sub-21847', 'ses-03'),
    ('sub-21235', 'ses-01'), ('sub-21235', 'ses-02'),
    ('sub-24390', 'ses-01'), ('sub-22519', 'ses-01'), ('sub-21502', 'ses-01'), ('sub-23608', 'ses-01'),
    ('sub-29387', 'ses-01'), ('sub-24263', 'ses-01'), ('sub-20634', 'ses-01'), ('sub-21388', 'ses-01'),
    ('sub-20347', 'ses-01'), ('sub-20365', 'ses-01'), ('sub-21753', 'ses-01'), ('sub-29732', 'ses-01'),
    ('sub-26530', 'ses-01'), ('sub-23152', 'ses-01'), ('sub-25634', 'ses-01'), ('sub-29341', 'ses-01'),
    ('sub-29827', 'ses-01'), ('sub-25219', 'ses-01'), ('sub-26374', 'ses-01'), ('sub-23650', 'ses-01'),
    ('sub-25316', 'ses-01'), ('sub-22176', 'ses-01'), ('sub-22964', 'ses-01'), ('sub-25609', 'ses-01'), ('sub-22873', 'ses-01')
])

dlb_order = sorted([
    ('sub-31976', 'ses-01'), ('sub-31976', 'ses-02'), ('sub-31976', 'ses-03'),
    ('sub-36821', 'ses-01'), ('sub-36821', 'ses-02'),
    ('sub-35712', 'ses-01'), ('sub-35712', 'ses-02'), ('sub-35712', 'ses-03'),
    ('sub-34701', 'ses-01'), ('sub-30985', 'ses-01'), ('sub-32801', 'ses-01'),
    ('sub-37915', 'ses-01'), ('sub-39201', 'ses-01'), ('sub-37790', 'ses-01')
])

hc_order = sorted([
    'sub-18657', 'sub-15749', 'sub-13167', 'sub-17829', 'sub-18423',
    'sub-14890', 'sub-17098', 'sub-14360', 'sub-19742'
])

group_orders = [
    ("HC", hc_order),
    ("DLB", dlb_order),
    ("AD", ad_order)
]

# Custom label lists: now HC subjects have ses-01 and all lists sorted ascending
ad_custom_labels = [f"{sub}, {ses}" for (sub, ses) in ad_order]
dlb_custom_labels = [f"{sub}, {ses}" for (sub, ses) in dlb_order]
hc_custom_labels = [f"{sub}, ses-01" for sub in hc_order]

custom_label_dict = {
    "AD": ad_custom_labels,
    "DLB": dlb_custom_labels,
    "HC": hc_custom_labels
}

# Load and organize data by (subject, session)
data_root = r"O:\Tech_NeuroData\Code\Prime_sleep_stages\New_Majority\new_majority"
csv_files = []
for root, dirs, files in os.walk(data_root):
    for fname in files:
        if fname.lower().endswith(".csv"):
            csv_files.append(os.path.join(root, fname))

epoch_duration_sec = 30

records_by_group = {'AD': {}, 'DLB': {}, 'HC': {}}  # Dict[group][(subject, session)] = df

for csv_path in csv_files:
    df = pd.read_csv(csv_path, dtype={'Subject': str, 'Session': str, 'Epoch': int, 'MajStage': int})
    if 'Subject' not in df.columns or 'Session' not in df.columns or 'Epoch' not in df.columns or 'MajStage' not in df.columns:
        continue
    subject = df['Subject'].iloc[0]
    session = df['Session'].iloc[0]
    if subject not in subject_to_group:
        continue
    df['Time_hours'] = df['Epoch'] * epoch_duration_sec / 3600
    if df['Time_hours'].max() < 24:
        continue
    df_24h = df[df['Time_hours'] < 24].copy()
    group = subject_to_group[subject]
    # Store by (subject, session) for all groups (ensures consistent labeling)
    records_by_group[group][(subject, session)] = df_24h

extra_blank_rows = 1  # Number of extra blank rows before and after each cohort header

all_rows = []
row_labels = []
row_groups = []
custom_labels_full = []

for group, order in group_orders:
    # --- Insert blank rows before ---
    for _ in range(extra_blank_rows):
        all_rows.append(None)
        row_labels.append('')
        row_groups.append(group)
        custom_labels_full.append('')

    # --- Insert cohort label row ---
    all_rows.append(None)
    row_labels.append(group)
    row_groups.append(group)
    custom_labels_full.append(group)

    # --- Insert blank rows after ---
    for _ in range(extra_blank_rows):
        all_rows.append(None)
        row_labels.append('')
        row_groups.append(group)
        custom_labels_full.append('')

    clist = custom_label_dict[group]
    label_index = 0
    if group == "HC":
        for subject in order:
            key = (subject, 'ses-01')
            if key in records_by_group["HC"]:
                all_rows.append(records_by_group["HC"][key])
                row_labels.append(f"{subject}, ses-01")
                row_groups.append(group)
                custom_labels_full.append(clist[label_index])
                label_index += 1
    else:
        for (subject, session) in order:
            key = (subject, session)
            if key in records_by_group[group]:
                all_rows.append(records_by_group[group][key])
                row_labels.append(f"{subject}, {session}")
                row_groups.append(group)
                custom_labels_full.append(clist[label_index])
                label_index += 1


nrows = len(all_rows)
fig_height = 0.3 * nrows + 1  # Decrease scaling factor for a tighter fit

fig, ax = plt.subplots(figsize=(16, fig_height))
ax.set_facecolor('white')

ax.set_xlim(0, 24)
ax.set_xlabel('Time (hours)', size=17, fontweight='bold')
ax.set_xticks(np.arange(0, 25, 2))

for idx, df in enumerate(all_rows):
    if df is None:
        continue
    ax.scatter(
        df['Time_hours'],
        np.full_like(df['Time_hours'], idx),
        c=df['MajStage'].map({
            0: '#FFFF00', 1: '#00FFFF', 2: '#4169E1', 3: '#000080', 4: '#FF0000'
        }).values,
        marker='s', s=200, edgecolor='black', linewidth=0.14, zorder=3
    )

ax.set_yticks(range(nrows))
ax.set_yticklabels([''] * nrows)
ax.set_ylim(nrows - 0.1, -0.9)
fig_height = 0.1 * nrows + 1  # Minimal vertical space, rows packed


for idx, label in enumerate(custom_labels_full):
    if all_rows[idx] is None:
        ax.text(-0.3, idx, label,
                ha='right', va='center', fontsize=17, fontweight='bold',
                color='black', clip_on=False)
    else:
        ax.text(-0.3, idx, label,
                ha='right', va='center', fontsize=12, fontweight='bold',
                color='black', clip_on=False
        )

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_color('black')

ax.set_ylim(-0.7, nrows - 0.3)
ax.invert_yaxis()
ax.tick_params(axis='x', labelsize=15, length=4)
ax.tick_params(axis='y', left=False, labelleft=False)

legend_handles = [
    Patch(facecolor='#FFFF00', edgecolor='black', label='W'),
    Patch(facecolor='#00FFFF', edgecolor='black', label='N1'),
    Patch(facecolor='#4169E1', edgecolor='black', label='N2'),
    Patch(facecolor='#000080', edgecolor='black', label='N3'),
    Patch(facecolor='#FF0000', edgecolor='black', label='REM')
]

# --- NEW: Place title and legend on the same line, closer to the plot ---

# Add a suptitle to the left
#fig.suptitle("Sleep Distribution", fontsize=20, fontweight='black', x=0.4, y=0.96, ha='left')

# Add legend to the right of the title, on the same line
#fig.legend(
#    handles=legend_handles,
#    loc='upper center',
#    bbox_to_anchor=(0.7, 0.97),  # adjust this to move the legend horizontally
#    ncol=5,
#    frameon=False,
#    handlelength=1.7,
#    handleheight=1.5,
#    fontsize=13,
#    columnspacing=1.4,
#    borderaxespad=0.1
#)

# Reduce top padding to bring both elements closer to the axes
fig.subplots_adjust(top=0.93, bottom=0.08, left=0.19, right=0.99)


# Thicken only the outer frame of the plot (axes spines)
for position, spine in ax.spines.items():
    spine.set_visible(True)
    if position in ['top', 'bottom', 'left', 'right']:
        spine.set_linewidth(2)  # Make as thick as desired
        spine.set_color('black')

# --- Cohort recording and unique subject summary ---

# --- Count only the custom labels actually present in the plot ---

from collections import defaultdict

# Initialize counters for each cohort
actual_label_counts = defaultdict(list)
actual_subject_sets = defaultdict(set)

for label, group, row in zip(custom_labels_full, row_groups, all_rows):
    # Skip blank rows and cohort headers
    if row is not None and label not in ['', 'HC', 'DLB', 'AD']:
        actual_label_counts[group].append(label)
        subject_id = label.split(',')[0].strip()
        actual_subject_sets[group].add(subject_id)

for cohort in ['HC', 'DLB', 'AD']:
    label_list = actual_label_counts[cohort]
    subject_set = actual_subject_sets[cohort]
    print(f"{cohort}:")
    print(f"  Number of labels actually plotted: {len(label_list)}")
    print(f"  Number of unique subjects actually plotted: {len(subject_set)}")
    print(f"  Unique subject list: {sorted(subject_set)}")
    print("-" * 50)

plt.show()
#plt.savefig("sleep_distribution.pdf", bbox_inches='tight')
