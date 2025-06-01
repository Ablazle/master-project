"""
Script Purpose:
Performs confidence-weighted majority voting across CSV files containing sleep stage predictions.
For each unique (Subject, Session, Epoch), selects PredictedStage with the highest total summed confidence.
If a tie in confidence, uses the count of stages (number of models voting for each).
If still a tie, interpolates from the epoch before/after (if available); otherwise, chooses the lowest numeric stage.
Produces a CSV file: Subject, Session, Epoch, MajStage, sorted by Subject (integer order), Session, and Epoch.
"""

import os
import glob
import pandas as pd
from collections import defaultdict, Counter

# Specify input and output directories
input_directory = r'O:\Tech_NeuroData\Code\New_1CH_AUHCologne\OUS_files\sleepstages_scalp'
output_directory = r'O:\Tech_NeuroData\Code\New_1CH_AUHCologne\OUS_files\Majority_scalp'
output_filename = 'majority_voted_scalp_sleep_stages.csv'

# Find all CSV files in the input directory
csv_file_paths = glob.glob(os.path.join(input_directory, '*.csv'))

# Load all CSVs into a list of DataFrames
dataframes = []
for file_path in csv_file_paths:
    dataframe = pd.read_csv(file_path, dtype={
        'Subject': str, 'Session': str, 'Epoch': int, 'PredictedStage': int, 'Confidence': float
    })
    dataframes.append(dataframe)
combined_dataframe = pd.concat(dataframes, ignore_index=True)

def format_session(session_value):
    if session_value.startswith('ses-'):
        return session_value
    session_number = session_value.split('-')[-1] if '-' in session_value else session_value
    return f'ses-{int(session_number):02d}'

combined_dataframe['Session'] = combined_dataframe['Session'].apply(format_session)

# Step 1: Aggregate predictions by (Subject, Session, Epoch)
aggregation = defaultdict(lambda: defaultdict(float))   # Summed confidence
stage_counts = defaultdict(lambda: defaultdict(int))    # Vote counts

for _, row in combined_dataframe.iterrows():
    subject = row['Subject']
    session = row['Session']
    epoch = int(row['Epoch'])
    predicted_stage = int(row['PredictedStage'])
    confidence = float(row['Confidence'])
    key = (subject, session, epoch)
    aggregation[key][predicted_stage] += confidence
    stage_counts[key][predicted_stage] += 1

# Step 2: Prepare for temporal interpolation
# Build structure: {(subject, session): {epoch: maj_stage}}
epochwise_majority = defaultdict(dict)
all_epochs_by_sub_ses = defaultdict(list)
for subject, session, epoch in set((k[0], k[1], k[2]) for k in aggregation.keys()):
    all_epochs_by_sub_ses[(subject, session)].append(epoch)

# Step 3: Majority voting with advanced tie-break logic
majority_votes = []
for key in sorted(aggregation.keys()):
    subject, session, epoch = key
    stage_conf = aggregation[key]
    stage_cnt = stage_counts[key]

    # (1) Stages with highest confidence
    max_conf = max(stage_conf.values())
    conf_candidates = [stage for stage, conf in stage_conf.items() if conf == max_conf]

    # (2) If tie, use highest count
    if len(conf_candidates) > 1:
        counts = {stage: stage_cnt[stage] for stage in conf_candidates}
        max_votes = max(counts.values())
        count_candidates = [stage for stage, cnt in counts.items() if cnt == max_votes]
    else:
        count_candidates = conf_candidates

    # (3) If still tie, interpolate temporally
    if len(count_candidates) > 1:
        # Look for previous and next epoch's maj_stage (if available and not tied)
        prev_epoch = epoch - 1
        next_epoch = epoch + 1
        prev_maj = epochwise_majority.get((subject, session), {}).get(prev_epoch, None)
        next_maj = epochwise_majority.get((subject, session), {}).get(next_epoch, None)
        interp_stage = None
        if prev_maj in count_candidates:
            interp_stage = prev_maj
        elif next_maj in count_candidates:
            interp_stage = next_maj
        else:
            interp_stage = min(count_candidates)
        maj_stage = interp_stage
    else:
        maj_stage = count_candidates[0]

    # Save for interpolation (future epochs may use this)
    epochwise_majority[(subject, session)][epoch] = maj_stage
    majority_votes.append({'Subject': subject, 'Session': session, 'Epoch': epoch, 'MajStage': maj_stage})

# Sorting utilities
def extract_subject_number(subject_string):
    return int(subject_string.split('-')[1])

majority_votes_df = pd.DataFrame(majority_votes)
majority_votes_df = majority_votes_df.sort_values(
    by=['Subject', 'Session', 'Epoch'],
    key=lambda column: column.map(lambda x: extract_subject_number(x) if column.name == 'Subject' else x)
).reset_index(drop=True)

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)
output_path = os.path.join(output_directory, output_filename)
majority_votes_df.to_csv(output_path, index=False)

"""
Result Summary:
- Confidence-weighted, vote-count, and temporal interpolation (epoch) tie-breaking is performed.
- The output CSV contains one row per (Subject, Session, Epoch) with the majority-voted stage.
- Sorting is performed by integer value of Subject, then by Session, then by Epoch.
- No ties resolved solely by lowest stage unless no interpolation possible.
"""
