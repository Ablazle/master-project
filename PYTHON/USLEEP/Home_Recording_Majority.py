# Script Purpose:
# Aggregates only valid CSV files (where all rows have non-NaN Confidence values) for each subject/session.
# For each (Subject, Session, Epoch), performs confidence-weighted majority voting with advanced tie-breaking.
# Outputs one CSV per subject-session: Subject, Session, Epoch, MajStage, sorted by Epoch, in the specified directory structure.

import os
import glob
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# Specify input and output directories
input_root_directory = '/home/au682014/eeglab_project/new_sleepstages'
output_root_directory = '/home/au682014/eeglab_project/new_majority'

print("Searching for CSV files...")
csv_file_paths = glob.glob(
    os.path.join(input_root_directory, 'sub-*', 'ses-*', '*.csv'),
    recursive=True
)
print(f"Discovered {len(csv_file_paths)} CSV files.")

if not csv_file_paths:
    raise FileNotFoundError('No CSV files found in the specified directory structure.')

dataframes = []
omitted_files = []

print("Loading and filtering CSV files...")
for file_path in tqdm(csv_file_paths, desc="Loading CSVs", unit="file"):
    try:
        dataframe = pd.read_csv(
            file_path,
            dtype={'Subject': str, 'Session': str, 'Epoch': int, 'PredictedStage': int, 'Confidence': float}
        )
    except Exception:
        omitted_files.append(file_path)
        continue
    # Omit files where any value in Confidence is NaN
    if dataframe['Confidence'].isna().any():
        omitted_files.append(file_path)
        continue
    # Optional: if all PredictedStage are 0 and all Confidence are NaN, omit
    if (dataframe['PredictedStage'] == 0).all() and dataframe['Confidence'].isna().all():
        omitted_files.append(file_path)
        continue
    dataframes.append(dataframe)

print(f"Included {len(dataframes)} CSV files for aggregation.")
print(f"Omitted {len(omitted_files)} files due to invalid or missing Confidence values.")

if not dataframes:
    raise ValueError('No valid CSV files found for aggregation (all files omitted due to NaN Confidence values).')

# Combine all DataFrames into a single DataFrame
print("Concatenating all valid CSV files...")
combined_dataframe = pd.concat(dataframes, ignore_index=True)

def format_subject(subject_value):
    if subject_value.startswith('sub-'):
        return subject_value
    subject_number = subject_value.split('-')[-1] if '-' in subject_value else subject_value
    return f'sub-{int(subject_number):03d}'

def format_session(session_value):
    if session_value.startswith('ses-'):
        return session_value
    session_number = session_value.split('-')[-1] if '-' in session_value else session_value
    return f'ses-{int(session_number):02d}'

print("Standardizing Subject and Session identifiers...")
combined_dataframe['Subject'] = combined_dataframe['Subject'].apply(format_subject)
combined_dataframe['Session'] = combined_dataframe['Session'].apply(format_session)

print("Aggregating predictions by (Subject, Session, Epoch)...")
aggregation = defaultdict(lambda: defaultdict(float))   # Summed confidence
stage_counts = defaultdict(lambda: defaultdict(int))    # Vote counts

for _, row in tqdm(combined_dataframe.iterrows(), total=combined_dataframe.shape[0], desc="Aggregating", unit="row"):
    subject = row['Subject']
    session = row['Session']
    epoch = int(row['Epoch'])
    predicted_stage = int(row['PredictedStage'])
    confidence = float(row['Confidence'])
    key = (subject, session, epoch)
    aggregation[key][predicted_stage] += confidence
    stage_counts[key][predicted_stage] += 1

# Prepare for temporal interpolation
epochwise_majority = defaultdict(dict)

print("Performing majority voting and tie-breaking...")
majority_votes = []
sorted_keys = sorted(aggregation.keys())
for idx, key in enumerate(tqdm(sorted_keys, desc="Majority voting", unit="epoch")):
    subject, session, epoch = key
    stage_conf = aggregation[key]
    stage_cnt = stage_counts[key]

    # 1. Stages with highest confidence
    max_conf = max(stage_conf.values())
    conf_candidates = [stage for stage, conf in stage_conf.items() if conf == max_conf]

    # 2. If tie, use highest count
    if len(conf_candidates) > 1:
        counts = {stage: stage_cnt[stage] for stage in conf_candidates}
        max_votes = max(counts.values())
        count_candidates = [stage for stage, cnt in counts.items() if cnt == max_votes]
    else:
        count_candidates = conf_candidates

    # 3. If still tie, interpolate temporally
    if len(count_candidates) > 1:
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

    epochwise_majority[(subject, session)][epoch] = maj_stage
    majority_votes.append({'Subject': subject, 'Session': session, 'Epoch': epoch, 'MajStage': maj_stage})

print("Converting to DataFrame and sorting results...")
majority_votes_df = pd.DataFrame(majority_votes)

def extract_subject_number(subject_string):
    return int(subject_string.split('-')[1])

def extract_session_number(session_string):
    return int(session_string.split('-')[1])

majority_votes_df = majority_votes_df.sort_values(
    by=['Subject', 'Session', 'Epoch'],
    key=lambda column: (
        column.map(extract_subject_number) if column.name == 'Subject' else
        column.map(extract_session_number) if column.name == 'Session' else
        column
    )
).reset_index(drop=True)

print("Writing per subject-session output files...")
subject_session_groups = list(majority_votes_df.groupby(['Subject', 'Session']))
for (subject, session), group_df in tqdm(subject_session_groups, desc="Writing CSVs", unit="group"):
    output_path = os.path.join(
        output_root_directory,
        subject,
        session,
        'majorityvoted.csv'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    group_df_sorted = group_df.sort_values('Epoch').reset_index(drop=True)
    group_df_sorted.to_csv(output_path, index=False)

print("Processing complete. All per subject-session majorityvoted.csv files have been written.")
print(f"Total output files: {len(subject_session_groups)}")
print(f"Total omitted input files: {len(omitted_files)}")

# Optional: Uncomment to print omitted files for troubleshooting
# if omitted_files:
#     print("Omitted files:")
#     for path in omitted_files:
#         print(path)
