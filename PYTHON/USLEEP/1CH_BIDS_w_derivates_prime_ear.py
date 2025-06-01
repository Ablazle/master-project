import os
import csv
import json
import warnings
from datetime import datetime
from pathlib import Path
import mne

from csdp_training.experiments.bids_predictor import BIDS_USleep_Predictor

# ----------------------------------------------------------------------------
# 0) Suppress irrelevant warnings
# ----------------------------------------------------------------------------
warnings.filterwarnings('ignore', message='.*boundary events.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
mne.set_log_level('ERROR')

# ----------------------------------------------------------------------------
# 1) Load bipolar derivation specification (JSON)
# ----------------------------------------------------------------------------
derivation_spec_path = "/home/au682014/eeglab_project/OUS/all_ear_derivates.json"
with open(derivation_spec_path, 'r', encoding='utf-8') as f:
    derivation_spec = json.load(f)

# Flatten all unique derivations in JSON
all_derivations = sorted({
    deriv
    for subj in derivation_spec.values()
    for sess_deriv in subj.values()
    for deriv in sess_deriv
})

# ----------------------------------------------------------------------------
# 2) Patch MNE's EDF reader to apply bipolar derivations per subject/session
# ----------------------------------------------------------------------------
_original_read_raw_edf = mne.io.read_raw_edf

def patched_read_raw_edf(file_path, preload=True, *args, **kwargs):
    """
    Patches MNE's read_raw_edf to apply subject/session-specific bipolar derivations.
    """
    raw = _original_read_raw_edf(file_path, preload=preload, *args, **kwargs)
    # Parse subject and session from the filename
    fname = os.path.basename(file_path)
    # Expecting: sub-XXX_ses-XX_task-sleep_eeg.edf
    # Parse subject and session codes
    name_parts = fname.split('_')
    if len(name_parts) < 3:
        return raw  # Unexpected file name, skip
    subj_key = name_parts[0]   # sub-XXX
    sess_key = name_parts[1]   # ses-XX
    # Look up the derivations for this subject/session
    derivs = derivation_spec.get(subj_key, {}).get(sess_key, [])
    if not derivs:
        return raw  # No derivations to apply, return unchanged
    anodes = []
    cathodes = []
    for pair in derivs:
        # Each 'pair' string is expected as 'A-B'
        if '-' not in pair:
            continue
        left, right = pair.split('-')
        anodes.append(left)
        cathodes.append(right)
    if anodes and cathodes and len(anodes) == len(cathodes):
        bipolar = mne.set_bipolar_reference(
            inst      = raw,
            anode     = anodes,
            cathode   = cathodes,
            ch_name   = derivs,
            drop_refs = False,
            copy      = True
        )
        bipolar.pick(derivs)
        return bipolar
    return raw

mne.io.read_raw_edf = patched_read_raw_edf

# ----------------------------------------------------------------------------
# 3) Configuration paths and session keys
# ----------------------------------------------------------------------------
checkpoint_paths = [
    "/home/au682014/eeglab_project/Weights/m1m2_auhcologne_onechannel_lightning.ckpt"
]
data_directory     = "/home/au682014/eeglab_project/OUS/Usleep_files_resampled128Hz_cropped"
data_extension     = ".edf"
task_name          = "sleep"
output_base_folder = "/home/au682014/eeglab_project/OUS_files/sleepstages_ear"

# Determine all .edf files present
edf_files = []
for root, dirs, files in os.walk(data_directory):
    for file in files:
        if file.endswith(data_extension):
            edf_files.append(os.path.join(root, file))

# Build a subject-session-file mapping for efficiency
file_map = dict()
for edf in edf_files:
    fname = os.path.basename(edf)
    name_parts = fname.split('_')
    if len(name_parts) < 3:
        continue
    subj_key = name_parts[0]  # sub-XXX
    sess_key = name_parts[1]  # ses-XX
    file_map[(subj_key, sess_key)] = edf

subject_keys = list(derivation_spec.keys())
session_keys = ['ses-01', 'ses-02', 'ses-03']

# ----------------------------------------------------------------------------
# 4) Main: predict and export CSV per derivation
# ----------------------------------------------------------------------------
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    for ckpt in checkpoint_paths:
        ckpt_name = Path(ckpt).stem
        print(f"\n[INFO] Processing weights: {ckpt_name}")

        out_dir = os.path.join(output_base_folder, ckpt_name)
        os.makedirs(out_dir, exist_ok=True)

        for deriv in all_derivations:
            print(f"\n[INFO] Derivation: {deriv}")
            rows = []

            for subj in subject_keys:
                for sess in session_keys:
                    # Only process if file exists and derivation is specified for this subj/sess
                    if (subj, sess) not in file_map:
                        continue
                    deriv_list = derivation_spec.get(subj, {}).get(sess, [])
                    if deriv not in deriv_list:
                        continue
                    sid = subj.split('-')[1]
                    ses = sess.split('-')[1]
                    print(f"  [INFO] Processing {subj} {sess}")

                    predictor = BIDS_USleep_Predictor(
                        data_dir       = data_directory,
                        data_extension = data_extension,
                        data_task      = task_name,
                        subjects       = [sid],
                        sessions       = [ses]
                    )

                    # Build dataset
                    try:
                        dataset = predictor.build_dataset([deriv])
                    except Exception as e:
                        print(f"    [ERROR] Build error: {e}")
                        continue
                    if not dataset:
                        print("    [WARNING] No data, skipping.")
                        continue

                    # Prediction
                    try:
                        preds_list, confs_list = predictor.predict_all(ckpt, dataset)
                    except Exception as e:
                        print(f"    [ERROR] Prediction error: {e}")
                        continue

                    # Each (pred_tensor, conf_tensor) pair:
                    for pred_tensor, conf_tensor in zip(preds_list, confs_list):
                        n_epochs = pred_tensor.shape[-1]
                        for epoch_idx in range(n_epochs):
                            try:
                                stage = int(pred_tensor[epoch_idx].item())
                                confidence_raw = float(conf_tensor[stage, epoch_idx].item())
                            except Exception as err:
                                stage = -1
                                confidence_raw = float('nan')
                            rows.append({
                                "Subject"       : subj,
                                "Session"       : sess,
                                "Epoch"         : epoch_idx,
                                "PredictedStage": stage,
                                "Confidence"    : round(confidence_raw, 2)
                            })
                    print(f"    [INFO] Extracted {len(rows)} epochs so far.")

            # Write CSV for this derivation
            safe_name = deriv.replace('<', '').replace('>', '').replace('-', '_')
            filename  = f"{safe_name}_{ckpt_name}_{timestamp}.csv"
            out_path  = os.path.join(out_dir, filename)

            with open(out_path, 'w', newline='') as csvf:
                writer = csv.DictWriter(
                    csvf,
                    fieldnames=["Subject", "Session", "Epoch", "PredictedStage", "Confidence"]
                )
                writer.writeheader()
                writer.writerows(rows)

            print(f"  [INFO] Saved {len(rows)} rows to {out_path}")

if __name__ == "__main__":
    main()
