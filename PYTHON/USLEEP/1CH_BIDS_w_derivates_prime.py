import os
import csv
import warnings
import glob
import re
from datetime import datetime
import importlib
import sys
import contextlib
import numpy

import mne
import mne.io.eeglab.eeglab as eeglab_mod
from csdp_training.experiments.bids_predictor import BIDS_USleep_Predictor

print("\n==== Script started ====\n")

# ----------------------------------------------------------------------------
# Suppress boundary/deprecation warnings, restrict MNE logs to errors
# ----------------------------------------------------------------------------
warnings.filterwarnings(
    'ignore',
    message='.*boundary events.*',
    category=RuntimeWarning
)
warnings.filterwarnings('ignore', category=DeprecationWarning)
mne.set_log_level('ERROR')

# ----------------------------------------------------------------------------
# Patch MNE EEGLAB loader for UTF-16 and auto-apply bipolar referencing
# ----------------------------------------------------------------------------
def patch_mne_eeglab_utf16_and_bipolar():
    mod = importlib.reload(eeglab_mod)
    _orig_check = mod._check_load_mat

    def patched_check_load_mat(fname, *args, **kwargs):
        return _orig_check(fname, "utf-16")
    mod._check_load_mat = patched_check_load_mat

    from mne.io.eeglab.eeglab import RawEEGLAB as OrigRaw
    _orig_init = OrigRaw.__init__
    def patched_init(self, input_fname, preload=False, uint16_codec="utf-16", verbose=None, **kwargs):
        return _orig_init(self, input_fname, preload=preload, uint16_codec="utf-16", verbose=verbose, **kwargs)
    OrigRaw.__init__ = patched_init
    mod.RawEEGLAB.__init__ = patched_init

    # Patch annotation reading
    from mne.annotations import read_annotations as orig_read_ann
    def patched_read_annotations(fname, sfreq="auto", *args, **kwargs):
        return orig_read_ann(fname, sfreq=sfreq, uint16_codec="utf-16")
    mne.annotations.read_annotations = patched_read_annotations

    # Patch raw loader to apply bipolar referencing automatically
    orig_read = mne.io.read_raw_eeglab
    def patched_read_raw_eeglab(fname, *args, **kwargs):
        raw = orig_read(fname, *args, **kwargs)
        anodes = ['T7-REF', 'C3-REF', 'P3-REF', 'F7-REF', 'F3-REF']
        cathodes = ['T8-REF', 'C4-REF', 'P4-REF', 'F8-REF', 'F4-REF']
        derivs = ['T7-T8', 'C3-C4', 'P3-P4', 'F7-F8', 'F3-F4']
        local_anodes = []
        local_cathodes = []
        local_derivs = []
        for a, c, d in zip(anodes, cathodes, derivs):
            if a in raw.info['ch_names'] and c in raw.info['ch_names']:
                local_anodes.append(a)
                local_cathodes.append(c)
                local_derivs.append(d)
        if local_anodes:
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                bipolar = mne.set_bipolar_reference(
                    inst=raw, anode=local_anodes, cathode=local_cathodes,
                    ch_name=local_derivs, drop_refs=False, copy=True
                )
                bipolar.pick(local_derivs)
            return bipolar
        else:
            return raw
    mne.io.read_raw_eeglab = patched_read_raw_eeglab

# ----------------------------------------------------------------------------
# Main processing logic
# ----------------------------------------------------------------------------
def main():
    checkpoint = "/home/au682014/eeglab_project/Weights/m1m2_auhcologne_onechannel_lightning.ckpt"
    data_directory = "/home/au682014/eeglab_project/OUS/Usleep_files_resampled128Hz_cropped"
    extension = ".set"
    task = "sleep"
    output_dir = "/home/au682014/eeglab_project/OUS_files/sleepstages_scalp"

    # Find all .set files under data_directory
    set_pattern = os.path.join(data_directory, "sub-*", "ses-*", "sub-*_ses-*_task-sleep_eeg.set")
    all_set_files = glob.glob(set_pattern)

    # Extract subject and session using regex
    set_regex = re.compile(r"sub-(\d+)/ses-(\d+)/sub-(\d+)_ses-(\d+)_task-sleep_eeg\.set$")

    file_entries = []
    for file_path in all_set_files:
        match = set_regex.search(file_path)
        if match:
            subject = match.group(1)
            session = match.group(2)
            file_entries.append((subject, session, file_path))

    print(f"[Info] Detected {len(file_entries)} files to process.")

    config_to_channels = {
        'T7T8': ['T7-T8'],
        'C3C4': ['C3-C4'],
        'P3P4': ['P3-P4'],
        'F7F8': ['F7-F8'],
        'F3F4': ['F3-F4']
        # Optionally: 'ALL': ['T7-T8', 'C3-C4', 'P3-P4', 'F7-F8', 'F3-F4']
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    for config_name, channel_list in config_to_channels.items():
        print(f"\n[Step] Processing channel configuration: {config_name}")
        rows = []
        for subject, session, set_file_path in file_entries:
            print(f"[Debug] Checking file: {set_file_path}")
            try:
                raw = mne.io.read_raw_eeglab(set_file_path, preload=True)
                if channel_list[0] in raw.info['ch_names']:
                    arr = raw.copy().pick(channel_list).get_data()
                    has_nan = numpy.isnan(arr).any()
                    is_all_zero = numpy.all(arr == 0)
                    print(f"[Debug] {channel_list[0]}: shape={arr.shape}, min={arr.min()}, max={arr.max()}, mean={arr.mean()}, std={arr.std()}, any NaN={has_nan}, all zero={is_all_zero}")
                else:
                    print(f"[Warning] {channel_list[0]} NOT FOUND in raw data channels for {set_file_path}")
                    continue
            except Exception as err:
                print(f"[Error] Could not load or check data for {channel_list[0]} ({set_file_path}): {err}")
                continue

            # Prediction step
            try:
                predictor = BIDS_USleep_Predictor(
                    data_dir=data_directory,
                    data_extension=extension,
                    data_task=task,
                    subjects=[subject],
                    sessions=[session]
                )
                dataset = predictor.build_dataset(channel_list)
            except Exception as err:
                print(f"[Error] Error building dataset: {err}")
                continue

            if not dataset:
                print(f"[Warning] No data returned by build_dataset for {set_file_path}, skipping prediction.")
                continue
            try:
                preds_list, confs_list = predictor.predict_all(checkpoint, dataset)
                print("[Debug] Prediction completed.")
            except Exception as err:
                print(f"[Error] Error during prediction: {err}")
                continue

            for i, (pred_tensor, conf_tensor) in enumerate(zip(preds_list, confs_list)):
                n_epochs = pred_tensor.shape[-1]
                for epoch_idx in range(n_epochs):
                    try:
                        stage = int(pred_tensor[epoch_idx].item())
                        confidence_raw = float(conf_tensor[stage, epoch_idx].item())
                    except Exception as err:
                        print(f"[Error] Prediction/confidence indexing failed for epoch {epoch_idx}: {err}")
                        stage = -1
                        confidence_raw = float('nan')
                    rows.append({
                        "Subject": f"sub-{subject}",
                        "Session": session,
                        "Epoch": epoch_idx,
                        "PredictedStage": stage,
                        "Confidence": round(confidence_raw, 2)
                    })
            print(f"[Debug] {len(rows)} rows for subject/session/config so far.")

        # Write output CSV per config
        config_dir = os.path.join(output_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)
        filename = f"{config_name}_{timestamp}.csv"
        out_path = os.path.join(config_dir, filename)
        try:
            with open(out_path, "w", newline="") as out_file:
                writer = csv.DictWriter(
                    out_file,
                    fieldnames=["Subject", "Session", "Epoch", "PredictedStage", "Confidence"]
                )
                writer.writeheader()
                writer.writerows(rows)
            print(f"[Info] Saved {len(rows)} rows to {out_path}")
        except Exception as err:
            print(f"[Error] Failed to write CSV for {config_name}: {err}")

    print("\n==== Script complete ====\n")

if __name__ == "__main__":
    patch_mne_eeglab_utf16_and_bipolar()
    main()
