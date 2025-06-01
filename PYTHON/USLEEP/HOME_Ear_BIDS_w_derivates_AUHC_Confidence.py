import os
import csv
import warnings
import sys
import contextlib
from datetime import datetime
import numpy as np
import mne
from csdp_training.experiments.bids_predictor import BIDS_USleep_Predictor

print("\n==== Script started ====\n")

# ----------------------------------------------------------------------------
# Suppress non-critical warnings and suppress torchvision's userwarning
# ----------------------------------------------------------------------------
print("[Info] Suppressing warnings...")
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*Failed to load image Python extension.*",
        category=UserWarning,
        module="torchvision"
    )
    try:
        import torch
        import torchvision
    except ImportError:
        print("[Warning] torch/torchvision not installed, skipping import.")

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
mne.set_log_level('ERROR')

# ----------------------------------------------------------------------------
# Patch MNE to apply all 36 ear bipolar derivations
# ----------------------------------------------------------------------------
def patch_mne_eeglab_ear_bipolar():
    print("[Debug] Patching mne.io.read_raw_eeglab for 36 ear derivations...")
    original_read_raw_eeglab = mne.io.read_raw_eeglab

    left_ear_channels = ['ELA', 'ELB', 'ELC', 'ELT', 'ELE', 'ELI']
    right_ear_channels = ['ERA', 'ERB', 'ERC', 'ERT', 'ERE', 'ERI']
    deriv_names = [f"{rc}-{lc}" for rc in right_ear_channels for lc in left_ear_channels]

    def patched_read_raw_eeglab(fname, *args, **kwargs):
        print(f"[Debug] Loading raw EEGLAB file: {fname}")
        raw = original_read_raw_eeglab(fname, *args, **kwargs)
        print(f"[Debug] Raw loaded. Available channels: {raw.info['ch_names']}")
        anodes = []
        cathodes = []
        derivs = []
        for rc in right_ear_channels:
            for lc in left_ear_channels:
                if rc in raw.info['ch_names'] and lc in raw.info['ch_names']:
                    anodes.append(rc)
                    cathodes.append(lc)
                    derivs.append(f"{rc}-{lc}")
        print(f"[Debug] Number of derivations to create: {len(derivs)}")
        if len(anodes) > 0:
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                bipolar = mne.set_bipolar_reference(
                    inst      = raw,
                    anode     = anodes,
                    cathode   = cathodes,
                    ch_name   = derivs,
                    drop_refs = False,
                    copy      = True
                )
                bipolar.pick(derivs)
            print(f"[Debug] Bipolar derivations added. Channels now: {bipolar.info['ch_names']}")
            return bipolar
        else:
            print("[Debug] No valid anode/cathode pairs found, returning raw unchanged.")
            return raw

    mne.io.read_raw_eeglab = patched_read_raw_eeglab
    print("[Debug] Patching complete.\n")

def main():
    print("[Step] Main pipeline starting...")
    checkpoint = "/home/au682014/eeglab_project/Weights/m1m2_auhcologne_onechannel_lightning.ckpt"
    data_directory = "/home/au682014/eeglab_project/set_files"
    extension      = ".set"
    task           = "sleep"
    output_dir     = "/home/au682014/eeglab_project/new_sleepstages"

    subjects = [
    "17829", "26578", "29732", "23650","22073", "22987", "21235",
    "23608", "36821", "39201", "25634", "21753",
    "24390", "26530", "22964", "23842", "23152", "20619", "19742", "31976",
    "13167", "22176", "15749", "14890", "20347", "29708", "21388", "14360",
    "37915", "30985", "20365", "22519", "37790", "25316", "23906"
    ]


    sessions = ["01"]

    left_ear_channels  = ['ELA', 'ELB', 'ELC', 'ELT', 'ELE', 'ELI']
    right_ear_channels = ['ERA', 'ERB', 'ERC', 'ERT', 'ERE', 'ERI']
    all_deriv_names = [f"{rc}-{lc}" for rc in right_ear_channels for lc in left_ear_channels]
    deriv_names = all_deriv_names  # Use all 36 derivations

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    for subject in subjects:
        for session in sessions:
            set_file_path = os.path.join(
                data_directory,
                f"sub-{subject}",
                f"ses-{session}",
                f"sub-{subject}_ses-{session}_task-sleep_eeg.set"
            )
            print(f"[Debug] Will check data in: {set_file_path}")

            for deriv_name in deriv_names:
                print(f"\n[Step] Processing derivation: {deriv_name}")
                rows = []
                # ----- Debug: Check derivation data before running predictor -----
                skip_derivation = False
                try:
                    raw = mne.io.read_raw_eeglab(set_file_path, preload=True)
                    if deriv_name in raw.info['ch_names']:
                        arr = raw.copy().pick([deriv_name]).get_data()
                        print(f"[Debug] {deriv_name}: shape={arr.shape}, min={arr.min()}, max={arr.max()}, mean={arr.mean()}, std={arr.std()}, any NaN={np.isnan(arr).any()}, all zero={np.all(arr==0)}")
                    else:
                        print(f"[Debug] {deriv_name} NOT FOUND in raw data channels.")
                        skip_derivation = True
                except Exception as err:
                    print(f"[Error] Could not load or check data for {deriv_name}: {err}")
                    skip_derivation = True

                if skip_derivation:
                    continue  # Skip to next derivation if data file missing or derivation not present

                # -------- Predictor Pipeline ---------
                predictor = BIDS_USleep_Predictor(
                    data_dir       = data_directory,
                    data_extension = extension,
                    data_task      = task,
                    subjects       = [subject],
                    sessions       = [session]
                )
                print("[Debug] Predictor instantiated.")
                try:
                    dataset = predictor.build_dataset([deriv_name])
                    print("[Debug] Dataset built successfully.")
                except Exception as err:
                    print(f"[Error] Error building dataset: {err}")
                    continue
                if not dataset:
                    print("[Warning] No data returned by build_dataset, skipping prediction.")
                    continue
                try:
                    preds_list, confs_list = predictor.predict_all(checkpoint, dataset)
                    print("[Debug] Prediction completed.")
                except Exception as err:
                    print(f"[Error] Error during prediction: {err}")
                    continue

                for i, (pred_tensor, conf_tensor) in enumerate(zip(preds_list, confs_list)):
                    print(f"[Debug] Prediction tensor shape: {getattr(pred_tensor, 'shape', None)}, Confidence tensor shape: {getattr(conf_tensor, 'shape', None)}")
                    n_epochs = pred_tensor.shape[-1]
                    print(f"[Debug] n_epochs: {n_epochs}, Unique stages: {np.unique(pred_tensor)}")
                    if hasattr(conf_tensor, "min") and hasattr(conf_tensor, "max"):
                        print(f"[Debug] Conf min: {conf_tensor.min()}, max: {conf_tensor.max()}, any NaN: {np.isnan(conf_tensor).any()}")
                    for epoch_idx in range(n_epochs):
                        try:
                            stage = int(pred_tensor[epoch_idx].item())
                            confidence_raw = float(conf_tensor[stage, epoch_idx].item())
                        except Exception as err:
                            print(f"[Error] Prediction/confidence indexing failed for epoch {epoch_idx}: {err}")
                            stage = -1
                            confidence_raw = float('nan')
                        rows.append({
                            "Subject"        : f"sub-{subject}",
                            "Session"        : session,
                            "Epoch"          : epoch_idx,
                            "PredictedStage" : stage,
                            "Confidence"     : round(confidence_raw, 2)
                        })
                print(f"[Debug] {len(rows)} rows for subject/session/derivation.")

                # Create output directory per subject/session
                subject_session_dir = os.path.join(output_dir, f"sub-{subject}", f"ses-{session}")
                os.makedirs(subject_session_dir, exist_ok=True)
                safe_name = deriv_name.replace('-', '_')
                filename  = f"{safe_name}_{timestamp}.csv"
                out_path  = os.path.join(subject_session_dir, filename)
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
                    print(f"[Error] Failed to write CSV for {deriv_name}: {err}")

    print("\n==== Script complete ====\n")

if __name__ == "__main__":
    patch_mne_eeglab_ear_bipolar()
    main()
