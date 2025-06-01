import os
import csv
import numpy as np
import mne
import matplotlib.pyplot as plt
from datetime import datetime

# ======================= Configuration =======================
base_dir = r"O:\Tech_NeuroData\Code\Usleep_set_files"  # Base folder containing all subject folders
offset_dir = r"O:\Tech_NeuroData\Code\Offset"          # Folder where the CSV summary will be saved
os.makedirs(offset_dir, exist_ok=True)

# Create a timestamp string for filenames
timestamp = datetime.now().strftime("%d%m_%H%M")

# CSV file now includes the timestamp in its filename
csv_filename = os.path.join(offset_dir, f"results_{timestamp}.csv")
target_sf = 64  # Downsample target in Hz

# Dynamic sliding-window parameters (in seconds)
window_length_sec = 750   # Length of each window
step_sec = 100            # Step size between windows
expected_offset_limit = 400  # Maximum plausible offset (in seconds)

# Default window (if dynamic selection fails)
default_start_sec = 500
default_end_sec = 1500

# Flag to apply dynamic window selection
apply_dynamic_window = True

# ======================= Ear Channel Specifications =======================
ear_channels_spec = {
    # Group 1 – Healthy Controls
    "sub-325": { 
        "ses-01": {
            "left": ["ELB", "ELC", "ELT", "ELE", "ELI"],
            "right": ["ERC", "ERT", "ERE"]
        }
    },
    "sub-562": {
        "ses-01": {
            "left": ["ELC", "ELT", "ELE", "ELI"],
            "right": ["ERB", "ERC", "ERT", "ERE", "ERI"]
        }
    },
    "sub-152": {
        "ses-01": {
            "left": ["ELA", "ELB", "ELC", "ELE", "ELI"],
            "right": ["ERB", "ERC", "ERT", "ERE", "ERI"]
        }
    },
    "sub-957": {
        "ses-01": {
            "left": ["ELB", "ELC", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC", "ERT", "ERE"]
        }
    },
    "sub-777": {
        "ses-01": {
            "left": ["ELT", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC", "ERT", "ERE", "ERI"]
        }
    },
    "sub-817": {
        "ses-01": {
            "left": ["ELB", "ELC", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC", "ERI"]
        }
    },
    "sub-906": {
        "ses-01": {
            "left": ["ELA", "ELB", "ELC", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC", "ERE", "ERI"]
        }
    },
    "sub-869": {
        "ses-01": {
            "left": ["ELA", "ELB", "ELC", "ELT", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC", "ERT", "ERE", "ERI"]
        }
    },
    "sub-843": {
        "ses-01": {
            "left": ["ELA", "ELB", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC"]
        }
    },
    "sub-400": {
        "ses-01": {
            "left": ["ELB", "ELC", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC", "ERE", "ERI"]
        }
    },
    "sub-260": {
        "ses-01": {
            "left": ["ELB", "ELC", "ELE", "ELI"],
            "right": ["ERB", "ERE", "ERI"]
        }
    },
    "sub-142": {
        "ses-01": {
            "left": ["ELB", "ELC", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC", "ERT", "ERE", "ERI"]
        }
    },
    "sub-422": {
        "ses-01": {
            "left": ["ELC", "ELE", "ELI"],
            "right": ["ERA", "ERT", "ERI"]
        }
    },
    "sub-916": {
        "ses-01": {
            "left": ["ELB", "ELC", "ELT", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERT", "ERE", "ERI"]
        }
    },
    # Group 2 – Alzheimer’s Disease
    "sub-582": {
        "ses-01": {
            "left": ["ELA", "ELB", "ELC", "ELT", "ELE", "ELI"],
            "right": ["ERB", "ERC", "ERT", "ERE", "ERI"]
        }
    },
    "sub-852": {
        "ses-01": {
            "left": ["ELB", "ELC", "ELT", "ELE"],
            "right": ["ERB", "ERC", "ERT", "ERE"]
        }
    },
    "sub-298": {
        "ses-01": {
            "left": ["ELA", "ELB", "ELC", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC", "ERT", "ERE", "ERI"]
        },
        "ses-02": {
            "left": ["ELB", "ELC", "ELE"],
            "right": ["ERA", "ERB", "ERC", "ERT", "ERE", "ERI"]
        },
        "ses-03": {
            "left": ["ELA", "ELB", "ELC", "ELE", "ELI"],
            "right": ["ERB", "ERC", "ERE", "ERI"]
        }
    },
    "sub-856": {
        "ses-01": {
            "left": ["ELA", "ELB", "ELC", "ELE", "ELI"],
            "right": ["ERB", "ERC", "ERT", "ERE", "ERI"]
        },
        "ses-02": {
            "left": ["ELA", "ELB", "ELC", "ELE", "ELI"],
            "right": ["ERB", "ERC", "ERT", "ERE", "ERI"]
        }
    },
    "sub-360": {
        "ses-01": {
            "left": ["ELB", "ELC", "ELT", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC", "ERT", "ERI"]
        },
        "ses-02": {
            "left": ["ELC", "ELT", "ELE"],
            "right": ["ERB", "ERC", "ERT", "ERE", "ERI"]
        },
        "ses-03": {
            "left": ["ELB", "ELC", "ELE"],
            "right": ["ERA", "ERB", "ERC", "ERE", "ERI"]
        }
    },
    "sub-749": {
        "ses-01": {
            "left": ["ELB", "ELC", "ELT", "ELE", "ELI"],
            "right": ["ERA", "ERC", "ERT", "ERE", "ERI"]
        },
        "ses-02": {
            "left": ["ELA", "ELB", "ELC", "ELT", "ELE", "ELI"],
            "right": ["ERB", "ERC", "ERT", "ERI"]
        },
        "ses-03": {
            "left": ["ELA", "ELB", "ELC", "ELE", "ELI"],
            "right": ["ERA", "ERC", "ERT", "ERE", "ERI"]
        }
    },
    "sub-462": {
        "ses-01": {
            "left": ["ELA", "ELC", "ELE"],
            "right": ["ERA", "ERB", "ERC", "ERE", "ERI"]
        },
        "ses-02": {
            "left": ["ELB", "ELC", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC", "ERE"]
        },
        "ses-03": {
            "left": ["ELC", "ELT", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC", "ERE", "ERI"]
        }
    },
    "sub-263": {
        "ses-01": {
            "left": ["ELB", "ELC", "ELE"],
            "right": ["ERA", "ERB", "ERC", "ERT", "ERE"]
        },
        "ses-02": {
            "left": ["ELB", "ELC", "ELT", "ELE"],
            "right": ["ERB", "ERT", "ERE", "ERI"]
        },
        "ses-03": {
            "left": ["ELA", "ELB", "ELC", "ELT", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC", "ERT", "ERE"]
        }
    },
    "sub-689": {
        "ses-01": {
            "left": ["ELB", "ELI"],
            "right": ["ERC", "ERE", "ERI"]
        },
        "ses-02": {
            "left": ["ELA", "ELB", "ELC", "ELT", "ELE", "ELI"],
            "right": ["ERB", "ERC", "ERE", "ERI"]
        }
    },
    "sub-913": {
        "ses-01": {
            "left": ["ELA", "ELB", "ELC", "ELE", "ELI"],
            "right": ["ERA", "ERI"]
        },
        "ses-02": {
            "left": ["ELA", "ELB", "ELT", "ELE", "ELI"],
            "right": ["ERA", "ERC", "ERT"]
        }
    },
    "sub-826": {
        "ses-01": {
            "left": ["ELB", "ELC", "ELT"],
            "right": ["ERB", "ERC", "ERT", "ERE", "ERI"]
        }
    },
    "sub-538": {
        "ses-01": {
            "left": ["ELB", "ELT", "ELE", "ELI"],
            "right": ["ERB", "ERE", "ERI"]
        },
        "ses-02": {
            "left": ["ELA", "ELC", "ELE", "ELI"],
            "right": ["ERB", "ERE", "ERI"]
        },
        "ses-03": {
            "left": ["ELA", "ELE"],
            "right": ["ERB", "ERC", "ERE", "ERI"]
        }
    },
    "sub-443": {
        "ses-01": {
            "left": ["ELA", "ELB", "ELC", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERE", "ERI"]
        },
        "ses-02": {
            "left": ["ELA", "ELC", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC", "ERE", "ERI"]
        }
    },
    "sub-962": {
        "ses-01": {
            "left": ["ELB", "ELT", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERT", "ERE", "ERI"]
        }
    },
    "sub-98": {
        "ses-01": {
            "left": ["ELA", "ELB", "ELC", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC", "ERE"]
        }
    },
    "sub-958": {
        "ses-01": {
            "left": ["ELA", "ELB", "ELC", "ELT", "ELE", "ELI"],
            "right": ["ERB", "ERC", "ERT", "ERE"]
        },
        "ses-02": {
            "left": ["ELA", "ELB", "ELC", "ELT", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC", "ERT", "ERE", "ERI"]
        },
        "ses-03": {
            "left": ["ELA", "ELB", "ELC", "ELT", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC", "ERT", "ERE", "ERI"]
        }
    },
    "sub-971": {
        "ses-01": {
            "left": ["ELA", "ELB", "ELC", "ELT", "ELE"],
            "right": ["ERA", "ERB", "ERC", "ERT", "ERE", "ERI"]
        },
        "ses-02": {
            "left": ["ELC", "ELT", "ELE", "ELI"],
            "right": ["ERB", "ERC", "ERT"]
        },
        "ses-03": {
            "left": ["ELC", "ELT", "ELE"],
            "right": ["ERB", "ERC"]
        }
    },
    "sub-800": {
        "ses-01": {
            "left": ["ELC", "ELT", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC", "ERT", "ERE", "ERI"]
        }
    },
    "sub-549": {
        "ses-01": {
            "left": ["ELA", "ELB", "ELC", "ELE"],
            "right": ["ERB", "ERC", "ERT", "ERE"]
        }
    },
    # Group 3 – Dementia with Lewy Bodies
    "sub-849": {
        "ses-01": {
            "left": ["ELB", "ELC", "ELT", "ELE", "ELI"],
            "right": ["ERB", "ERC", "ERT", "ERE", "ERI"]
        },
        "ses-02": {
            "left": ["ELB", "ELT", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERT", "ERE", "ERI"]
        },
        "ses-03": {
            "left": ["ELA", "ELB", "ELC", "ELT", "ELE", "ELI"],
            "right": ["ERB", "ERC", "ERE", "ERI"]
        }
    },
    "sub-815": {
        "ses-01": {
            "left": ["ELB", "ELT", "ELE", "ELI"],
            "right": ["ERB", "ERE", "ERI"]
        }
    },
    "sub-3906": {
        "ses-01": {
            "left": ["ELA", "ELC", "ELT", "ELE"],
            "right": ["ERB", "ERE", "ERI"]
        }
    },
    "sub-3127": {
        "ses-01": {
            "left": ["ELA", "ELB", "ELC", "ELT", "ELE"],
            "right": ["ERA", "ERE", "ERI"]
        },
        "ses-02": {
            "left": ["ELA", "ELB", "ELC", "ELT", "ELE"],
            "right": ["ERA", "ERB", "ERT", "ERE", "ERI"]
        },
        "ses-03": {
            "left": ["ELA", "ELC", "ELT", "ELE", "ELI"],
            "right": ["ERA", "ERC", "ERE"]
        }
    },
    "sub-398": {
        "ses-01": {
            "left": ["ELA", "ELC", "ELT", "ELE"],
            "right": ["ERB", "ERT", "ERE", "ERI"]
        },
        "ses-02": {
            "left": ["ELA", "ELC", "ELT", "ELE"],
            "right": ["ERB", "ERT", "ERE", "ERI"]
        },
        "ses-03": {
            "left": ["ELC", "ELE", "ELI"],
            "right": ["ERT", "ERE", "ERI"]
        }
    },
    "sub-3958": {
        "ses-01": {
            "left": ["ELB", "ELC", "ELE", "ELI"],
            "right": ["ERA", "ERC", "ERE", "ERI"]
        },
        "ses-02": {
            "left": ["ELA", "ELC", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC", "ERE"]
        }
    },
    "sub-3971": {
        "ses-01": {
            "left": ["ELA", "ELC", "ELE", "ERA"],
            "right": ["ERB", "ERC", "ERE", "ERI"]
        },
        "ses-02": {
            "left": ["ELA", "ELE", "ELI", "ERA"],
            "right": ["ERB", "ERE"]
        }
    },
    "sub-485": {
        "ses-01": {
            "left": ["ELE"],
            "right": ["ERB", "ERC", "ERE", "ERI"]
        },
        "ses-02": {
            "left": ["ELA", "ELB", "ELE"],
            "right": ["ERA", "ERB", "ERC", "ERT", "ERE", "ERI"]
        },
        "ses-03": {
            "left": ["ELB", "ELE"],
            "right": ["ERB", "ERC", "ERE", "ERI"]
        }
    },
    "sub-3422": {
        "ses-01": {
            "left": ["ELB", "ELC", "ELE"],
            "right": ["ERA", "ERB", "ERC", "ERE", "ERI"]
        },
        "ses-02": {
            "left": ["ELA", "ELB", "ELC", "ELE"],
            "right": ["ERA", "ERB", "ERC", "ERT", "ERE"]
        },
        "ses-03": {
            "left": ["ELA", "ELB", "ELC", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC", "ERE", "ERI"]
        }
    },
    "sub-959": {
        "ses-01": {
            "left": ["ELB", "ELC", "ELT", "ELE", "ELI"],
            "right": ["ERA", "ERB", "ERC", "ERT", "ERE", "ERI"]
        },
        "ses-02": {
            "left": ["ELB", "ELC", "ELT", "ELE", "ELI"],
            "right": ["ERB", "ERC", "ERI"]
        },
        "ses-03": {
            "left": ["ELC", "ELT", "ELE", "ELI"],
            "right": ["ERB", "ERT", "ERE", "ERI"]
        }
    }
}

# ======================= Helper Function: Dynamic Window Selection =======================
def select_dynamic_window(scalp_signal_full, ear_signal_full, scalp_times, target_sf,
                          window_length_sec, step_sec, expected_offset_limit,
                          default_start_sec, default_end_sec):
    window_length_samples = int(window_length_sec * target_sf)
    step_samples = int(step_sec * target_sf)
    num_samples = len(scalp_signal_full)
    
    candidate_qualities = []
    candidate_offsets = []
    candidate_windows = []
    
    for start_sample in range(0, num_samples - window_length_samples, step_samples):
        end_sample = start_sample + window_length_samples
        scalp_segment = scalp_signal_full[start_sample:end_sample].copy()
        ear_segment = ear_signal_full[start_sample:end_sample].copy()
        
        quality = np.std(scalp_segment) + np.std(ear_segment)
        corr_seg = np.correlate(scalp_segment, ear_segment, mode='full')
        lags_seg = np.arange(-len(scalp_segment) + 1, len(ear_segment))
        lag_seconds_seg = lags_seg / target_sf
        max_corr_index_seg = np.argmax(corr_seg)
        offset_seg = lag_seconds_seg[max_corr_index_seg]
        
        if np.abs(offset_seg) <= expected_offset_limit:
            candidate_qualities.append(quality)
            candidate_offsets.append(offset_seg)
            candidate_windows.append((start_sample, end_sample))
    
    if candidate_qualities:
        best_idx = np.argmin(candidate_qualities)
        best_window = candidate_windows[best_idx]
        best_offset = candidate_offsets[best_idx]
        return best_window, best_offset
    else:
        start_sample = int(default_start_sec * target_sf)
        end_sample = int(default_end_sec * target_sf)
        return (start_sample, end_sample), None

# ======================= Main Processing Loop =======================
results = []

for subj in sorted(ear_channels_spec.keys()):
    try:
        subj_path = os.path.join(base_dir, subj)
        
        # Loop over sessions for the subject
        for sess in sorted(ear_channels_spec[subj].keys()):
            session_path = os.path.join(subj_path, sess)
            
            # Construct file paths
            scalp_file = os.path.join(subj_path, sess, f"{subj}_{sess}_task-sleep_eeg.set")
            ear_file = os.path.join(subj_path, sess, f"{subj}_{sess}_task-sleep_eeg.edf")
            
            if not os.path.exists(scalp_file) or not os.path.exists(ear_file):
                raise FileNotFoundError(f"Missing scalp or ear file for {subj} {sess}.")
            
            print(f"\nProcessing {subj} {sess}...")
            
            # ---------------------- Load Scalp EEG ----------------------
            raw_scalp = mne.io.read_raw_eeglab(scalp_file, uint16_codec='utf-16', preload=True)
            raw_scalp.filter(l_freq=0.5, h_freq=40, fir_design='firwin')
            raw_scalp.resample(target_sf, npad="auto")
            
            # ---------------------- Load Ear EEG ----------------------
            raw_ear = mne.io.read_raw_edf(ear_file, preload=True, verbose=False)
            raw_ear.resample(target_sf, npad="auto")
            
            # ---------------------- Channel Selection ----------------------
            scalp_diff_channels = [ch for ch in ['T7-REF', 'T8-REF'] if ch in raw_scalp.ch_names]
            if len(scalp_diff_channels) < 2:
                raise ValueError(f"Scalp channels T7-REF/T8-REF not found for {subj} {sess}.")
            
            spec = ear_channels_spec[subj].get(sess)
            if spec is None:
                raise ValueError(f"No ear channel specification for {subj} {sess}.")
            
            left_ear_channels = [ch for ch in spec["left"] if ch in raw_ear.ch_names]
            right_ear_channels = [ch for ch in spec["right"] if ch in raw_ear.ch_names]
            if not left_ear_channels or not right_ear_channels:
                raise ValueError(f"Insufficient ear channels for {subj} {sess}.")
            
            # ---------------------- Extract Data ----------------------
            scalp_data, scalp_times = raw_scalp.get_data(picks=scalp_diff_channels, return_times=True)
            scalp_signal_full = scalp_data[0] - scalp_data[1]
            
            ear_data_left = raw_ear.get_data(picks=left_ear_channels)
            ear_data_right = raw_ear.get_data(picks=right_ear_channels)
            ear_signal_left = np.mean(ear_data_left, axis=0)
            ear_signal_right = np.mean(ear_data_right, axis=0)
            ear_signal_full = ear_signal_left - ear_signal_right
            
            # ---------------------- Dynamic Window Selection ----------------------
            if apply_dynamic_window:
                (win_start, win_end), window_offset = select_dynamic_window(
                    scalp_signal_full, ear_signal_full, scalp_times, target_sf,
                    window_length_sec, step_sec, expected_offset_limit,
                    default_start_sec, default_end_sec
                )
                if window_offset is not None:
                    print(f"Selected window from {win_start/target_sf:.1f} s to {win_end/target_sf:.1f} s")
                    print(f"Cross-correlation offset in this window: {window_offset:.3f} s")
                    selected_offset = window_offset
                else:
                    print("No candidate window found; using default window.")
                    win_start = int(default_start_sec * target_sf)
                    win_end = int(default_end_sec * target_sf)
                    selected_offset = None
            else:
                win_start = int(default_start_sec * target_sf)
                win_end = int(default_end_sec * target_sf)
                selected_offset = None
            
            scalp_signal = scalp_signal_full[win_start:win_end]
            ear_signal = ear_signal_full[win_start:win_end]
            
            # ---------------------- Cross-Correlation Computation ----------------------
            corr = np.correlate(scalp_signal, ear_signal, mode='full')
            lags = np.arange(-len(scalp_signal) + 1, len(ear_signal))
            lag_seconds = lags / target_sf
            max_corr_index = np.argmax(corr)
            final_offset = lag_seconds[max_corr_index]
            
            if final_offset < 0:
                alignment = "Ear EEG starts earlier than Scalp EEG"
            elif final_offset > 0:
                alignment = "Scalp EEG starts earlier than Ear EEG"
            else:
                alignment = "Signals are aligned (offset is 0)"
            
            print("\nFinal Cross-Correlation Results:")
            print(f"Offset (seconds): {final_offset:.3f}")
            print(alignment)
            
            # ---------------------- Save Cross-Correlation Plot ----------------------
            plt.figure(figsize=(10, 5))
            plt.plot(lag_seconds, corr, label="Cross-Correlation")
            plt.title(f"Cross-Correlation: {subj} {sess}")
            plt.xlabel("Lag (s)")
            plt.ylabel("Correlation")
            plt.legend()
            plt.tight_layout()
            
            # The plot filename now includes the timestamp
            plot_filename = os.path.join(session_path, f"{subj}_{sess}_crosscorr_{timestamp}.png")
            plt.savefig(plot_filename)
            plt.close()
            print(f"Cross-correlation plot saved to {plot_filename}")
            
            # ---------------------- Record Results ----------------------
            results.append({
                "Subject": subj,
                "Session": sess,
                "Selected_Window_Start_sec": win_start / target_sf,
                "Selected_Window_End_sec": win_end / target_sf,
                "CrossCorr_Offset_sec": final_offset,
                "Alignment": alignment
            })
    
    except Exception as e:
        print(f"Error encountered with subject {subj}: {e}")
        print("Skipping this subject.")
        continue

# ======================= Write Results to CSV =======================
csv_fields = [
    "Subject",
    "Session",
    "Selected_Window_Start_sec",
    "Selected_Window_End_sec",
    "CrossCorr_Offset_sec",
    "Alignment"
]
with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\nAll results have been written to {csv_filename}")
