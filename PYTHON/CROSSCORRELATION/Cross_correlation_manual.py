import numpy as np 
import mne
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Subject and Session Configuration
# ------------------------------------------------------------------
subject_id = "sub-400"
subject_ses = "01"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
target_sf = 64                 # Downsample target: 64 Hz

# Flag to automatically select a time window via sliding window analysis
apply_dynamic_window = True

# Parameters for sliding window analysis (in seconds)
window_length_sec = 750    # Length of each window (e.g., 750 s)
step_sec = 100             # Step size between windows (e.g., 100 s)
expected_offset_limit = 400  # Maximum expected offset (in seconds) for a candidate window

# If not using dynamic window, these defaults will be used:
default_start_sec = 500
default_end_sec   = 1500

# ------------------------------------------------------------------
# File paths
# ------------------------------------------------------------------
scalp_file = rf"O:\Tech_NeuroData\Code\All_Usleep_files\{subject_id}\ses-{subject_ses}\{subject_id}_ses-{subject_ses}_task-sleep_eeg.set"
ear_file   = rf"O:\Tech_NeuroData\Code\All_Usleep_files\{subject_id}\ses-{subject_ses}\{subject_id}_ses-{subject_ses}_task-sleep_eeg.edf"

# ------------------------------------------------------------------
# Load Scalp EEG
# ------------------------------------------------------------------
raw_scalp = mne.io.read_raw_eeglab(scalp_file, uint16_codec='utf-16', preload=True)
print("Scalp EEG info before resampling:")
print(raw_scalp.info)

# (Optional) Bandpass filter before resampling (helps avoid aliasing)
raw_scalp.filter(l_freq=0.5, h_freq=40, fir_design='firwin')

# Downsample to target_sf
raw_scalp.resample(target_sf, npad="auto")
print("Scalp EEG info after resampling:")
print(raw_scalp.info)

# ------------------------------------------------------------------
# Load Ear EEG
# ------------------------------------------------------------------
raw_ear = mne.io.read_raw_edf(ear_file, preload=True)
print("Ear EEG info before resampling:")
print(raw_ear.info)

# (Optional) Bandpass filter before resampling (helps avoid aliasing)
# raw_ear.filter(l_freq=0.5, h_freq=40, fir_design='firwin')

raw_ear.resample(target_sf, npad="auto")
print("Ear EEG info after resampling:")
print(raw_ear.info)

# ------------------------------------------------------------------
# Channel Selection
# ------------------------------------------------------------------
# 1) Scalp difference: T7-REF minus T8-REF
scalp_diff_channels = ['C3-REF', 'C4-REF']
scalp_diff_channels = [ch for ch in scalp_diff_channels if ch in raw_scalp.ch_names]
if len(scalp_diff_channels) < 2:
    raise ValueError("Could not find both T7-REF and T8-REF in scalp data.")

# 2) Ear channels split by left vs. right
left_ear_channels = ["ELB", "ELC", "ELE"]
right_ear_channels = ["ERA","ERB","ERC","ERT", "ERE", "ERI"]
left_ear_channels  = [ch for ch in left_ear_channels if ch in raw_ear.ch_names]
right_ear_channels = [ch for ch in right_ear_channels if ch in raw_ear.ch_names]
if not left_ear_channels:
    raise ValueError("No usable left ear channels found in the data.")
if not right_ear_channels:
    raise ValueError("No usable right ear channels found in the data.")

# ------------------------------------------------------------------
# Extract Data and Compute Differences
# ------------------------------------------------------------------
# Scalp difference: T7 minus T8
scalp_data, scalp_times = raw_scalp.get_data(picks=scalp_diff_channels, return_times=True)
scalp_signal_full = scalp_data[0] - scalp_data[1]  # T7 - T8

# Ear difference: (Average of left channels) minus (Average of right channels)
ear_data_left = raw_ear.get_data(picks=left_ear_channels)   # shape (n_left, n_samples)
ear_data_right = raw_ear.get_data(picks=right_ear_channels) # shape (n_right, n_samples)
ear_signal_left  = np.mean(ear_data_left, axis=0)   # average across left channels
ear_signal_right = np.mean(ear_data_right, axis=0)  # average across right channels
ear_signal_full  = ear_signal_left - ear_signal_right

# ------------------------------------------------------------------
# Dynamic Time Window Selection via Sliding Window Analysis
# ------------------------------------------------------------------
if apply_dynamic_window:
    # Convert window parameters from seconds to samples
    window_length_samples = int(window_length_sec * target_sf)
    step_samples = int(step_sec * target_sf)
    num_samples = len(scalp_signal_full)
    
    candidate_qualities = []  # Lower values indicate fewer artifacts
    candidate_offsets = []    # Cross-correlation offset (in seconds)
    candidate_windows = []    # Tuples (start_sample, end_sample)
    
    # Slide a window over the entire signal
    for start_sample in range(0, num_samples - window_length_samples, step_samples):
        end_sample = start_sample + window_length_samples
        
        # Extract segments for this window
        scalp_segment = scalp_signal_full[start_sample:end_sample].copy()
        ear_segment   = ear_signal_full[start_sample:end_sample].copy()
        
        # Compute a simple quality metric: sum of standard deviations of the two signals.
        # (High variability may indicate artifacts.)
        quality = np.std(scalp_segment) + np.std(ear_segment)
        
        # Compute cross-correlation for the window
        corr_seg = np.correlate(scalp_segment, ear_segment, mode='full')
        lags_seg = np.arange(-len(scalp_segment) + 1, len(ear_segment))
        lag_seconds_seg = lags_seg / target_sf
        max_corr_index_seg = np.argmax(corr_seg)
        offset_seg = lag_seconds_seg[max_corr_index_seg]
        
        # Only consider candidate windows with an offset within the expected range
        if np.abs(offset_seg) <= expected_offset_limit:
            candidate_qualities.append(quality)
            candidate_offsets.append(offset_seg)
            candidate_windows.append((start_sample, end_sample))
    
    # If one or more candidate windows are found, select the one with the lowest quality metric
    if len(candidate_qualities) > 0:
        best_idx = np.argmin(candidate_qualities)
        best_window = candidate_windows[best_idx]
        best_offset = candidate_offsets[best_idx]
        print(f"Selected window from {best_window[0] / target_sf:.1f} s to {best_window[1] / target_sf:.1f} s")
        print(f"Cross-correlation offset in this window: {best_offset:.3f} s")
        
        # Crop signals based on the selected window
        scalp_signal = scalp_signal_full[best_window[0]:best_window[1]]
        ear_signal   = ear_signal_full[best_window[0]:best_window[1]]
        scalp_times  = scalp_times[best_window[0]:best_window[1]]
    else:
        print("No candidate window found that meets criteria. Falling back to default window.")
        start_sample = int(default_start_sec * target_sf)
        end_sample   = int(default_end_sec * target_sf)
        scalp_signal = scalp_signal_full[start_sample:end_sample]
        ear_signal   = ear_signal_full[start_sample:end_sample]
        scalp_times  = scalp_times[start_sample:end_sample]
else:
    # Use the default fixed window if dynamic selection is not applied
    start_sample = int(default_start_sec * target_sf)
    end_sample   = int(default_end_sec * target_sf)
    scalp_signal = scalp_signal_full[start_sample:end_sample]
    ear_signal   = ear_signal_full[start_sample:end_sample]
    scalp_times  = scalp_times[start_sample:end_sample]

# ------------------------------------------------------------------
# Cross-Correlation Computation on the Selected Window
# ------------------------------------------------------------------
corr = np.correlate(scalp_signal, ear_signal, mode='full')
lags = np.arange(-len(scalp_signal) + 1, len(ear_signal))
lag_seconds = lags / target_sf

max_corr_index = np.argmax(corr)
offset_seconds = lag_seconds[max_corr_index]

print("\nFinal Cross-Correlation Results:")
print(f"Offset (seconds): {offset_seconds:.3f}")
if offset_seconds < 0:
    print("Ear EEG starts earlier than the Scalp EEG.")
elif offset_seconds > 0:
    print("Scalp EEG starts earlier than the Ear EEG.")
else:
    print("Signals are aligned (offset is 0).")

# ------------------------------------------------------------------
# Plot Cross-Correlation
# ------------------------------------------------------------------

plt.rcParams.update({
    # disable external TeX
    "text.usetex": False,
    # use Computer Modern (“cm”) math font via MathText
    "mathtext.fontset": "cm",
    # and set serif family to Computer Modern Roman
    "font.family": "serif",
})

plt.figure(figsize=(10, 5))
plt.plot(lag_seconds, corr, label="Cross-Correlation")
#plt.title("Cross-Correlation: (T7 - T8) vs. [LeftEar - RightEar]")
plt.xlabel("Lag (s)")
plt.ylabel("Correlation")
plt.legend()
plt.tight_layout()
plt.show()
