import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, filtfilt, find_peaks
from collections import Counter
import pandas as pd

#get ECG signal data from database
# 1. LOAD MIT-BIH RECORD VIA wfdb
record = wfdb.rdrecord('108', pn_dir = 'mitdb', sampto=90*360)
fs = record.fs #sampling freq for MIT-BIH record, in Hz
assert fs == 360, f"Expected 360 Hz, got {fs} Hz" 

print(record.sig_name)
signal = record.p_signal[:, 0]

#raw ECG signal
t = np.arange(len(signal)) / fs
plt.figure(figsize=(12,4))
plt.plot(t, signal, linewidth=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('Raw ECG Signal - Record 100, Lead MLII')
plt.grid(alpha=0.3) 

#bandpass filter (0.5-45 Hz) for clean visualization
low_cutoff_freq = 0.5
high_cutoff_freq = 45.0

nyquist = fs/2
low = low_cutoff_freq/nyquist
high = high_cutoff_freq/nyquist
order = 4
b, a = butter(order, [low, high], btype = 'band')
filtered_signal = filtfilt(b, a, signal)

t2 = np.arange(len(filtered_signal)) / fs
plt.figure(2, figsize=(12,4))
plt.plot(t2, filtered_signal, linewidth=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('Filtered ECG Signal - Record 100, Lead MLII')
plt.grid(alpha=0.3)

# QRS-BAND FILTER (5-15 Hz) FOR DETECTION
low_cutoff_freq2 = 5.0
high_cutoff_freq2 = 15.0

nyquist2 = fs/2
low2 = low_cutoff_freq2/nyquist2
high2 = high_cutoff_freq2/nyquist2
b2, a2 = butter(order, [low2, high2], btype = 'band')
filtered_signal2 = filtfilt(b2, a2, signal)

t3 = np.arange(len(filtered_signal2)) / fs
plt.figure(3, figsize=(12,4))
plt.plot(t3, filtered_signal2, linewidth=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('Filtered QRS Signal - Record 100, Lead MLII')
plt.grid(alpha=0.3)
plt.show()

#take derivative of signals and square them to see difference
# DIFFERENTIATE -> SQUARE -> INTEGRATE (Pan-Tompkins transform stages)
diff_signal = np.diff(filtered_signal2)
squared_signal = diff_signal ** 2
window_size = int(0.15 * fs)  # 150 ms window
integrated_signal = np.convolve(squared_signal, np.ones(window_size), mode = 'same')

t_diff = np.arange(len(diff_signal)) / fs
fig, axes = plt.subplots(4, 1, figsize=(12,9), sharex=True)
axes[0].plot(t2, filtered_signal2, linewidth=0.8)
axes[0].set_title('QRS-band Filtered Signal (5-15 Hz)')
axes[0].set_ylabel('Amplitude (mV)')
axes[0].grid(alpha=0.3)

axes[1].plot(t_diff, diff_signal, linewidth=0.8, color='orange')
axes[1].set_title('Differentiated Signal')
axes[1].set_ylabel('Slope')
axes[1].grid(alpha=0.3)

axes[2].plot(t_diff, squared_signal, linewidth=0.8, color='green')
axes[2].set_title('Squared Signal')
axes[2].set_ylabel('Slope')
axes[2].grid(alpha=0.3)

axes[3].plot(t_diff, integrated_signal, linewidth=0.8, color='red')
axes[3].set_title('Integrated Signal')
axes[3].set_ylabel('Integrated Amplitude')
axes[3].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ADAPTIVE THRESHOLDING (SPKI/NPKI)
init_window = int(2 * fs)  # 2 seconds
spki = 0.25 * np.max(integrated_signal[:init_window])
npki = 0.5 * np.mean(integrated_signal[:init_window])
threshold = npki + 0.25 * (spki - npki)
print(f"SPKI={spki:.4f}, NPKI={npki:.4f}, Threshold={threshold:.4f}")

min_distance = int(0.2*fs)  # Minimum distance between peaks (200 ms)
prominence_value = 0.3 * spki
candidate_indices, _ = find_peaks(integrated_signal, distance=min_distance, prominence=prominence_value)
print(len(candidate_indices))

detected_peaks = []
rr_intervals = []
beat_labels = []
searchback_count = 0
total_recovered = 0

#adaptive thresholding loop
for idx in candidate_indices:
    peak_height = integrated_signal[idx]
    if peak_height > threshold:
        detected_peaks.append(idx)
        spki = 0.125 * peak_height + 0.875 * spki
        if len(detected_peaks) >= 2:
            rr = detected_peaks[-1] - detected_peaks[-2]
            if rr_intervals:
                avg_rr = np.mean(rr_intervals[-8:]) #running average from 8 most recent R-R intervals
                if rr < 0.8 * avg_rr:
                    beat_labels.append("Premature Beat")
                elif rr > 1.2 * avg_rr:
                    beat_labels.append("Delayed Beat")
                else:
                    beat_labels.append("Normal Beat")
                if rr > 1.66 * avg_rr: #checks for 166% threshold of average R-R interval
                    # SEARCHBACK FOR RECOVERING GENUINELY MISSED BEATS
                    print(f"Searchback Needed: gap of {rr} samples between "
                          f"index {detected_peaks[-2]} and {detected_peaks[-1]}")
                    searchback_count += 1
                    gap_start = detected_peaks[-2]
                    gap_end = detected_peaks[-1]
                    trimmed_start = gap_start + min_distance
                    trimmed_end = gap_end - min_distance
                    search_region = integrated_signal[gap_start:gap_end]
                    lowered_threshold = 0.5*threshold
                    if trimmed_start < trimmed_end:
                        search_region = integrated_signal[trimmed_start:trimmed_end]
                        recovered_indices, _ = find_peaks(search_region, height=lowered_threshold, distance=min_distance)
                        recovered_indices = recovered_indices+gap_start  # Adjust indices to the original signal
                        detected_peaks.extend(recovered_indices)
                        total_recovered += len(recovered_indices)
                        detected_peaks.sort()
                        print(f"Recovered {len(recovered_indices)} peaks: {list(recovered_indices)}")
            else:
                beat_labels.append("Normal Beat")
            rr_intervals.append(rr)
        else:
            beat_labels.append("Normal Beat")
    else:
        npki = 0.125 * peak_height + 0.875 * npki
    threshold = npki + 0.25 * (spki - npki)

print(len(detected_peaks))
print(f"Searchback Triggered {searchback_count} times, Recovered {total_recovered} peaks")
print(f"Detected peaks: {len(detected_peaks)}, Beat labels: {len(beat_labels)}")
print(Counter(beat_labels))

peak_times = np.array(detected_peaks) / fs
peak_amplitudes = filtered_signal[detected_peaks]
plt.figure(figsize=(14,4))
plt.plot(t, filtered_signal, linewidth=0.8, label='Filtered ECG')
plt.scatter(peak_times, peak_amplitudes, color='red', marker = 'o', s=30, label='Detected R Peaks', zorder =3)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('Detected R Peaks on Filtered ECG')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# HEART RATE AND HRV (SDNN) COMPUTED FROM THE RESULTING R-R INTERVALS
rr_intervals_ms = np.array(rr_intervals) / fs*1000
mean_rr_sec = np.mean(rr_intervals_ms) / 1000
heart_rate_bpm = 60 / mean_rr_sec
print(f"Average Heart Rate: {heart_rate_bpm:.1f} BPM")
short_gap_positions = np.where(rr_intervals_ms < 300)[0]
for pos in short_gap_positions:
    print(f"Short RR interval: {rr_intervals_ms[pos]:.1f} ms "
          f"between detected_peaks index {pos} and {pos+1} "
          f"(sample {detected_peaks[pos]} -> {detected_peaks[pos+1]})")

sdnn = np.std(rr_intervals_ms)
print(f"SDNN (Heart Rate Variability):{sdnn:.2f} ms")
print(sorted(rr_intervals_ms))

zoom_start = 250
zoom_end = 550

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(np.arange(zoom_start, zoom_end)/fs, filtered_signal2[zoom_start:zoom_end])
axes[0].set_title('QRS-band Filtered Signal (zoomed)')
axes[0].axvline(356/fs, color='red', linestyle='--', alpha=0.6)
axes[0].axvline(442/fs, color='red', linestyle='--', alpha=0.6)

axes[1].plot(np.arange(zoom_start, zoom_end)/fs, integrated_signal[zoom_start:zoom_end])
axes[1].set_title('Integrated Signal (zoomed)')
axes[1].axvline(356/fs, color='red', linestyle='--', alpha=0.6)
axes[1].axvline(442/fs, color='red', linestyle='--', alpha=0.6)
axes[1].axhline(threshold, color='gray', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

export_data = {
    "beat_index": np.arange(len(detected_peaks)),
    "sample_index": detected_peaks,
    "time_sec": peak_times,
    "amplitude_mV": peak_amplitudes,
    "label": beat_labels
}

print(len(detected_peaks), len(peak_times), len(peak_amplitudes), len(beat_labels))