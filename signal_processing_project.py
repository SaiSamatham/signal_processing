import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, filtfilt, find_peaks

#get ECG signal data from database
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

#bandpass filter
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

#detection-only signal (Pan-Tompkins passband)
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

min_distance = int(0.2*fs)  # Minimum distance between peaks (200 ms)
candidate_indices, _ = find_peaks(integrated_signal, distance=min_distance)
print(len(candidate_indices))

init_window = int(2 * fs)  # 2 seconds
spki = 0.25 * np.max(integrated_signal[:init_window])
npki = 0.5 * np.mean(integrated_signal[:init_window])
threshold = npki + 0.25 * (spki - npki)
print(f"SPKI={spki:.4f}, NPKI={npki:.4f}, Threshold={threshold:.4f}")

detected_peaks = []
rr_intervals = []

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
                if rr > 1.66 * avg_rr: #checks for 166% threshold of average R-R interval
                    print(f"Searchback Needed: gap of {rr} samples between "
                          f"index {detected_peaks[-2]} and {detected_peaks[-1]}")
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
                        detected_peaks.sort()
                        print(f"Recovered {len(recovered_indices)} peaks: {list(recovered_indices)}")
            rr_intervals.append(rr)
        else:
            npki = 0.125 * peak_height + 0.875 * npki
        threshold = npki + 0.25 * (spki - npki)

print(len(detected_peaks))


