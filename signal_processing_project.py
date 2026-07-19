import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, filtfilt

#get ECG signal data from database
record = wfdb.rdrecord('100', pn_dir = 'mitdb', sampto=25*360)
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
plt.show()




