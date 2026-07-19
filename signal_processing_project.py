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
plt.show()







