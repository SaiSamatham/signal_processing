import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, filtfilt

record = wfdb.rdrecord('100', pn_dir = 'mitdb', sampto=25*360)
fs = record.fs
assert fs == 360, f"Expected 360 Hz, got {fs} Hz" #sampling freq for MIT-BIH record, in Hz

print(record.sig_name)
signal = record.p_signal[:, 0]






