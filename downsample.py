"""
downsample files to run the tests
"""

import os
import numpy as np

import librosa
from scipy import interpolate
from scipy.signal import decimate

# ----------------------------------------------------------------------------
from scipy.signal import butter, lfilter
import re

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def downsample(input_file, sample_rate=16000):
  '''
  reduce the sample rate to 16khz or 8khz
  input_file = input file path
  sample_rate = desired sample rate, default=16000
  '''
  # load audio file
  x, fs = librosa.load(input_file)
  
  # generate low-res version
  x_lr = decimate(x, sample_rate)
  
  return x_lr

def upsample(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)
  
  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)
  
  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)

  return x_sp
