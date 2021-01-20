"""
downsample files to run the tests
"""

import numpy as np

from scipy.signal import decimate

def downsample(input_file, sample_rate=16000):
  '''
  reduce the sample rate to 16khz or 8khz
  input_file = input file path
  sample_rate = desired sample rate, default=16000
  '''  
  # generate low-res version
  x_lr = decimate(input_file, sample_rate)
  
  return x_lr