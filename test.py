import numpy as np
import librosa
from metrics import pesq_score, stoi_score
import argparse

# create arguments
parser = argparse.ArgumentParser()
 
parser.add_argument('--original', required=True,
    help='path to original file')
  
parser.add_argument('--clean', required=True,
    help='path to processed/clean file')


# get arguments back
get_inputs = vars(parser.parse_args())
file_original=get_inputs['original']
file_clean=get_inputs['clean']

# load file
snd_orig, sr_0 = librosa.load(file_original)
snd_denoise, sr_1 = librosa.load(file_clean)

print(type('PESQ: ', pesq_score(snd_orig, snd_denoise[:3391270], samplerate=sr_0)))
print(type('STOI: ', stoi_score(snd_orig, snd_denoise[:3391270], fs=sr_0)))
