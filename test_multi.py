import numpy as np
import librosa
from metrics import pesq_score, stoi_score
from downsample import downsample


# get file paths
file_original=input('enter original file path:\n')
file_clean=input('enter clean file path:\n')

# load file
snd_orig, sr_0 = librosa.load(file_original, sr=16000)
snd_denoise, sr_1 = librosa.load(file_clean, sr=16000)

print('PESQ: ', pesq_score(snd_orig, snd_denoise, samplerate=sr_0))
print('STOI: ', stoi_score(snd_orig, snd_denoise, samplerate=sr_0))
