import numpy as np
import librosa
from metrics import pesq_score, stoi_score
from downsample import downsample


# get arguments back
#file_original=input('enter original file path')
file_original = '/home/taimur/Documents/Online Courses/Fourth Brain/Projects/Audio_super_res/STOI & PESQ/sound_metrics/sample_audio/run 0 only wavenet/f1_script2_ipad_balcony1.wav'
#file_clean=input('enter clean file path')
file_clean = '/home/taimur/Documents/Online Courses/Fourth Brain/Projects/Audio_super_res/STOI & PESQ/sound_metrics/sample_audio/run 0 only wavenet/f1_script2_ipad_balcony1_denoised.wav'

# load file
snd_orig, sr_0 = librosa.load(file_original)
snd_denoise, sr_1 = librosa.load(file_clean)
print(sr_0)

if sr_0 != 16000 or sr_0 != 8000:
    snd_orig = downsample(snd_orig)
    snd_denoise = downsample(snd_denoise)
    sr_0 = 16000

print(snd_orig.shape)
print(snd_denoise.shape)

print('PESQ: ', pesq_score(snd_orig, snd_denoise, samplerate=16000))
#print('STOI: ', stoi_score(snd_orig, snd_denoise, samplerate=16000))
