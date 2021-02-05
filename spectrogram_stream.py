# python3

''' create librosa spectorgram '''

import librosa
from librosa.feature import melspectrogram
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

#audio_file=input('enter original file path:\n')
#audio_file = '/home/taimur/Documents/Online Courses/Fourth Brain/Projects/Audio_super_res/STOI & PESQ/sound_metrics/sample_audio/run 0 only wavenet/f1_script2_ipad_balcony1.wav'
audio_folder = '/home/taimur/Documents/Online Courses/Fourth Brain/Projects/Audio_super_res/STOI & PESQ/sound_metrics/sample_audio/run 0 only wavenet'

file_list = os.listdir(audio_folder)
os.chdir(audio_folder)


for fname in file_list:
   # load audio
   snd, sr = librosa.load(fname, sr=16000)

   spect = melspectrogram(y=snd, sr=sr)
   fig, ax = plt.subplots()
   S_dB = librosa.power_to_db(spect, ref=np.max)
   img = librosa.display.specshow(S_dB, x_axis='time',
                                  y_axis='mel', sr=sr,
                                  fmax=8000, ax=ax)
   fig.colorbar(img, ax=ax, format='%+2.0f dB')
   ax.set(title='Mel-frequency spectrogram')

   print(fname[:-4])
   fig.savefig(fname[:-4], format='png')
   
   plt.show()
