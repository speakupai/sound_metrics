# python3

''' create librosa spectorgram '''

import librosa
from librosa.feature import melspectrogram
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_file=input('enter original file path:\n')

# load audio
def load_audio(audio_file):
   snd, sr = librosa.load(audio_file, sr=16000) 

   return snd, sr

def create_spectrogram(snd_file, srate):
   spect = melspectrogram(y=snd_file, sr=srate)
   fig, ax = plt.subplots()
   S_dB = librosa.power_to_db(spect, ref=np.max)
   img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=srate,
                         fmax=8000, ax=ax)
   fig.colorbar(img, ax=ax, format='%+2.0f dB')
   ax.set(title='Mel-frequency spectrogram')
   
   plt.show()

sound_clip, sample_rate = load_audio(audio_file)
create_spectrogram(sound_clip, sample_rate)