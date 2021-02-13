# python3

''' create librosa spectorgram '''

import librosa
from librosa.feature import melspectrogram
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

#audio_file=input('enter original file path:\n')
audio_file = 'sample_audio/script5/f1_script5_ipad_office1_denoised.wav'

# load audio
snd, sr = librosa.load(audio_file, sr=16000) 

spect = melspectrogram(y=snd, sr=sr)
fig, ax = plt.subplots()
S_dB = librosa.power_to_db(spect, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                        y_axis='mel', sr=sr,
                        fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
   
plt.show()