# Sound Metrics - Implementation of STOI & PESQ

The files here calculate PESQ and STOI score for originla and processed audio clips

## Environment
* Python 3.6+
* pypesq
* pystoi
* librosa 0.8

## Setup
To use this implementation locally simply clone the repo

```
git clone https://github.com/speakupai/sound_metrics.git;
cd sound_metrics;
```

## Usage and Limitations (more updates soon)
The file can be run through command line using the following commnads
```
usage: python test.py [-h help] [--original Original_file_path] [--clean Processed_file_path]
```

The current version can only process audio files with 8khz and 16khz sampling rates.
