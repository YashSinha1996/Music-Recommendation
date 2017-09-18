import numpy as np
from pylab import *
import pandas as pd
import subprocess as sp
from scipy.io import wavfile as wv
import pydub
import python_speech_features as sf

mp3_loc = "/home/pranshu/Dropbox/Music Recommendation/file.mp3"
wav_loc = "/home/pranshu/Dropbox/Music Recommendation/wavfile.wav"
lifter = 0
numcep = 13

sound = pydub.AudioSegment.from_mp3(mp3_loc)
sound.export(wav_loc, format="wav")
fs,s=wv.read(wav_loc)
mf=sf.mfcc(s,samplerate=fs,numcep=numcep,ceplifter=lifter)

pd.DataFrame(mf).plot()
show()