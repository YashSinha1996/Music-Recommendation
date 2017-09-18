import python_speech_features as sf
import numpy as np
from pylab import *
from scipy.io import wavfile as wv
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import cv2

fs,s=wv.read("/home/yash/Dropbox/Music Recommendation/wavfile.wav")
mf=sf.mfcc(s,samplerate=fs)
img=np.matrix(euclidean_distances(mf))
print(img.shape)
maxim=np.max(img)
img2=1-(img*(1/maxim))
res = cv2.resize(img2,(1024,768), interpolation = cv2.INTER_CUBIC)
cv2.imshow("ok",res)
cv2.waitKey(0)	