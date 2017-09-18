import numpy as np
from collections import Counter
from pylab import *
import warnings
from scipy.io import wavfile as wv
from scipy.cluster.vq import kmeans,vq
import python_speech_features as sf
from pylab import *
warnings.filterwarnings("ignore")



file1 = "/home/pranshu/Dropbox/Music Recommendation/Piano1_1_1.wav"
file2 = "/home/pranshu/Dropbox/Music Recommendation/piano.wav"
lifter = 0
numcep = 13
v=3
code=[]

sb=[]
def feat(wav,c=False,code=[],lifter=0,numcep=13,v=3):
	fs,s=wv.read(wav)
	mf=sf.mfcc(s,samplerate=fs,numcep=numcep,ceplifter=lifter)
	norm_feat=[]
	for i,feat in enumerate(mf):
		der = np.gradient(feat)
		der2 = np.gradient(feat,2)
		der = np.concatenate((feat,der,der2))
		norm_feat.append((der-np.mean(der))/np.std(der))
	if c==True:
		codebook, distortion = kmeans(norm_feat, v)
	else:
		codebook = code
	codewords, dist = vq(norm_feat, codebook)
	sb.append(codewords)
	histo = np.array(list(Counter(codewords).values()))#/len(mf)
	print(wav,"\t",histo)
	return histo,codebook,sb

def plot_(val,data):
	s=subplot(val)
	title("Histogram-"+str(val)[2])
	s.set_xlabel("Code")
	s.set_ylabel("Frequency")
	s.legend()
	hist(data)
a,code,sbp = feat(file1,True,v=v)
b,code,sbp = feat(file2,code=code,v=v)
figure(1)
plot_(211,sb[0])
plot_(212,sb[1])
show()
ans1 = np.linalg.norm(a-b)
print(ans1)