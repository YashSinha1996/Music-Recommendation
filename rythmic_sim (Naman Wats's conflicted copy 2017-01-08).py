import python_speech_features as sf
import numpy as np
from scipy.io import wavfile as wv

def beat_spectrum(mf):
	length=len(mf)
	beat_sims=[]
	for lag in range(1,10000,1):
		beat_sim=0
		actual=0
		for i in range(length-lag):
			beat_sim+=np.linalg.norm(mf[i+lag]-mf[i])
			actual+=1
		if actual>0:
			beat_sims.append(beat_sim)
	return np.array(beat_sims)

def rs(mf1,mf2):
	a=beat_spectrum(mf1)
	print(a)
	b=beat_spectrum(mf2)
	print(b)
	if(len(a)>len(b)):
		a=a[:len(b)]
	elif(len(b)>len(a)):
		b=b[:len(a)]
	return np.linalg.norm(a-b)

fs,s=wv.read("/home/yash/Kun Faya Kun.wav")
mf=sf.mfcc(s,samplerate=fs)
fs2,s2=wv.read("/home/yash/Sanson Ki Mala Pe.mp3.wav")
mf2=sf.mfcc(s2,samplerate=fs)
print("mfcc done")
print(rs(mf,mf2))

#print(img.shape)
#for
#img2=1-(img*(1/maxim))
#res = cv2.resize(img2,(1024,768), interpolation = cv2.INTER_CUBIC)
#cv2.imshow("ok",res)
#cv2.waitKey(0)
