import python_speech_features as sf
import numpy as np
from scipy.io import wavfile as wv
from matplotlib import pyplot as plt

def beat_spectrum(mf):
	length=len(mf)
	beat_sims=[]
	for lag in range(1,10000,50):
		beat_sim=0
		actual=0
		for i in range(length-lag):
			beat_sim+=np.linalg.norm(mf[i+lag]-mf[i])
			actual+=1
		if actual>0:
			beat_sims.append(beat_sim)
	return np.array(beat_sims)

def rs(mf1,mf2,ax):
	a=beat_spectrum(mf1)
	print(a)
	line_a,=ax.plot(a,label="Numb by Linkin Park")
	b=beat_spectrum(mf2)
	print(b)
	line_b,=ax.plot(b,label="Faint by Linkin Park")
	ax.legend();
	if(len(a)>len(b)):
		a=a[:len(b)]
	elif(len(b)>len(a)):
		b=b[:len(a)]
	return np.linalg.norm(a-b)

fig = plt.figure()
fig.suptitle('Beat Spectrum summations', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

ax.set_xlabel('Lag in milliseconds')
ax.set_ylabel('Beat Spectrum Summation with lag x')
fs,s=wv.read("/home/yash/Dropbox/Music Recommendation/Outputs/numb.wav")
mf=sf.mfcc(s,samplerate=fs)
fs2,s2=wv.read("/home/yash/Dropbox/Music Recommendation/Outputs/faint.wav")
mf2=sf.mfcc(s2,samplerate=fs)
print("mfcc done")
print(rs(mf,mf2,ax))
plt.show()


# print(img.shape)
#  for
#  maxim=np.max(img)
#  img2=1-(img*(1/maxim))
#  res = cv2.resize(img2,(1024,768), interpolation = cv2.INTER_CUBIC)
#  cv2.imshow("ok",res)
#  cv2.waitKey(0)
