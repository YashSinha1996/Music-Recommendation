import python_speech_features as sf
import numpy as np
import pydub
from pylab import *
from scipy.io import wavfile as wv
from collections import Counter
import warnings
from scipy.io import wavfile as wv
from scipy.cluster.vq import kmeans,vq
import python_speech_features as sf
from scipy.io import wavfile as wv
from array import *
from sklearn.metrics.pairwise import euclidean_distances
from ncdlib import compute_ncd, available_compressors
warnings.filterwarnings("ignore")

#Rythmic
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

def rythm(mf1,mf2):
		a=beat_spectrum(mf1)
		#print(a)
		b=beat_spectrum(mf2)
		#print(b)
		if(len(a)>len(b)):
			a=a[:len(b)]
		elif(len(b)>len(a)):
			b=b[:len(a)]
		return np.linalg.norm(a-b)

#Spectral
sb=[]
def feat(mf,c=False,code=[],lifter=0,numcep=13,v=3):
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
	#print(wav,"\t",histo)
	return histo,codebook,sb

def spec(mfcc1,mfcc2,v=3):
	v=3
	code=[]
	def plot_(val,data):
		s=subplot(val)
		title("Histogram-"+str(val)[2])
		s.set_xlabel("Code")
		s.set_ylabel("Frequency")
		s.legend()
		hist(data)
	a,code,sbp = feat(mfcc1,True,v=v)
	b,code,sbp = feat(mfcc2,code=code,v=v)
	figure(1)
	plot_(211,sb[0])
	plot_(212,sb[1])
	show()
	return np.linalg.norm(a-b)	

#Structural
def mat_binary_file(npmat,filename,width=3):
	bin_str=""
	up_tri=npmat.shape
	bin_array=array('B')
	for row in range(up_tri[0]):
		if row<up_tri[1]:
			for col in range(row,up_tri[1]):
				bin_str=bin_str+np.binary_repr(int(npmat[row,col]),width=3)
				if len(bin_str)>=8:
					bin_array.append(int(bin_str[:8],2))
					bin_str=bin_str[8:]
	if bin_str:
		bin_array.append(int(bin_str,2))
	f=open(filename,"wb")
	bin_array.tofile(f)
	f.close()

def get_filename(mf,filename,bits=3):
	img=np.matrix(euclidean_distances(mf))
	maxim=np.max(img)
	img2=np.matrix.round(((img*(1/maxim))*(2**bits)))	#Normalizes to 0 and 2^no_of_bits
	mat_binary_file(img2,filename,bits)
	
def struct(mfcc1,mfcc2):
	get_filename(mfcc1,"mfcc1.qsim")
	get_filename(mfcc2,"mfcc2.qsim")
	cmps=available_compressors()
	ncd=0
	for compressor in cmps:
		ncd+=compute_ncd("mfcc1.qsim", "mfcc2.qsim", compressor)
	# os.remove("mfcc1.qsim")
	# os.remove("mfcc2.qsim")
	if not cmps:
		return np.nan
	return ncd/len(cmps)

def covert(mp3_loc,wav_loc):
	sound = pydub.AudioSegment.from_mp3(mp3_loc)
	sound.export(wav_loc, format="wav")

def mfcc(wav,lifter=0,numcep=13):
	fs,s=wv.read(wav)
	return sf.mfcc(s,samplerate=fs,numcep=numcep,ceplifter=lifter)

def arr(mf1,mf2):
	f1 = spec(mf1,mf2)
	f2 = rythm(mf1,mf2)
	f3 = struct(mf1,mf2)
	return [f1,f2,f3]

file=[]
file.append("/home/pranshu/Dropbox/Music Recommendation/Piano150.wav")
file.append("/home/pranshu/Dropbox/Music Recommendation/Piano1_1_1.wav")
file.append("/home/pranshu/Dropbox/Music Recommendation/piano.wav")
file.append("/home/pranshu/Dropbox/Music Recommendation/wavfile.wav")

def mat(file=file):
	mf=[]
	l=len(file)
	for f in file:
		mf.append(mfcc(f))
	m = np.zeros(shape=(l,l,3))
	for i in range(l):
		for j in range(l):
			for k in range(3):
				if i<j:
					m[i,j,k] = arr(mf[i],mf[j])[k]
				else:
					m[i,j,k] = m[j,i,k]
				print(i,j,k,m[i,j,k])
	for k in range(3):
		m[:,:,k] = m[:,:,k]/np.max(m[:,:,k])
	print(m)
	#ans = np.zeros(shape=(l,l))
	#for i in range(l):
	#	for j in range(l):
	#		ans[i,j] = (m[i,j,0]+m[i,j,1]+m[i,j,2])/3
	#return ans

print(mat(file))
