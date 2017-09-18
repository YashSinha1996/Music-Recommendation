import python_speech_features as sf
import numpy as np
from scipy.io import wavfile as wv
from array import *
from sklearn.metrics.pairwise import euclidean_distances
from ncdlib import compute_ncd, available_compressors
import gpu_euc
import cv2
import os
def mat_binary_file(npmat,filename,width=3):
	"""
		Stores any given numpy matrix of integers into a binary file named filename
		parametres: np.matrix,string filename,int width denoting number of ints to convert 
	"""
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
	"""
	Driver function of the above function,
	also normalizes and quantizes the matrix to fit in bits bits 
	parametres: np.matrix mf, string filename, int bots
	"""
	img=np.matrix(gpu_euc.eucl_gpu(mf))
	maxim=np.max(img)
	#--
	# img2=1-(img*(1/maxim))
	# res = cv2.resize(img2,(1024,768), interpolation = cv2.INTER_CUBIC)
	# cv2.imshow("ok",res)
	# cv2.waitKey(0)
	#--
	img3=np.matrix.round(((img*(1/maxim))*(2**bits)))	#Normalizes to 0 and 2^no_of_bits
	mat_binary_file(img3,filename,bits)
	
#Takes 2 mfcc's as input.
def struct_sim(mfcc1,mfcc2):
	"""
	Finds average NCD from all available compressors between 2 mfccs
	Returns nan if no compressor available
	"""
	get_filename(mfcc1,"mfcc1.qsim")
	get_filename(mfcc2,"mfcc2.qsim")
	cmps=available_compressors()
	ncd=0
	for compressor in cmps:
		ncd+=compute_ncd("numb.qsim", "faint.qsim", compressor)
	# os.remove("mfcc1.qsim")
	# os.remove("mfcc2.qsim")
	if not cmps:
		return np.nan
	return ncd/len(cmps)


if __name__ == "__main__":
	fs,s=wv.read("/home/yash/Dropbox/Music Recommendation/Outputs/Tears.wav")
	mf=sf.mfcc(s,samplerate=fs)
	get_filename(mf,"Tears.qsim")
	# print(struct_sim(mf,mf2))
	# print(struct_sim(mf,mf3))
	# print(struct_sim(mf2,mf3))
	# Output:
	# /usr/local/lib/python3.5/dist-packages/scipy/io/wavfile.py:267: WavFileWarning: Chunk (non-data) not understood, skipping it.
	#   WavFileWarning)
	# 1.0384982342720916
	# 1.000010374350967
	# 1.0000030504568742
	# [Finished in 1948.8s]