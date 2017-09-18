import cv2
import numpy as np
img1 = cv2.imread("song 5 limit.png")
img2 = cv2.imread("song 26limit.png")
img12 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img22 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
print mse(img12,img22)