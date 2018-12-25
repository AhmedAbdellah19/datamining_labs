import cv2, os
import numpy as np

path = 'ncut_nn/'
image = '124084_'
for c in [3, 5, 7, 9, 11]:
	a = np.load(path + image + str(c) + '.npy')
	cv2.imshow('Segmented', a)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
