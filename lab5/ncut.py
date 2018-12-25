import numpy as np
import matplotlib.pyplot as plt
import cv2, random
from sklearn.metrics import f1_score

from sklearn.feature_extraction import image
from sklearn.cluster import SpectralClustering

from skimage.transform import resize

def normalizedCutSegmentation(img, cl, aff, nn, g):
	X = img.reshape((-1, 3))

	labels = SpectralClustering(n_clusters=cl, affinity=aff, n_neighbors=nn, gamma=g).fit_predict(X)

	colors = {}
	segImg = img.copy()
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			l = labels[i * img.shape[1] + j]
			colors[l] = img[i][j]

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			segImg[i][j] = colors[labels[i * img.shape[1] + j]]


	return segImg

if __name__ == '__main__':
	path = '124084'
	n, m = 100, 100

	img = cv2.imread(path + '.jpg')
	img = resize(img, (n, m), anti_aliasing=True)

	cv2.imshow('Original', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	for clusters in [3, 5, 7, 9, 11]:
		segImg = normalizedCutSegmentation(img, clusters, 'nearest_neighbors', 5, 1.0)
		cv2.imshow('Segmented', segImg)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		np.save('ncut_nn/' + path + '_' + str(clusters), segImg)
