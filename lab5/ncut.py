import numpy as np
import matplotlib.pyplot as plt
import cv2, random

from sklearn.feature_extraction import image
from sklearn.cluster import SpectralClustering

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
	path = '124084.jpg'
	n, m = 60, 60
	clusters = 20
	affinity = 'nearest_neighbors'
	knn = 5
	gamma = 1.0

	img = cv2.imread(path)
	img = cv2.resize(img, (n, m))

	segImg = normalizedCutSegmentation(img, clusters, affinity, knn, gamma)

	cv2.imshow('Original', img)
	cv2.waitKey(0)
	cv2.imshow('Segmented', segImg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()