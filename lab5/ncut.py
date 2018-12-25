import numpy as np
import matplotlib.pyplot as plt
import cv2, random
from sklearn.metrics import f1_score

from sklearn.feature_extraction import image
from sklearn.cluster import SpectralClustering

from skimage.transform import resize
from skimage import img_as_uint

import eval


def show_image_and_segImages(img, segImgs):
    y = 3
    f, axes = plt.subplots(2, y, figsize=(5, 5))
    r, c = 0, 1
    axes[0, 0].imshow(img, aspect='auto')
    axes[0, 0].axis('off')
    print(len(segImgs))
    for i in range(len(segImgs)):
        print(r,c)
        axes[r, c].imshow(segImgs[i])
        axes[r, c].axis('off', aspect='auto')
        c += 1
        if c == y:
            c = 0
            r += 1
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.show()



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

	return segImg, labels

if __name__ == '__main__':
	path = '8068'
	n, m = 75, 75

	img = cv2.imread(path + '.jpg')
	img = resize(img, (n, m), anti_aliasing=True)

	'''
	cv2.imshow('Original', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''

	gt = eval.get_ground_truth_labels(path)
	gt_labels = img_as_uint(resize(gt, (n, m), anti_aliasing=False))
	gt_labels = gt_labels.flatten()

	for clusters in [3, 5, 7, 9, 11]:
		segImg, labels = normalizedCutSegmentation(img, clusters, 'rbf', 5, 1.0)

		'''
		cv2.imshow('Segmented', segImg)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		np.save('ncut_nn/' + path + '_' + str(clusters), segImg)
		'''

		print('F-measure = ', eval.f_measure(gt_labels, labels, clusters))
		print('Conditional Entropy = ', eval.conditional_entropy(gt_labels, labels, clusters))

'''
	# cv2.imshow('Original', img)
	# cv2.waitKey(0)
	cv2.destroyAllWindows()
	segImgs = []

	for clusters in [3, 5, 7, 9, 11]:
		print(clusters)
		segImg = normalizedCutSegmentation(img, clusters, 'nearest_neighbors', 5, 1.0)
		np.save('ncut_nn/' + path + '_' + str(clusters), segImg)
		segImgs.append(segImg)

	show_image_and_segImages(img, segImgs)
'''
