from math import log
import cv2, scipy.io
import numpy as np

def f_measure(ground_truth_labels, clustered_labels, clusters):
	ground_truth_clusters = max(ground_truth_labels)

	tj = [0] * ground_truth_clusters
	for i in ground_truth_labels:
		tj[i - 1] += 1

	maxni = [0] * clusters
	ni = [[0 for i in range(ground_truth_clusters)] for j in range(clusters)]
	for i in range(len(clustered_labels)):
		l = clustered_labels[i]
		ni[l][ground_truth_labels[i] - 1] += 1
		maxni[l] = max(maxni[l], ni[l][ground_truth_labels[i] - 1])

	prec = [maxni[i] / sum(ni[i]) for i in range(clusters)]

	rec = []
	for i in range(clusters):
		mx = 0
		for j in range(ground_truth_clusters):
			mx = max(mx, ni[i][j] / tj[j])
		rec.append(mx)

	f_score = 0
	for i in range(clusters):
		f_score += 2 * prec[i] * rec[i] / (prec[i] + rec[i])
	return f_score / clusters

def conditional_entropy(ground_truth_labels, clustered_labels, clusters):
	n = len(ground_truth_labels)
	ground_truth_clusters = max(ground_truth_labels)

	tj = [0] * ground_truth_clusters
	for i in ground_truth_labels:
		tj[i - 1] += 1

	maxni = [0] * clusters
	ni = [[0 for i in range(ground_truth_clusters)] for j in range(clusters)]
	for i in range(len(clustered_labels)):
		l = clustered_labels[i]
		ni[l][ground_truth_labels[i] - 1] += 1
		maxni[l] = max(maxni[l], ni[l][ground_truth_labels[i] - 1])

	ret = 0
	for i in range(clusters):
		h = 0
		for j in range(ground_truth_clusters):
			p = ni[i][j] / sum(ni[i])
			if p > 0: h -= p * log(p)
		ret += sum(ni[i]) / n * h
	return ret

def get_ground_truth_labels(path):
	mat = scipy.io.loadmat(path + '.mat')
	groundTruth = mat.get('groundTruth')
	return groundTruth[0][0]['Segmentation'][0][0]