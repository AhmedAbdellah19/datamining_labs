import numpy as np
import cv2
from sklearn.metrics import f1_score


def f1_score_single(y_true, y_pred):
    y_true = set(y_true)
    y_pred = set(y_pred)
    cross_size = len(y_true & y_pred)
    if cross_size == 0: return 0.
    p = 1. * cross_size / len(y_pred)
    r = 1. * cross_size / len(y_true)
    return 2 * p * r / (p + r)

def f1_score(y_true, y_pred):
    return np.mean([f1_score_single(x, y) for x, y in zip(y_true, y_pred)])



img = cv2.imread('23084.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 20
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
segImg = center[label.flatten()].reshape((img.shape))

# print('f1_score', f1_score(img.flatten(), segImg.flatten(), average='weighted'))
print('f1_score', f1_score(img.flatten(), segImg.flatten()))
cv2.imshow('segImg',segImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
