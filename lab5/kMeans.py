import numpy as np
import cv2
import matplotlib.pyplot as plt
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



img = cv2.imread('124084.jpg')
Z = img.reshape((-1,3))
# plt.imshow(img)
# plt.show()

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
Ks = [3,5,7,9,11]
segImgs = []
print(len(Ks))
for K in Ks:
    #print(K)
    ret,labels,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    segImg = center[labels.flatten()].reshape((img.shape))

    segImgs.append(segImg)

    gt = eval.get_ground_truth_labels('124084')
    gt = img_as_uint(gt).flatten()
    labels = labels.flatten()
    print('F-measure = ', eval.f_measure(gt, labels, K))
    print('Conditional Entropy = ', eval.conditional_entropy(gt, labels, K))

#show_image_and_segImages(img, segImgs)
