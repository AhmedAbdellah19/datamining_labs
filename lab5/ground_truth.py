import scipy.io
import numpy as np
import matplotlib.image as mpimg
import PIL
import matplotlib.pyplot as plt
import cv2


def show_image_and_groundTruth(img, gt):
    f, axes = plt.subplots(1, len(gt) + 1, figsize=(10, 10))
    axes[0].imshow(img, aspect='auto')
    axes[0].axis('off')
    for i in range(len(gt)):
        axes[i+1].imshow(gt[i])
        axes[i+1].axis('off', aspect='auto')
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.show()

def get_groundTruth(path):
    """
    return the nparray of boundary (0 for boundary and 255 for area)
    :param path:
    :return:
    """
    img = cv2.imread(path + '.jpg')
    mat = scipy.io.loadmat(path + '.mat')
    groundTruth = mat.get('groundTruth')
    label_num = groundTruth.size
    shape = (1, img.shape[0], img.shape[1])
    gt = np.empty((0, img.shape[0], img.shape[1]))
    print(img.shape)
    for i in range(label_num):
        boundary = groundTruth[0][i]['Segmentation'][0][0].reshape(shape)
        print(boundary.shape)
        gt = np.concatenate((gt, boundary))
        # plt.imshow(boundary)
        # plt.show()
    show_image_and_groundTruth(img, gt)

    return


get_groundTruth('2092')
