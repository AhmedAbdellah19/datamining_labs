import scipy.io
import numpy as np
import matplotlib.image as mpimg
import PIL
import matplotlib.pyplot as plt
import cv2


# def show_image_and_groundTruth(img, gt):
#     f, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))
#     axes[0,0].imshow(img, aspect='auto')
#     axes[0,0].axis('off')
#     # for i in range(len(gt)):
#     #     x = int(i/3)
#     #     y = (i+1)%3
#     #     print(x,y)
#     #     axes[x, y].imshow(gt[i])
#     #     axes[x, y].axis('off', aspect='auto')
#
#     axes[0, 1].imshow(gt[0])
#     axes[0, 1].axis('off', aspect='auto')
#     axes[0, 2].imshow(gt[1])
#     axes[0, 2].axis('off', aspect='auto')
#     axes[1, 0].imshow(gt[2])
#     axes[1, 0].axis('off', aspect='auto')
#     axes[1, 1].imshow(gt[3])
#     axes[1, 1].axis('off', aspect='auto')
#     axes[1, 2].imshow(gt[4])
#     axes[1, 2].axis('off', aspect='auto')
#
#     plt.subplots_adjust(wspace=0.01, hspace=0.01)
#     plt.show()


def show_image_and_groundTruth(img, gt):
    y = int(len(gt) / 2 + 1)
    f, axes = plt.subplots(2, y, figsize=(5, 5))
    r, c = 0, 1
    axes[0, 0].imshow(img, aspect='auto')
    axes[0, 0].axis('off')
    print(len(gt))
    for i in range(len(gt)):
        print(r,c)
        axes[r, c].imshow(gt[i])
        axes[r, c].axis('off', aspect='auto')
        c += 1
        if c == y:
            c = 0
            r += 1
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
    print(label_num)
    for i in range(label_num):
        boundary = groundTruth[0][i]['Segmentation'][0][0].reshape(shape)
        print(boundary.shape)
        gt = np.concatenate((gt, boundary))
        # plt.imshow(boundary)
        # plt.show()
    show_image_and_groundTruth(img, gt)

    return


get_groundTruth('8068')
