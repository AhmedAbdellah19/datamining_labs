import scipy.io
import numpy as np
import matplotlib.image as mpimg
import PIL

def get_groundTruth(path):
    """
    return the nparray of boundary (0 for boundary and 255 for area)
    :param path:
    :return:
    """
    mat = scipy.io.loadmat(path)
    groundTruth = mat.get('groundTruth')
    label_num = groundTruth.size

    for i in range(label_num):
        boundary = groundTruth[0][i]['Segmentation'][0][0]
        # print(boundary)
        # print(boundary.shape)
        height = boundary.shape[0]
        width = boundary.shape[1]
        boundary = boundary.reshape(1, height, width)
        # boundary = 255 * np.ones([height, width, 1], dtype="uint8") - (boundary > 0) * 255
        # boundary = np.uint8(boundary)
        # print(boundary)
        # img = Image.fromarray(boundary)
        # img.show()

    return


get_groundTruth('2092.mat')
# img = Image.fromarray(data, 'RGB')
# img.show()



# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.io as sio
# import glob
# import os
# import matplotlib.image as mpimg
# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# dir = "/home/moustafa/PycharmProjects/lab #5/BSR/BSDS500/data/"
#
#
# def show_image_and_groundTruth(img, gt):
#     f, axes = plt.subplots(1, len(gt) + 1, figsize=(10, 10))
#     axes[0].imshow(img, aspect='auto')
#     axes[0].axis('off')
#     for i in range(len(gt)):
#         axes[i+1].imshow(gt[i])
#         axes[i+1].axis('off', aspect='auto')
#     plt.subplots_adjust(wspace=0.01, hspace=0.01)
#     plt.show()
#
# def load_images(path):
#     og_images = []
#     gt_images = []
#     images = glob.glob(dir + "images/" + path + "/*.jpg")
#     for image in images:
#         _, t = os.path.split(image)
#         filename = os.path.splitext(t)[0]
#         img = mpimg.imread(image)
#         gt_img = sio.loadmat(dir + "groundTruth/" + path + "/" + str(filename) + ".mat")
#         gt = np.empty((0, img.shape[0], img.shape[1]))
#         for i in range(len(gt_img['groundTruth'][0])):
#             cur = gt_img['groundTruth'][0][i][0][0][0].reshape((1, img.shape[0], img.shape[1]))
#             gt = np.concatenate((gt, cur))
#         og_images.append(img)
#         gt_images.append(gt)
#         show_image_and_groundTruth(img, gt)
#     return np.asarray(og_images), np.asarray(gt_images)
#
# if __name__ == "__main__":
#     # train, train_gt = load_images("train")
#     test, test_gt = load_images("test")
#     # val, val_gt = load_images("val")
