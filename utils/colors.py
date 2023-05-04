import numpy as np
import matplotlib.pyplot as plt
import cv2

COLORS = (
    (255, 255, 255),    # 0
    (244, 67, 54),      # 1
    (233, 30, 99),      # 2
    (156, 39, 176),     # 3
    (103, 58, 183),     # 4
    (63, 81, 181),      # 5
    (33, 150, 243),     # 6
    (3, 169, 244),      # 7
    (0, 188, 212),      # 8
    (0, 150, 136),      # 9
    (76, 175, 80),      # 10
    (139, 195, 74),     # 11
    (205, 220, 57),     # 12
    (255, 235, 59),     # 13
    (255, 193, 7),      # 14
    (255, 152, 0),      # 15
    (255, 87, 34),
    (121, 85, 72),
    (158, 158, 158),
    (96, 125, 139))


def plot_legend(COLORS):
    plt.figure(figsize=(6, 0.8), dpi=100)
    for idx in np.arange(1, 13):
        plt.subplot(1, 12, idx)
        data = np.array(COLORS[idx] * 30000).reshape(100, 300, 3)
        plt.imshow(data)
        plt.axis('off')
        plt.title('%d' % idx, fontdict={'size': 15, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig('temp.png')

    img = cv2.imread('temp.png')
    img_sum = np.sum(np.array(img), axis=2)
    img[img_sum == 765] = [0, 0, 0]
    img[img_sum == 0] = [255, 255, 255]
    cv2.imwrite('temp.png', img)