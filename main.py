import cv2
import numpy as np
from typing import List
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from math import pi, exp


class Image:
    def __init__(self, path: str = None, data: np.array = None):
        if path:
            data = cv2.imread(path)
            # grey scale data
            self.data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        else:
            self.data = data

    def plot(self) -> None:
        max_ = np.max(self.data)
        min_ = np.min(self.data)
        plt.imshow(self.data, cmap="gray", vmin=min_, vmax=max_)
        plt.show()


class CannyEdgeDetector:
    SOBEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    SOBEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    KERNEL_SIZE = 5
    KERNEL_SD = 1.41

    def __init__(self, img: Image) -> None:
        self.img = img

    def intensityGradient(self) -> List[np.array]:
        Gx = convolve2d(self.img.data, self.SOBEL_X)
        Gy = convolve2d(self.img.data, self.SOBEL_Y)
        G = np.sqrt(Gx ** 2 + Gy ** 2)
        O = np.arctan2(Gy, Gx)
        return G, O

    def __gaussianKernel__(self, n: int, s: float) -> np.array:
        k = (n-1) / 2
        kernel = np.zeros((n, n))
        for i in range(1, n+1):
            for j in range(1, n+1):
                kernel[i-1][j-1] = 1/(2*pi*s*s) * \
                    exp(-((i-(k+1))**2+(j-(k+1))**2)/(2*s*s))
        # Normalization Step
        kernel /= np.sum(kernel)
        return kernel

    def gaussianFilter(self) -> Image:
        kernel = self.__gaussianKernel__(self.KERNEL_SIZE, self.KERNEL_SD)
        return Image(data=convolve2d(self.img.data, kernel))


if __name__ == '__main__':
    img = Image('women.jpg')
    ced = CannyEdgeDetector(img)

    img.plot()

    gauss_filter_img = ced.gaussianFilter()
    gauss_filter_img.plot()

    grads = ced.intensityGradient()[0]
    img_grads = Image(data=grads)
    img_grads.plot()

    # grad = CannyEdgeDetector(img).intensityGradient()[0]
    # img_grad = Image(data=grad)
    # img.plot()
    # img_grad.plot()
