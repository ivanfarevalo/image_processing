import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import time
from PIL import Image

def create_gaussian_kernel(sigma_size, separable=False):
    """
    Creates a gaussian kernel based on sigma and dimensions of kernel matrix
    :param sigma_size: standard deviation of gaussian
    :param kernel_dimensions: kernel dimensions
    """
    # Can modify kernel_dimension relative to sigma_size
    kernel_dimensions = int(6 * sigma_size)
    gaussian_coefficient =  1/math.sqrt(2*math.pi)*sigma_size if separable else 1 / (2 * math.pi * (sigma_size ** 2))
    gaussian_sum = 0
    gaussian_kernel = np.zeros(kernel_dimensions) if separable else np.zeros([kernel_dimensions, kernel_dimensions])

    for x in range(0, kernel_dimensions):
        i = x - kernel_dimensions // 2
        if separable:
            gaussian_exponential = math.exp(-1/2*((i/sigma_size) ** 2))
            gaussian_kernel[x] = gaussian_coefficient * gaussian_exponential
            gaussian_sum += gaussian_kernel[x]
        else:
            for y in range(0, kernel_dimensions):
                # i = x - math.floor(kernel_dimensions / 2)
                j = y - kernel_dimensions // 2
                gaussian_exponential = math.exp(-((i**2)+(j**2))/(2*sigma_size**2))
                # print("iteration {}:{}--> i: {} j: {}   var = {}".format(x, y, i, j, gaussian_exponential))
                gaussian_kernel[x, y] = gaussian_coefficient*gaussian_exponential
                gaussian_sum += gaussian_kernel[x, y]

    normalized_gaussian_kernel = (1/gaussian_sum)*gaussian_kernel
    return normalized_gaussian_kernel.reshape([1, -1]) if separable else normalized_gaussian_kernel

def filter_image(image, kernel, seperable=False):
    start_time = time.time()
    output_im = cv2.filter2D(image, -1, kernel)
    total_time = time.time() - start_time
    if seperable:
        start_time = time.time()
        output_im = cv2.filter2D(output_im, -1, np.transpose(kernel))
        total_time = time.time() - start_time
    return output_im, total_time

if __name__ == '__main__':
    separable = True
    peppers_img = Image.open("peppers.png")
    peppers_img_pxl = np.asarray(peppers_img)

    # Part a) Create a discrete 2D gaussian kernel of size 9x9
    # Part b) We can seperate out the exponential into 2 multiplications.
    # Part c) Perform full and separable convolution on peppers.png and time each operation

    gaussian_kernel = create_gaussian_kernel(sigma_size=1.5, separable=separable)
    print("Gaussian kernel shape: " + str(gaussian_kernel.shape))

    plt.imsave("IvanArevalo-HW3-P3{}.png".format('B1' if separable else 'A'), gaussian_kernel)
    plt.imshow(gaussian_kernel)
    plt.colorbar()
    if separable:
        plt.imsave("IvanArevalo-HW3-P3{}.png".format("B2" if separable else "error"), np.transpose(gaussian_kernel))
        plt.figure()
        plt.imshow(np.transpose(gaussian_kernel))
        plt.colorbar()
        print("Given an image MxN and kernel size L, it would require 2*L multiplications for each output pixel and"
              "given there are (M - floor(L/2)) * (N - floor(L/2)) output pixels, there is a total of "
              "(M - floor(L/2)) * (N - floor(L/2)) * (2xL) multiplications")
        filtered_im, runtime = filter_image(peppers_img_pxl, gaussian_kernel, seperable=separable)
        print("Performing convolution with separable kernels h_x[m] and h_y[n] took {} seconds".format(runtime))
        plt.imsave('IvanArevalo-HW3-P3C_Separable.png', filtered_im, cmap='gray')
        plt.figure()
        plt.imshow(filtered_im, cmap='gray')
        plt.title("Convolution with {}".format("h_x[m] and h_y[n]" if separable else "h[m, n]"))
    else:
        print("Given an image MxN and kernel LxL, it would require (LxL) multiplications for each output pixel and"
              "given there are (M - floor(L/2)) * (N - floor(L/2)) output pixels, there is a total of "
              "(M - floor(L/2)) * (N - floor(L/2)) * (LxL) multiplications")
        filtered_im, runtime = filter_image(peppers_img_pxl, gaussian_kernel, seperable=separable)
        print("Performing convolution with h[m, n] took {} seconds".format(runtime))
        plt.imsave('IvanArevalo-HW3-P3C_Not-Separable.png', filtered_im, cmap='gray')
        plt.figure()
        plt.imshow(filtered_im, cmap='gray')
        plt.title("Convolution with separable kernels h_x[m] and h_y[n]")

    plt.show()