import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def plot_downsampled_image(image, sampling_factor):
    img_pxl = np.asarray(img)[::sampling_factor, ::sampling_factor]
    fft_img = np.fft.fft2(img_pxl)
    fig = plt.figure(sampling_factor)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Downsamples by {}".format(sampling_factor))
    ax1.imshow(img_pxl, cmap='gray', vmin=0, vmax=255)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('Fourier Transform')
    ax2.imshow(np.fft.fftshift(np.abs(fft_img)))
    fig.savefig("{}/downsampled/downsampled_{}.png".format(os.getcwd(), sampling_factor))


# Downsample and display each image FFT Pair
if not os.path.exists("{}/downsampled".format(os.getcwd())):
    os.makedirs("{}/downsampled".format(os.getcwd()))
img = Image.open("barbara.png")
plot_downsampled_image(img, 1)
plot_downsampled_image(img, 2)
plot_downsampled_image(img, 4)
plot_downsampled_image(img, 8)
plt.show()
