import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from HW3.ECE178_HW3_P3 import create_gaussian_kernel


def compute_DFT(signal):
    fig = plt.figure()
    fig_a = fig.add_subplot(1, 2, 1)
    fig_a.set_title("Signal")
    fig_a.imshow(signal, cmap='gray')

    signal_fft = np.fft.fft2(signal)
    fig_b = fig.add_subplot(1, 2, 2)
    fig_b.set_title("DFT")
    fig_b.imshow(abs(np.fft.fftshift(signal_fft)), cmap='gray')

    return signal_fft


M = 512 # Height image
N = 256 # Width image

# Visualize FT of the following signals.
# f[m,n] = 1
signal1 = np.ones([M, N])
signal1_fft = compute_DFT(signal1)

# f[m,n] = sin(20*pi*m/M) + cos(2*pi*n/N)
signal2a = np.outer(np.sin(20*np.pi*np.arange(0,M)/M), np.ones(N))
signal2b = np.outer(np.ones(M), np.cos(6*np.pi*np.arange(0,N)/N))
signal2 = signal2a + signal2b
signal2_fft = compute_DFT(signal2)

# f[m,n] = sin(20*pi*m/M) * cos(2*pi*n/N)
signal3a = np.outer(np.sin(20*np.pi*np.arange(0,M)/M), np.ones(N))
signal3b = np.outer(np.ones(M), np.cos(6*np.pi*np.arange(0,N)/N))
signal3 = signal3a * signal3b
signal3_fft = compute_DFT(signal3)


# Filtering in frequency domain
peppers_img = Image.open("/Users/ivanarevalomac/Google Drive/School/Signal Processing/ECE178/ECE178_Assignments/HW3/peppers.png")
peppers_img_pxl = np.asarray(peppers_img)
peppers_img_fft = np.fft.fft2(peppers_img)

gaussian_kernel = create_gaussian_kernel(sigma_size=1.5)
ypad_size = int((peppers_img_pxl.shape[0]-gaussian_kernel.shape[0])//2)
xpad_size = int((peppers_img_pxl.shape[1]-gaussian_kernel.shape[1])//2)
pad_gaussian_kernel = np.pad(gaussian_kernel, ( [0, 2*ypad_size+1], [0, 2*xpad_size+1]))

# reconstructed_signal1 = np.fft.ifftshift(np.fft.ifft2(pad_gaussian_kernel*np.fft.fftshift(peppers_img_fft)))
reconstructed_signal1 = np.fft.ifft2(pad_gaussian_kernel*(peppers_img_fft))
# reconstructed_signal2 = np.fft.ifft2(np.fft.ifftshift(pad_gaussian_kernel*np.fft.fftshift(np.fft.fft2(peppers_img_pxl))))
# reconstructed_signal3 = np.fft.ifftshift(np.fft.ifft2(pad_gaussian_kernel*np.fft.fftshift(signal3_fft)))

plt.figure()
plt.imshow(abs(reconstructed_signal1), cmap='gray')

# Main
img = Image.open("goldhill.png")
img_pxl = np.asarray(img)[::1,::1] # Can downsample for testing
fft_img = np.fft.fft2(img_pxl)