import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import glob
from moviepy.editor import *


# Generate Fourier Basis for each Fourier Coefficient F(µ, v).
def generate_fourier_basis(fy, fx, image):
    x_vec = np.arange(0, image.shape[1])
    y_vec = np.arange(0, image.shape[0])
    ex = np.exp(1j * 2 * np.pi * fx/x_vec.size * x_vec)
    ey = np.exp(1j * 2 * np.pi * fy/y_vec.size * y_vec)
    fourier_basis = np.outer(ey, ex)
    return fourier_basis


def generate_reconstruction_video():
    current_directory = os.getcwd()
    img_array = []

    for filename in sorted(glob.glob("{}/video_frames/*.jpg".format(current_directory))):
        img = cv2.imread(filename)
        img_array.append(img)

    # creating a Image sequence clip with fps = 10
    clip = ImageSequenceClip(img_array, fps=10)
    clip.write_videofile("FT_reconstruction.mp4")


# Main
img = Image.open("goldhill.png")
img_pxl = np.asarray(img)[::1,::1] # Can downsample for testing
fft_img = np.fft.fft2(img_pxl)
high_energy_fft_idx = np.dstack(np.unravel_index(np.argsort(abs(fft_img).ravel())[-1:0:-1], (fft_img.shape[1], fft_img.shape[0]))).squeeze()
reconstructed_image = np.zeros(fft_img.shape, dtype='complex128')

# Create video frames folder
if not os.path.exists("{}/video_frames".format(os.getcwd())):
    os.makedirs("{}/video_frames".format(os.getcwd()))

frame_num = 1
for i, idx in enumerate(high_energy_fft_idx):

    # Update reconstructed image
    fourier_basis = generate_fourier_basis(idx[0], idx[1], img_pxl)
    fourier_phase = np.exp(1j*np.angle(fft_img[idx[0], idx[1]]))
    reconstructed_image += 1/img_pxl.size * abs(fft_img[idx[0], idx[1]]) * fourier_basis * fourier_phase
    if i <= 100 or (i < 500 and i % 5 == 0) or (i < 5000 and i % 100 == 0) or (i < 50000 and i % 500 == 0) or (i % 5000 == 0):
        # Save video frame
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title('Reconstructed Image')
        ax1.imshow(np.real(reconstructed_image), cmap='gray')
        ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title("{}th Highest Energy FT Basis: ƒ({}, {})".format(i, idx[0], idx[1]))
        ax2.imshow(np.real(fourier_basis), cmap='gray')
        ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        fig.savefig("video_frames/frame{:010d}.jpg".format(frame_num))
        frame_num += 1
        plt.close()

generate_reconstruction_video()

plt.imshow(np.real(reconstructed_image), cmap='gray')
plt.show()
