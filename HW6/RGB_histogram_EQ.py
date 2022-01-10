import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def histogram_EQ(channel, num_levels = 256):
    MN = channel.size
    histogram = np.zeros(256)
    out_channel = np.zeros(channel.shape)
    for j in range(channel.shape[0]):
        for i in range(channel.shape[1]):
            pxl_lvl = channel[j, i]
            histogram[pxl_lvl] += 1

    norm_histogram = histogram/MN
    cdf_histogram = norm_histogram

    for i in range(1, cdf_histogram.size):
        cdf_histogram[i] = cdf_histogram[i] + cdf_histogram[i-1]

    eq_histogram = (num_levels - 1) * cdf_histogram
    round_eq_histogram = np.round(eq_histogram).astype(np.uint8)

    for j in range(channel.shape[0]):
        for i in range(channel.shape[1]):
            pxl_lvl = channel[j, i]
            new_pxl_lvl = round_eq_histogram[pxl_lvl]
            out_channel[j, i] = new_pxl_lvl

    return out_channel

def compute_histogram(channel):
    histogram = np.zeros(256)
    for j in range(channel.shape[0]):
        for i in range(channel.shape[1]):
            pxl_lvl = channel[j, i]
            histogram[pxl_lvl] += 1
    return histogram

# Approach 1: Perform histogram equalization on each of the red, green, blue channels separately
img = Image.open("tree-dark.png")
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()
img_pxl = np.asarray(img)
output_img = np.zeros(img_pxl.shape)
output_img2 = np.zeros(img_pxl.shape)
output_img3 = np.zeros(img_pxl.shape)

# Options: 0 for Linear transformation on pixel
# Options: 1 for Non-Linear transformation on pixel
# Options: 2 for Histogram EQ
option = 2

for i in range(img_pxl.shape[2]):
    plt.imshow(img_pxl[:, :, i], cmap='gray', vmin=0, vmax=255)
    plt.show()

    if option == 0:
        # Linear transformation on pixel
        im_min = np.min(img_pxl[:,:,i])
        im_max = np.max(img_pxl[:,:,i])
        output_img[:,:,i] = np.asarray((255 / (im_max - im_min)) * (img_pxl[:,:,i] - im_min))
        output_img[:,:,i] = np.round(output_img[:,:,i])
        plt.imshow(output_img[:, :, i].astype(np.uint8), cmap='gray', vmin=0, vmax=255)
        plt.show()

    elif option == 1:
        # Non-linear transformation on pixel (No specification on which method to use, chose this one)
        gamma = 0.4
        if i == 2:
            gamma = 0.25
        output_img2[:, :, i] = 255 * np.power(img_pxl[:,:,i]/255, gamma)
        output_img2[:, :, i] = np.round(output_img2[:, :, i])
        plt.imshow(output_img2[:, :, i].astype(np.uint8), cmap='gray', vmin=0, vmax=255)
        plt.show()

    elif option == 2:
        output_img3[:, :, i] = histogram_EQ(img_pxl[:,:,i])
        plt.imshow(output_img3[:, :, i].astype(np.uint8), cmap='gray', vmin=0, vmax=255)
        plt.show()
    else:
        print("Option unavailable")

if option == 0:
    plt.imshow(output_img.astype(np.uint8), cmap='gray', vmin=0, vmax=255)
    plt.show()

elif option == 1:
    plt.imshow(output_img2.astype(np.uint8), cmap='gray', vmin=0, vmax=255)
    plt.show()

elif option == 2:
    plt.imshow(output_img3.astype(np.uint8), cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.imsave("tree-per-channel-hist-eq.png", output_img3.astype(np.uint8), cmap='gray')

else:
    print("Option unavailable")

# Approach 2 : Convert YIQ color and perform histogram EQ
YIQ_transform = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.526, 0.311]])
yiq_img = np.zeros(img_pxl.shape)

for j in range(img_pxl.shape[0]):
    for i in range(img_pxl.shape[1]):
        yiq_img[j,i,:] = YIQ_transform@img_pxl[j,i,:]

plt.imshow(yiq_img.astype(np.uint8), vmin=0, vmax=255)
plt.show()

yiq_eq = yiq_img
yiq_eq[:,:,0] = histogram_EQ(np.round(yiq_img[:,:,0]).astype(np.uint8))

RGB_transform = np.array([[1.000, 0.956, 0.602], [1.000, -0.272, -0.647], [1.000, -1.108, 1.700]])
rgb_img = np.zeros(img_pxl.shape)

for j in range(img_pxl.shape[0]):
    for i in range(img_pxl.shape[1]):
        rgb_img[j,i,:] = RGB_transform@yiq_eq[j,i,:]

plt.imshow(rgb_img.astype(np.uint8), vmin=0, vmax=255)
plt.imsave('tree-y-channel-hist-eq.png', rgb_img.astype(np.uint8), vmin=0, vmax=255)
plt.show()

# Compare histograms
x_vec = np.arange(256)

Red_RGB_hist = compute_histogram(np.round(output_img3[:,:,0]).astype(np.uint8))
plt.bar(x_vec, Red_RGB_hist)
plt.title("Red-per-channel-hist-eq")
plt.savefig('Red-per-channel-hist-eq.png')
plt.clf()
Green_RGB_hist = compute_histogram(np.round(output_img3[:,:,1]).astype(np.uint8))
plt.bar(x_vec, Green_RGB_hist)
plt.title("Green-per-channel-hist-eq")
plt.savefig('Green-per-channel-hist-eq.png')
plt.clf()
Blue_RGB_hist = compute_histogram(np.round(output_img3[:,:,2]).astype(np.uint8))
plt.bar(x_vec, Blue_RGB_hist)
plt.title("Blue-per-channel-hist-eq")
plt.savefig('Blue-per-channel-hist-eq.png')
plt.clf()

Red_YIQ_hist = compute_histogram(np.round(rgb_img[:,:,0]).astype(np.uint8))
plt.bar(x_vec, Red_YIQ_hist)
plt.title("Red-y-channel-hist-eq")
plt.savefig('Red-y-channel-hist-eq.png')
plt.clf()
Green_YIQ_hist = compute_histogram(np.round(rgb_img[:,:,1]).astype(np.uint8))
plt.bar(x_vec, Green_YIQ_hist)
plt.title("Green-y-channel-hist-eq")
plt.savefig('Green-y-channel-hist-eq.png')
plt.clf()
Blue_YIQ_hist = compute_histogram(np.round(rgb_img[:,:,2]).astype(np.uint8))
plt.bar(x_vec, Blue_YIQ_hist)
plt.title("Blue-y-channel-hist-eq")
plt.savefig('Blue-y-channel-hist-eq.png')
plt.clf()

