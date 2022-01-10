import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d

alpha = 1
impulse = np.array([[0,0,0],[0,1,0],[0,0,0]])
laplace = np.array([[1,1,1],[1,-8,1],[1,1,1]])
sharp_filter = impulse - alpha*laplace #considering diagonal and alpha =1
print(sharp_filter)
img = Image.open("peppers_color.png")
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()
img_pxl = np.asarray(img).astype(np.uint8)
output_img = np.zeros(img_pxl.shape)


for i in range(img_pxl.shape[2]):
    plt.imshow(img_pxl[:, :, i], cmap='gray', vmin=0, vmax=255)
    plt.show()
    output_img[:, :, i] = convolve2d(img_pxl[:, :, i], sharp_filter, mode='same')
    plt.imshow(output_img[:, :, i], cmap='gray', vmin=0, vmax=255)
    plt.show()

plt.imshow(output_img.astype(np.uint8), vmin=0, vmax=255)
plt.show()
plt.imsave('peppers enhanced RGB.png', np.round(output_img).astype(np.uint8), vmin=0, vmax=255)

YIQ_transform = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.526, 0.311]])
print(YIQ_transform)
yiq_img = np.zeros(img_pxl.shape)

for j in range(img_pxl.shape[0]):
    for i in range(img_pxl.shape[1]):
        yiq_img[j,i,:] = YIQ_transform@img_pxl[j,i,:]

print("Displaying raw YIQ")
plt.imshow(yiq_img.astype(np.uint8), vmin=0, vmax=255)
plt.show()



yiq_eq = np.copy(yiq_img)
yiq_trial = np.copy(yiq_img)
yiq_eq[:,:,0] = convolve2d(yiq_img[:, :, 0], sharp_filter, mode='same')

print("Displaying Y after filter")
print(yiq_eq.shape)
plt.imshow(yiq_eq[:,:,0].astype(np.uint8), cmap='gray', vmin=0, vmax=255)
plt.show()

RGB_transform = np.array([[1.000, 0.956, 0.602], [1.000, -0.272, -0.647], [1.000, -1.108, 1.700]])
print(RGB_transform)
rgb_img = np.zeros(img_pxl.shape)

for j in range(img_pxl.shape[0]):
    for i in range(img_pxl.shape[1]):
        rgb_img[j,i,:] = RGB_transform@yiq_eq[j,i,:]
        yiq_trial[j,i,:] = RGB_transform@yiq_img[j,i,:]

print("Displaying RGB image")
plt.imshow(rgb_img.astype(np.uint8))
plt.imsave('peppers enhanced Y.png', rgb_img.astype(np.uint8))
plt.show()


