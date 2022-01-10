import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Generate contrast stretched image
peppers_img = Image.open("peppers_poor_constrast.png")
plt.imshow(peppers_img, cmap='gray', vmin=0, vmax=255)
peppers_img_pxl = np.asarray(peppers_img)
im_min = np.min(peppers_img_pxl)
im_max = np.max(peppers_img_pxl)
contrast_stretch_image = np.asarray((255 / (im_max - im_min)) * (peppers_img_pxl - im_min))
approx_contrast_stretch_image = np.round(contrast_stretch_image).astype(np.uint8)
plt.imshow(approx_contrast_stretch_image, cmap='gray', vmin=0, vmax=255)
plt.imshow(approx_contrast_stretch_image, cmap='gray',)
plt.imsave('IvanArevalo-HW1-P4A.png', approx_contrast_stretch_image, cmap='gray', vmin=0, vmax=255)

# Generate negative of contrast stretched image
negative_im = 255 - contrast_stretch_image
aprox_negative_im = 255 - approx_contrast_stretch_image
plt.imshow(negative_im, cmap='gray', vmin=0, vmax=255)
plt.imsave('IvanArevalo-HW1-P4B.png', aprox_negative_im, cmap='gray', vmin=0, vmax=255)

# Calculate rounding error
error_contrast = np.sqrt(np.mean(np.square(contrast_stretch_image - approx_contrast_stretch_image)))
print("The RMSE for the contrast stretched image was {}".format(error_contrast))
error_negative = np.sqrt(np.mean(np.square(negative_im - aprox_negative_im)))
print("The RMSE for the negative contrast stretched image was {}".format(error_negative))