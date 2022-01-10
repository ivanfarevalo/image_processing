import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def compute_lin_transform(input_img, out_img, num_pxls):
    selected_pxls = []
    M = np.ones([num_pxls, 4])
    t = np.zeros([num_pxls, 3])
    iter = 0
    while len(selected_pxls) < num_pxls:
        j = np.random.randint(0, input_img.shape[0])
        i = np.random.randint(0, input_img.shape[1])
        if (j,i) not in selected_pxls:
            selected_pxls.append((j,i))
            M[iter, :3] = input_img[j, i, :]
            t[iter] = out_img[j, i, :]
            iter +=1

    A = np.zeros([3, 4])
    for i in range(t.shape[1]):
        a =  np.linalg.lstsq(M, t[:,i], rcond=None)[0]
        A[i, :] = a
    print(A)

    return A

def perform_lin_transform(input, A):
    col_img = np.zeros(out_img_pxl.shape)
    for j in range(in_img_pxl.shape[0]):
        for i in range(in_img_pxl.shape[1]):
            x = np.concatenate((in_img_pxl[j, i, :], np.ones(1)))
            col_img[j, i, :] = A @ x
    return col_img

in_img = Image.open("q2_orig.png")
out_img = Image.open("q2_color_txm.png")
in_img_pxl = np.asarray(in_img)
out_img_pxl = np.asarray(out_img)

A4 = compute_lin_transform(in_img_pxl, out_img_pxl, 4)
col_img4 = perform_lin_transform(in_img_pxl, A4)
print("Displaying Color Calibrated Image")
plt.imshow(col_img4.astype(np.uint8), vmin=0, vmax=255)
plt.show()
# plt.imsave('q2_calibrated_4p.png', col_img4.astype(np.uint8))
MSE4 = ((out_img_pxl - col_img4)**2).mean(axis=None)
print("MSE with 4 pixels: {}".format(MSE4))

A8 = compute_lin_transform(in_img_pxl, out_img_pxl, 8)
col_img8 = perform_lin_transform(in_img_pxl, A8)
print("Displaying Color Calibrated Image")
plt.imshow(col_img8.astype(np.uint8), vmin=0, vmax=255)
plt.show()
# plt.imsave('q2_calibrated_8p.png', col_img8.astype(np.uint8))
MSE8 = ((out_img_pxl - col_img8)**2).mean(axis=None)
print("MSE with 8 pixels: {}".format(MSE8))
