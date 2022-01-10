import numpy as np

## this function takes an input image of size h x w (grayscale image) or h x w x 3 (colored image)
## and returns N x patch_size x patch_size or N x patch_size x patch_size x 3 array 
## containing all possible NON-OVERLAPPING patches in the image
## CAUTION: ensure that the height and width of the input image is an integral
## multiple of patch_size
def extract_patches(image, patch_size):
	if len(image.shape) == 3:
		h,w,c = image.shape
	elif len(image.shape) == 2:
		h,w = image.shape
		image = image[:,:,None]
		c = 1
	else:
		print('ERROR: unexpected image size')
		return NotImplementedError
	if h % patch_size != 0:
		print('ERROR: the height of the image is expected to be an integral multiple of patch_size')
		return NotImplementedError
	if w % patch_size != 0:
		print('ERROR: the height of the image is expected to be an integral multiple of patch_size')
		return NotImplementedError    

	n_patches = (h // patch_size) * (w // patch_size)
	all_patches = np.zeros((n_patches,patch_size, patch_size,c))
	patch_iter = 0
	for hi in np.arange(0,h,patch_size):
		for wi in np.arange(0,w,patch_size):
			all_patches[patch_iter,:,:,:] = image[hi:hi+patch_size,wi:wi+patch_size,:]
			patch_iter+=1
	if c == 1:
		return all_patches[:,:,:,0]
	if c == 3:
		return all_patches

## this function combines the input array of size N x  patch_size x patch_size x 3 (colored patches) or 
## of size N x patch_size x patch_size (grayscale patches) into an image of size
## h x w x 3 (colored image) or h x w (grayscale image)
## CAUTION: ensure that the h and w of the output image is an integral
## multiple of patch_size. Use the height and width of the image given to you.
def combine_patches(patches, h, w):
	if len(patches.shape) == 4:
		N,patch_size,_,c = patches.shape
		assert c  == 3
	elif len(patches.shape) == 3:
		N,patch_size,_ = patches.shape
		patches = patches[:,:,:,None]
		c = 1
	else:
		print('ERROR: unexpected size for patches array')
		return NotImplementedError
	if h % patch_size != 0:
		print('ERROR: the height of the image is expected to be an integral multiple of patch_size')
		return NotImplementedError
	if w % patch_size != 0:
		print('ERROR: the height of the image is expected to be an integral multiple of patch_size')
		return NotImplementedError    
	if (h // patch_size) * (w // patch_size) != N:
		print('ERROR: number of patches is inconsistent with given h,w')
		return NotImplementedError   
	
	n_patches = (h // patch_size) * (w // patch_size)
	image = np.zeros((h,w,c))
	patch_iter = 0
	for hi in np.arange(0,h,patch_size):
		for wi in np.arange(0,w,patch_size):
			image[hi:hi+patch_size,wi:wi+patch_size,:] = patches[patch_iter,:,:,:]
			patch_iter+=1
	if c == 1:
		return image[:,:,0]
	if c == 3:
		return image
