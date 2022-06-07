import numpy as np
import imageio


# a = np.ones((4, 2, 1))
# img_size = a.shape
# print(a.shape)
# b = np.pad(a, ((2, 2), (3, 3)))
# print(b)


img = imageio.imread('./video0/mask/mask_0000.png')

print(img.shape)
