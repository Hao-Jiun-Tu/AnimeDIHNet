import numpy as np

a = np.arange(24).reshape((4,3,2))
b = np.arange(12).reshape((4,3,1))

print(a, a.shape)
print(b, b.shape)

c = np.concatenate((a, b), axis=2)
print(c, c.shape)

print(c[:,:,2])