import numpy as np

N = 730
img_idx = np.arange(N)
print(f'img_idx:\n{img_idx}')
permutation = np.random.permutation(img_idx)
img_idx = img_idx[permutation]
print(f'After shuffling:\n{img_idx}')


with open('random.txt', 'w') as fw:
    for i in range(N):
        fw.write(str(img_idx[i])+'\n')
    