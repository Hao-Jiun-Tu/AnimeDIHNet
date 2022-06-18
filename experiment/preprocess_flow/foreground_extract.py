import numpy as np
import cv2 as cv
import os

real_name = 'real0.png'
# comp_name = 'comp0.png'
mask_name = 'mask0.png'
tran_name = 'tran0.png' 

real = cv.imread(real_name, cv.IMREAD_UNCHANGED)
tran = cv.imread(tran_name, cv.IMREAD_UNCHANGED)
mask = cv.imread(mask_name, cv.IMREAD_UNCHANGED)
mask = np.expand_dims(mask, 2)

f_real = (((real/255.0)*(mask/255.0)+(1.0-(mask/255.0)))*255.0).astype('uint8')
f_tran = (((tran/255.0)*(mask/255.0)+(1.0-(mask/255.0)))*255.0).astype('uint8')
cv.imshow('foreground real image', f_real)
cv.waitKey()
cv.destroyAllWindows()
cv.imshow('foreground color transfer image', f_tran)
cv.waitKey()
cv.destroyAllWindows()
cv.imwrite('foreground/real.png', f_real)
cv.imwrite('foreground/tran.png', f_tran)