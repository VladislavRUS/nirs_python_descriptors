from scipy import ndimage
from skimage.draw import ellipse
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


raster = ndimage.imread('./102_1.tif')
center = ndimage.measurements.center_of_mass(raster)

print(center)

rr, cc = ellipse(center[0], center[1], 10, 10, rotation=np.deg2rad(30))

raster[rr, cc] = 1

plt.imshow(raster)
plt.show()