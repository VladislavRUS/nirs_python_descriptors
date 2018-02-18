from scipy import ndimage
from scipy import misc
import os

images = os.listdir('./')

for image in images:
    if (image.endswith('.png')):
        img = ndimage.imread(image)
        img_resized = misc.imresize(img, 50)
        misc.imsave(image, img_resized)