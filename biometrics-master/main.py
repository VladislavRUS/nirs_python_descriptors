import os

images = os.listdir('./images')

for image in images:
    dot_position = image.find('.')
    gabor_name = image[:dot_position] + '_gabor' + image[dot_position:]
    os.system("python poincare.py ./images/" + image + " 16 1 --smooth --save")