import os

images = os.listdir('./images')

for image in images:
    dot_position = image.find('.')
    gabor_name = image[:dot_position] + '_gabor' + image[dot_position:]
    os.system("python gabor ./images/" + image + " ./gabor-images/" + gabor_name + " -b")