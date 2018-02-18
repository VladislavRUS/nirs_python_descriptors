"""Image model descriptors"""

import math
from sklearn.neighbors import KNeighborsClassifier
from scipy import misc
from skimage import exposure
import random
import os
import codecs
import math
import numpy as np
from matplotlib.mlab import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from scipy import ndimage
from skimage import exposure, img_as_float
import matplotlib.pyplot as plt
import scipy.misc
from skimage.draw import ellipse_perimeter

def get_error_prob(classes, predictions, examples_number):
    assert len(classes) == len(predictions)

    errors = 0

    for i in range(0, len(classes)):
        if classes[i] != predictions[i]:
            errors += 1

    return errors / examples_number


def get_classes(class_params, samples_params):
    classes = []

    max_class_idx = class_params['to'] - class_params['from']
    max_sample_idx = samples_params['to'] - samples_params['from']

    for i in range(0, max_class_idx):
        for j in range(0, max_sample_idx):
            classes.append(i)

    return classes


def AFS(vector_complex, vector_complex_base, max_val):
    afs_type = 2
    output_sum = 0
    output_value = 0
    max_val_base_1 = 1.0 / max_val
    max_val_base_2 = max_val_base_1 * max_val_base_1

    for i in range(0, len(vector_complex)):
        z = vector_complex[i]
        z_base = vector_complex_base[i]

        val = np.absolute(z)
        val_base = np.absolute(z_base)

        ang = np.angle(z)
        ang_base = np.angle(z_base)

        diff = ang - ang_base

        if diff < -180:
            diff = diff + 180

        elif diff > 180:
            diff = diff - 180

        angle_coefficient = 0.5 * (1 + math.cos(diff))

        if afs_type == 1:
            val = val * (val_base * max_val_base_1)

        elif afs_type == 2:
            val = val * (val_base * val_base * max_val_base_2)

        output_value += angle_coefficient * val
        output_sum += val

    return output_value / output_sum


def calc_features(vector, eigen_vectors, features_number):
    max_value = np.amax(np.absolute(eigen_vectors).flatten())

    feature = []

    for i in range(0, features_number):
        feature.append(AFS(np.conjugate(vector), eigen_vectors[:, i], max_value))

    return feature


def apply_pca(vectors, eigen_vectors, vec_mean, features_number):
    vectors = np.transpose(vectors)

    features = []

    for i in range(0, len(vectors)):
        vector = np.subtract(vectors[i, :], vec_mean)
        features.append(calc_features(vector, eigen_vectors, features_number))

    return features


def pca(vectors):
    pca_results = PCA(vectors, standardize=True)
    vec_mean = np.mean(np.transpose(vectors), axis=0)

    return pca_results.Y, vec_mean


def image_2_vector(image):
    re = ndimage.sobel(image, axis=-1)
    im = ndimage.sobel(image, axis=0)

    re = re.flatten()
    im = im.flatten()

    return re - 1j * im


def scale_matrix(matrix, scale_factor):
    width = len(matrix[0])
    height = len(matrix)

    ver = math.floor(height / scale_factor)
    hor = math.floor(width / scale_factor)
    coef = 1 / (math.pow(scale_factor, 2))

    result = np.zeros((ver, hor))

    for i in range(0, ver):
        for j in range(0, hor):
            val = 0

            for k in range(0, scale_factor):
                for l in range(0, scale_factor):
                    val += matrix[scale_factor * i + k][scale_factor * j + l]

            result[i][j] = val * coef

    return result


def read_pgm(pgmf):
    pgmf.readline()  # P5
    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255

    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)
    return raster


def matrixFromFile(fileName):
    f = open(fileName, mode='r')
    lines = f.readlines()

    matrix = []
    for line in lines:
        line = line.replace('i', 'j').split(',')

        row = []
        for number in line:
            row.append(complex(number))

        matrix.append(row)

    return np.array(matrix)


def vecMeanFromFile():
    fileName = 'vecMean.txt'

    f = open(fileName, mode='r')

    line = f.readlines()[0]

    line = line.replace('i', 'j').split(',')

    row = []
    for number in line:
        row.append(complex(number))

    return np.array(row)


def load_data(base_dir, class_params, samples_params):
    folders = os.listdir(base_dir)
    folders = folders[class_params['from']:class_params['to']]

    vectors = []

    box_size = 150

    for folder in folders:
        images = os.listdir(base_dir + '/' + folder)
        images = [img for img in images if not img.endswith('.txt')]
        images = images[samples_params['from']:samples_params['to']]

        for image in images:
            im_path = base_dir + '/' + folder + '/' + image
            ext = image[image.rfind('.'):]
            file_name = im_path.replace(ext, '.txt')
            
            if not os.path.isfile(file_name):
                os.system("python poincare.py " + im_path + " 16 1 --smooth")

            im_file = open(file_name, 'r')
            coordinates = im_file.readline().split(':')
            x = int(float(coordinates[1]))
            y = int(float(coordinates[0]))
            im_file.close()

            raster = ndimage.imread(im_path)

            raster = raster[x - box_size: x + box_size, y - box_size: y + box_size]
            
            print(image, x, y, box_size)
            assert raster.size != 0
            
            #scaled = scipy.misc.imresize(raster, 0.5)

            #img_eq = exposure.equalize_hist(img_as_float(raster))

            vectors.append(image_2_vector(raster))
            misc.imsave("./partials/" + image + ".png", raster)

            print("Processed image: " + image)

    return np.transpose(np.array(vectors))


def start(features_number, base_dir):
    class_params = {'from': 0, 'to': 4}
    train_samples_params = {'from': 0, 'to': 10}
    test_samples_params = {'from': 10, 'to': 15}
    classifier = 'KNN'

    train_vectors = load_data(base_dir, class_params, train_samples_params)
    eigen_vectors, vec_mean = pca(train_vectors)
    train_features = apply_pca(train_vectors, eigen_vectors, vec_mean, features_number)

    test_vectors = load_data(base_dir, class_params, test_samples_params)
    test_features = apply_pca(test_vectors, eigen_vectors, vec_mean, features_number)

    train_classes = get_classes(class_params, train_samples_params)

    predictions = None

    if classifier == 'KNN':
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(train_features, train_classes)
        predictions = knn.predict(test_features)

    elif classifier == 'SVM':
        clf = svm.SVC()
        clf.fit(train_features, train_classes)
        predictions = clf.predict(test_features)

    test_classes = get_classes(class_params, test_samples_params)
    error_prob = get_error_prob(test_classes, predictions, len(test_classes))

    print(features_number, base_dir, classifier)
    print('True probability: ', 1 - error_prob)


# for base_dir in ['./DB1_B', './DB2_B', './DB3_B', './DB4_B']:
for base_dir in ['./DB5_B_SHARPEN']:
    for features_number in [ 6 ]:
        start(features_number, base_dir)
