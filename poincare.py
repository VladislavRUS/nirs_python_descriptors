# Metody biometryczne
# Przemyslaw Pastuszka

from PIL import Image, ImageDraw
import utils
import argparse
import math
import os
from scipy import ndimage

signum = lambda x: -1 if x < 0 else 1

cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

def get_angle(left, right):
    angle = left - right
    if abs(angle) > 180:
        angle = -1 * signum(angle) * (360 - abs(angle))
    return angle

def poincare_index_at(i, j, angles, tolerance):
    deg_angles = [math.degrees(angles[i - k][j - l]) % 180 for k, l in cells]
    index = 0
    for k in range(0, 8):
        if abs(get_angle(deg_angles[k], deg_angles[k + 1])) > 90:
            deg_angles[k + 1] += 180
        index += get_angle(deg_angles[k], deg_angles[k + 1])

    if 180 - tolerance <= index and index <= 180 + tolerance:
        return "loop"
    if -180 - tolerance <= index and index <= -180 + tolerance:
        return "delta"
    if 360 - tolerance <= index and index <= 360 + tolerance:
        return "whorl"
    return "none"

def calculate_singularities(im, angles, tolerance, W, im_path):
    (x, y) = im.size
    result = im.convert("RGB")

    draw = ImageDraw.Draw(result)

    colors = {"loop" : (150, 0, 0), "delta" : (0, 150, 0), "whorl": (0, 0, 150)}

    singularities = []

    raster = ndimage.imread(im_path)

    shape = list(raster.shape)

    center = []
    center.append(shape[1] / 2)
    center.append(shape[0] / 2)

    for i in range(1, len(angles) - 1):
        for j in range(1, len(angles[i]) - 1):
            singularity = poincare_index_at(i, j, angles, tolerance)
            if singularity != "none":
                x_0 = i * W
                y_0 = j * W
                x_1 = (i + 1) * W
                y_1 = (j + 1) * W

                singularities.append([x_0, y_0, x_1, y_1])
                draw.ellipse([(x_0, y_0), (x_1, y_1)], outline = colors[singularity])

    avg = lambda arr: ((arr[0] + arr[2]) / 2, (arr[1] + arr[3]) / 2)
    distance = lambda p1, p2: ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1/2)

    min_s = singularities[0]
    min_distance = distance(center, avg(min_s))

    for s in singularities:
        d = distance(center, avg(s))
        if d < min_distance:
            min_distance = d
            min_s = s

    dot_idx = im_path.rfind('.')
    text_file = open(im_path[:dot_idx] + ".txt", "w")
    avg_min_s = avg(min_s)
    text_file.write(str(avg_min_s[0]) + ":" + str(avg_min_s[1]))
    text_file.close()

    draw.ellipse([(min_s[0], min_s[1]), (min_s[2], min_s[3])], outline = (255, 255, 255))

    del draw

    return result

parser = argparse.ArgumentParser(description="Singularities with Poincare index")
parser.add_argument("image", nargs=1, help = "Path to image")
parser.add_argument("block_size", nargs=1, help = "Block size")
parser.add_argument("tolerance", nargs=1, help = "Tolerance for Poincare index")
parser.add_argument('--smooth', "-s", action='store_true', help = "Use Gauss for smoothing")
parser.add_argument("--save", action='store_true', help = "Save result image as src_poincare.gif")
args = parser.parse_args()

im = Image.open(args.image[0])
im = im.convert("L")  # covert to grayscale

W = int(args.block_size[0])

f = lambda x, y: 2 * x * y
g = lambda x, y: x ** 2 - y ** 2

angles = utils.calculate_angles(im, W, f, g)
if args.smooth:
    angles = utils.smooth_angles(angles)

result = calculate_singularities(im, angles, int(args.tolerance[0]), W, args.image[0])
#result.show()

if args.save:
    base_image_name = os.path.splitext(os.path.basename(args.image[0]))[0]
    result.save(base_image_name + "_poincare.gif", "GIF")

