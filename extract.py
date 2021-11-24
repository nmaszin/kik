#!/usr/bin/env python3

import sys
import math
from pathlib import Path
import numpy as np
import skimage as sk
from skimage import io, filters, feature, morphology, measure, color, segmentation, util
from utils import Image
from matplotlib import pyplot as plt
from statistics import mean

def minmax(l):
    return min(l), max(l)

class DirectoryWriter:
    def __init__(self, directory, ext):
        self.directory_path = Path(directory)
        self.ext = ext
        self.current_id = 1
        self.directory_path.mkdir(exist_ok=True)
    
    def __iter__(self):
        all_files_in_directory = filter(lambda x: x.is_file(), self.directory_path.iterdir())
        all_filenames_without_suffix = map(lambda x: x.stem, all_files_in_directory)
        max_file_id = max(map(int, all_filenames_without_suffix), default=0)
        self.current_id = max_file_id + 1
        return self

    def __next__(self):
        value = self.current_id
        self.current_id += 1
        return self.directory_path / f'{value}.{self.ext}'

def extract_contours(image):
    data = image.data
    data = color.rgb2gray(data)
    data = sk.feature.canny(data)
    data = sk.morphology.dilation(data, sk.morphology.disk(4))
    data = sk.segmentation.flood(data, (0, 0))
    data = sk.util.invert(data)
    data = sk.morphology.remove_small_objects(data)

    contours = measure.find_contours(data, 0.8)
    return contours

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def remove_wrapped_contours(contours):
    result = []
    centers = [(mean(c[0], mean(c[1]))) for c in contours]
    radiuses = [mean(map(lambda x: dist(x, cent), cont)) for cont, cent in zip(contours, centers)]

    for i, contour in enumerate(contours):
        for j, other in enumerate(contours[i:], start=i):
            pass


def get_contour_wrapper_rect(contour):
    min_y, max_y = minmax(contour[:, 0])
    min_x, max_x = minmax(contour[:, 1])

    return (
        (math.floor(min_y), math.floor(min_x)),
        (math.ceil(max_y), math.ceil(max_x))
    )

def main():
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <image> <output_directory>')
        print('Where:')
        print('\t<image> - image which contains pattern repeated many times')
        print('\t<output_directory> - directory in which each extracted instance will be saved')
        sys.exit(1)

    image_filename, output_directory = sys.argv[1:]

    image = Image.from_file(image_filename)
    contours = extract_contours(image)
    writer = DirectoryWriter(output_directory, 'jpg')

    for obj, path in zip(contours, writer):
        rect = get_contour_wrapper_rect(obj)
        filename = str(path)
        image.crop(rect).save(filename)


if __name__ == '__main__':
    main()

