#!/usr/bin/env python3

import sys
import math
import cv2
import numpy as np
import skimage as sk
from skimage import io, filters, feature, morphology, measure, color, segmentation, util, exposure, transform
from utils import Image, get_contour_wrapper_rect
from matplotlib import pyplot as plt
from statistics import mean, stdev

def minmax(l):
    return min(l), max(l)

def crop(v, low, high):
    return np.min(np.max(v, low), high)

def dist(p1, p2):
    return math.sqrt(sum((p1 - p2) ** 2))

def approximate_polygon(cont):
    return sk.measure.approximate_polygon(cont, tolerance=5)[:-1]

class CalculateLineException(Exception): pass

def calculate_line(p1, p2):
    if p1.tolist() == p2.tolist():
        raise CalculateLineException('Punkty są takie same')

    y1, x1 = p1
    y2, x2 = p2
    if x1 != x2:
        a = (y1 - y2) / (x1 - x2)
    else:
        # Vertical line is almost vertical after that
        a = (y1 - y2) / (x1 - x2 + 997 * 10 ** (-12))

    b = y1 - a * x1
    return (a, b)


def get_center(contour):
    return np.array(list(map(mean, zip(*contour))))

def radius_sd_norm(contour):
    center = get_center(contour)
    distances = list(map(lambda x: dist(x, center), contour))
    return stdev(distances) / mean(distances)

def is_circle(contour):
    return radius_sd_norm(contour) < 0.25

def is_x(contour):
    return radius_sd_norm(contour) > 0.3

def is_trapezoid(cont):
    cont = approximate_polygon(cont)
    if radius_sd_norm(cont) > 0.25:
        return False
    return len(cont) in [4, 5]

def remove_rubbish_corners_of_trapezoid(trapezoid):
    original = corners = trapezoid.tolist()
    corners = list(map(np.array, corners))
    corners += [corners[0]]

    indexes = list(range(len(trapezoid)))
    sorted_corners_indexes = sorted(indexes, key=lambda i: dist(corners[i], corners[i + 1]), reverse=True)
    to_remove = sorted(sorted_corners_indexes[4:], reverse=True)
    for index in to_remove:
        del original[index]

    return np.array(original)


def to_list(func):
    def wrapper(*args, **kwargs):
        return list(func(*args, **kwargs))
    return wrapper


def make_binary_image(image):
    image = Image(image.data, gray=True)
    image.data = sk.exposure.rescale_intensity(image.data, (0.3, 0.7))
    image.data = sk.color.rgb2gray(image.data)
    image.data = sk.img_as_ubyte(image.data)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    bg = cv2.morphologyEx(image.data, cv2.MORPH_DILATE, se)
    image.data = cv2.divide(image.data, bg, scale=255)
    image.data = cv2.threshold(image.data, 0, 255, cv2.THRESH_OTSU)[1] 
    return image

def extract_games(image):
    data = image.data
    data = sk.feature.canny(data)
    data = sk.morphology.dilation(data, sk.morphology.disk(16))
    data = sk.morphology.remove_small_holes(data, 30000)
    data = sk.morphology.remove_small_objects(data, 15000)

    contours = sk.measure.find_contours(data, 0.8)
    rects = map(get_contour_wrapper_rect, contours)
    return rects


def extract_contours(image):
    data = image.data
    data = sk.filters.median(data, sk.morphology.disk(1))
    data = sk.feature.canny(data)
    data = sk.morphology.dilation(data, sk.morphology.disk(1))
    data = sk.morphology.remove_small_holes(data)

    data2 = sk.segmentation.flood(data, (0, 0))
    data3 = np.invert(data ^ data2)
    
    # Image(data, gray=True).show()
    # Image(data2, gray=True).show()
    # Image(data3, gray=True).show()

    data3 = sk.morphology.erosion(data3, sk.morphology.disk(1.4))
    data3 = sk.morphology.remove_small_objects(data3, 400)
    data3 = sk.morphology.remove_small_holes(data3, 400)
    data = data2 + data3


    contours = sk.measure.find_contours(data, 0.8)
    # Image(data, gray=True).show_with_contours(contours)
    return contours

def segregate_objects(contours):
    board, *contours = sorted(contours, key=lambda x: len(x), reverse=True)
    central_fields = filter_remove(is_trapezoid, contours)
    crosses = filter_remove(is_x, contours)
    circles = filter_remove(is_circle, contours)
    return board, circles, crosses, central_fields


def get_contour_wrapper_rect(contour):
    min_y, max_y = minmax(contour[:, 0])
    min_x, max_x = minmax(contour[:, 1])

    return (
        (math.floor(min_y), math.floor(min_x)),
        (math.ceil(max_y), math.ceil(max_x))
    )

@to_list
def get_corners(contour):
    contour = contour.tolist()
    cyclic = contour + contour[:2]
    for i in range(len(contour)):
        a, b, c = cyclic[i:(i + 3)]
        a, b, c = map(np.array, (a, b, c))
        angle = math.degrees(math.acos(np.dot(a - b, c - b) / (dist(a, b) * dist(b, c))))
        if angle < 180:
            yield b


def filter_remove(callback, lst):
    l = list(filter(lambda x: callback(x[1]), enumerate(lst)))
    if len(l) == 0:
        return []

    indexes, data = zip(*l)
    indexes = sorted(indexes, reverse=True)
    for index in indexes:
        del lst[index]

    return data

def is_point_below_line(point, line):
    yp, xp = point
    p1, p2 = line
    a, b = calculate_line(p1, p2)
    result = yp < a * xp + b
    return result

def logical_xor(a, b,):
    return not b if a else bool(b)

def is_point_on_the_right_of_line(point, line):
    p1, p2 = line
    a, b = calculate_line(p1, p2)
    return logical_xor(
        a <= 0,
        is_point_below_line(point, line)
    )

def create_objects_list(crosses, circles):
    objects = []
    for cross in crosses:
        objects.append(('X', get_center(cross)))
    for circle in circles:
        objects.append(('O', get_center(circle)))
    return objects
    
def map_objects(objects, rows, columns):
    objects_mapping = [None for _ in range(len(objects))]
    for index, (t, point) in enumerate(objects):
        row_index = len(rows) - sum(map(lambda l: is_point_below_line(point, l), rows))
        column_index = sum(map(lambda l: is_point_on_the_right_of_line(point, l), columns))
        objects_mapping[index] = (t, (row_index, column_index))

    return objects_mapping

def create_board_from_objects_mapping(objects, rows_count, columns_count):
    board = [[' ' for _ in range(columns_count)] for _ in range(rows_count)]
    for t, (y, x) in objects:
        board[y][x] = t
    return board

def draw_board(board):
    for i, row in enumerate(board):
        if i != 0:
            print('-' * (4 * len(row) - 1))
        print('|'.join(map(lambda e: f' {e} ', row)))
    print()

def main():
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <image>')
        print('Where:')
        print('\t<image> - image which contains tic tac toe board')
        sys.exit(1)

    image_filename = sys.argv[1]
    original_image = Image.from_file(image_filename)
    binary_image = make_binary_image(original_image)
    
    for i, game_rect in enumerate(extract_games(binary_image), start=1):
        print(f'Plansza nr {i}')

        resolution = (400, 400)
        orig = original_image.crop(game_rect).resize(resolution)
        game = binary_image.crop(game_rect).resize(resolution)

        contours = extract_contours(game)
        contours = list(filter(lambda x: len(x) > 130, contours))

        board, circles, crosses, central_fields = segregate_objects(contours)
        orig.show_with_contours([board])
        orig.show_with_contours(circles)
        orig.show_with_contours(crosses)
        orig.show_with_contours(map(approximate_polygon, central_fields))


        if len(central_fields) == 0:
            print('Brakuje pól na planszy\n')
            continue

        field = remove_rubbish_corners_of_trapezoid(approximate_polygon(central_fields[0]))
        index = min(enumerate(field), key=lambda f: dist(f[1], np.array([0, 0])))[0]
        field = np.array(field[index:].tolist() + field[:index].tolist())

        rows_lines = [field[:2], field[2:4]]
        columns_lines = [field[1:3], np.array([field[3], field[0]])]

        objects = create_objects_list(crosses, circles)
        objects_mapping = map_objects(objects, rows_lines, columns_lines)
        board = create_board_from_objects_mapping(objects_mapping, 3, 3)
        draw_board(board)
        orig.show()

if __name__ == '__main__':
    main()

