#!/usr/bin/env python3

import sys
import statistics
from pathlib import Path
from utils import Image

def correct_contrast(image):
    return image

def calculate_best_resolution(images):
    heights = list(map(lambda i: i.data.shape[0], images))
    widths = list(map(lambda i: i.data.shape[1], images))

    return tuple(map(
        lambda x: int(statistics.mean(x)),
        (heights, widths))
    )

def main():
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <input_dir> <output_dir>')
        sys.exit(1)

    input_directory, output_directory = map(Path, sys.argv[1:])
    output_directory.mkdir(exist_ok=True)

    paths = list(input_directory.iterdir())
    images = list(map(Image.from_file, paths))
    # resolution = calculate_best_resolution(images)
    resolution = (121, 121)

    for image, path in zip(images, paths):
        image = image.resize(resolution)
        image.save(output_directory / path.name)

if __name__ == '__main__':
    main()

