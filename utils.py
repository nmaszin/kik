from skimage import io, transform
import skimage as sk
from matplotlib import pyplot as plt

class Image:
    def __init__(self, data, gray=False):
        self.data = sk.img_as_float(data)
        self.is_gray = gray

    @staticmethod
    def from_file(filename):
        return Image(sk.io.imread(filename))

    def crop(self, area):
        (top, left), (bottom, right) = area
        data = self.data[top:bottom, left:right]
        return Image(data)

    def resize(self, shape):
        data = sk.transform.resize(self.data, shape)
        return Image(data)

    def show(self):
        plt.figure()
        if self.is_gray:
            plt.gray()
        plt.imshow(self.data)
        plt.show()

    def show_with_contours(self, contours):
        COLORS = 'cmyrgb'
        color = lambda i: COLORS[i % len(COLORS)]

        plt.figure()
        if self.is_gray:
            plt.gray()
        plt.imshow(self.data)
        for i, contour in enumerate(contours):
            plt.plot(contour[:, 1], contour[:, 0], color(i), linewidth=1)
        plt.show()

    def save(self, filename):
        io.imsave(filename, sk.img_as_ubyte(self.data))
    
    def save_with_contours(self, contours, filename):
        COLORS = 'cmyrgb'
        color = lambda i: COLORS[i % len(COLORS)]

        plt.figure()
        if self.is_gray:
            plt.gray()
        plt.imshow(self.data)
        for i, contour in enumerate(contours):
            plt.plot(contour[:, 1], contour[:, 0], color(i), linewidth=1)
        plt.savefig(filename)


def get_contour_wrapper_rect(contour):
    min_y, max_y = minmax(contour[:, 0])
    min_x, max_x = minmax(contour[:, 1])

    return (
        (math.floor(min_y), math.floor(min_x)),
        (math.ceil(max_y), math.ceil(max_x))
    )
