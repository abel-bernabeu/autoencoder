import cv2
import numpy as np
import PIL

"""This module contains some transformations for applying to the training of autoencoders. Notice that some transformations
have random and non-random variants. Random variant is for training while Non-Random variant is for validation."""


def get_item(array, cv2_img):
    """This helper function gets an item from an array depending on the color of the pixel at the top-left corner"""
    n = len(array)
    pix = int(cv2_img[0, 0, 0]) + int(cv2_img[0, 0, 1]) + int(cv2_img[0, 0, 2])
    return array[(pix * n) // 766]


class AddRandomGaussianNoise(object):
    """Transform to add random noise."""

    def __init__(self, mean=0., std=25.5):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        cv2_x = cv2.cvtColor(np.asarray(x), cv2.COLOR_RGB2BGR)
        cv2_x = cv2_x.astype(np.float32)
        noise = np.random.normal(self.mean, self.std, cv2_x.shape)
        cv2_x = cv2_x + noise
        cv2_x = np.clip(cv2_x, 0, 255)
        cv2_x = np.round(cv2_x, decimals=0).astype(np.uint8)
        cv2_x = cv2.cvtColor(cv2_x, cv2.COLOR_BGR2RGB)
        x = PIL.Image.fromarray(cv2_x)
        return x


class AddGaussianNoise(object):
    """Transform to add non-random noise."""

    def __init__(self, noises):
        self.noises = noises

    def __call__(self, x):
        cv2_x = cv2.cvtColor(np.asarray(x), cv2.COLOR_RGB2BGR)
        cv2_x = cv2_x.astype(np.float32)
        cv2_x = cv2_x + get_item(self.noises, cv2_x)
        cv2_x = np.clip(cv2_x, 0, 255)
        cv2_x = np.round(cv2_x, decimals=0).astype(np.uint8)
        cv2_x = cv2.cvtColor(cv2_x, cv2.COLOR_BGR2RGB)
        x = PIL.Image.fromarray(cv2_x)
        return x


class ConvertToGray(object):
    """Transform to convert to gray."""

    def __init__(self):
        pass

    def __call__(self, x):
        cv2_x = cv2.cvtColor(np.asarray(x), cv2.COLOR_RGB2GRAY)
        cv2_x = cv2.cvtColor(cv2_x, cv2.COLOR_GRAY2RGB)
        x = PIL.Image.fromarray(cv2_x)
        return x


class Blur(object):
    """Transform to add blurring."""

    def __init__(self, ksize=(5, 5)):
        self.ksize = ksize

    def __call__(self, x):
        cv2_x = cv2.cvtColor(np.asarray(x), cv2.COLOR_RGB2BGR)
        cv2_x = cv2.GaussianBlur(cv2_x, self.ksize, 0, borderType=cv2.BORDER_REPLICATE)
        cv2_x = cv2.cvtColor(cv2_x, cv2.COLOR_BGR2RGB)
        x = PIL.Image.fromarray(cv2_x)
        return x


class JPEG(object):
    """Transform to add JPEG artifacts."""

    def __init__(self, quality=20):
        self.encparams = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]

    def __call__(self, x):
        cv2_x = cv2.cvtColor(np.asarray(x), cv2.COLOR_RGB2BGR)
        result, encimg = cv2.imencode('.jpg', cv2_x, self.encparams)
        cv2_x = cv2.imdecode(encimg, 1)
        cv2_x = cv2.cvtColor(cv2_x, cv2.COLOR_BGR2RGB)
        x = PIL.Image.fromarray(cv2_x)
        return x


class AddRandomSquares(object):
    """Transform to add random squares."""

    def __init__(self, min_n=3, max_n=4, min_size=8, max_size=34, color=(255, 0, 255)):
        self.min_n = min_n
        self.max_n = max_n
        self.min_size = min_size
        self.max_size = max_size
        self.color = color

    def __call__(self, x):
        cv2_x = cv2.cvtColor(np.asarray(x), cv2.COLOR_RGB2BGR)
        squares = self.get_squares(cv2_x.shape[0], cv2_x.shape[1])
        for square in squares:
            size, x, y = square
            cv2_x = cv2.rectangle(cv2_x, (x, y), (x + size, y + size), self.color, -1)
        cv2_x = cv2.cvtColor(cv2_x, cv2.COLOR_BGR2RGB)
        x = PIL.Image.fromarray(cv2_x)
        return x

    def get_squares(self, height, width):
        squares = []
        n = np.random.randint(self.min_n, self.max_n + 1)
        for i in range(n):
            size = np.random.randint(self.min_size, self.max_size + 1)
            x = np.random.randint(0, width - size)
            y = np.random.randint(0, height - size)
            squares.append((size, x, y))
        return squares


class AddSquares(object):
    """Transform to add non-random squares."""

    def __init__(self, squares, color=(255, 0, 255)):
        self.squares = squares
        self.color = color

    def __call__(self, x):
        cv2_x = cv2.cvtColor(np.asarray(x), cv2.COLOR_RGB2BGR)
        squares = get_item(self.squares, cv2_x)
        for square in squares:
            size, x, y = square
            cv2_x = cv2.rectangle(cv2_x, (x, y), (x + size, y + size), self.color, -1)
        cv2_x = cv2.cvtColor(cv2_x, cv2.COLOR_BGR2RGB)
        x = PIL.Image.fromarray(cv2_x)
        return x


class Pixelate(object):
    """Transform to add pixelation."""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        size = x.size
        scaled_size = (i // self.scale for i in x.size)

        return x.resize(scaled_size, PIL.Image.BICUBIC).resize(size, PIL.Image.NEAREST)