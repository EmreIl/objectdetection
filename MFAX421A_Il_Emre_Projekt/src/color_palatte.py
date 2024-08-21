import numpy as np


class ColorPalette(object):
    BLUE = {
        'lower': np.array([110, 50, 50]),
        'upper': np.array([130, 255, 255])
    }
    ORANGE = {
        'lower': np.array([10, 100, 100]),
        'upper': np.array([20, 255, 255])
    }
    BLACK = {
        'lower': np.array([0, 0, 0]),
        'upper': np.array([180, 255, 30])
    }
    BROWN = {
        'lower': np.array([0, 60, 60]),
        'upper': np.array([30, 255, 255])
    }
    GREEN = {
        'lower': np.array([40, 40, 40]),
        'upper': np.array([80, 255, 255])
    }