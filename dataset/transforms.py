import numpy as np


class Window:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __call__(self, image):
        image = np.clip(image, self.lower, self.upper)

        return image

class Normalize:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, img):
        img = (img - self.low) / (self.high - self.low)
        img = img * 2 - 1
        return img

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
