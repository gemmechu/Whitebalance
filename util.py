import config
from boot import app


def get_distance(img1, img2):
    return 200


def is_same(img1, img2, threshold=0.5):
    distance = get_distance(img1, img2)
    return distance, distance <= threshold
