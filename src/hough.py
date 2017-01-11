from collections import defaultdict, namedtuple
from math import cos, sin, ceil, sqrt, pi

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.feature import peak_local_max


def discrete_direction(ndir, alpha):
    bucket_width = 2*pi/ndir
    return int((alpha / bucket_width + 0.5 * bucket_width) % ndir)


scales = [2**i for i in range(-3, 3)]
rotations = [i * (pi / 32) for i in range(0, 64)]


def hough_learn(img):
    rtable = defaultdict(list)
    mag_sum = img.magnitudes.sum()
    if mag_sum == 0:
        raise RuntimeError('no edges detected')
    center = (
        sum(
            mag * np.array([x, y])
            for ((x, y), mag) in np.ndenumerate(img.magnitudes)) /
        mag_sum
    )
    print 'center:', center[0], 'x', center[1]

    for (x, y), mag in np.ndenumerate(img.magnitudes):
        if mag <= 0:
            continue
        rx = int(center[0] - x)
        ry = int(center[1] - y)
        phi = discrete_direction(64, img.angles[x, y])
        rtable[phi].append((rx, ry))
    return rtable


HoughDetectionResult = namedtuple(
    'HoughDetectionResult',
    ['accumulator', 'candidates'])


def hough_detect(rtable, img, on_progress=None):
    acc = np.zeros((len(scales), len(rotations), img.w, img.h))
    num_pixels = img.w * img.h

    for (x, y), mag in np.ndenumerate(img.magnitudes):
        if mag < 1:
            continue

        if on_progress is not None:
            on_progress(100 * (x * img.h + y) / num_pixels)

        for rot_idx, rot in enumerate(rotations):
            c, s = np.cos(rot), np.sin(rot)
            Rot = np.array([[c, -s], [s, c]])

            alpha = (img.angles[x, y] + rot) % (2 * pi)
            rindex = discrete_direction(64, alpha)
            for r in rtable[rindex]:
                r = np.dot(Rot, r)
                for scale_idx, scale in enumerate(scales):
                    r_scaled = scale * r
                    center_x = int(x + r_scaled[0])
                    if not (0 <= center_x < img.w):
                        continue
                    center_y = int(y + r_scaled[1])
                    if not (0 <= center_y < img.h):
                        continue
                    acc[scale_idx, rot_idx, center_x, center_y] += 1

    acc = gaussian_filter(acc, 2)
    max_coordinates = peak_local_max(acc, min_distance=2, num_peaks=10)
    return HoughDetectionResult(
        acc,
        [
            (scales[max_scale_idx], rotations[max_rot_idx], cx, cy)
            for max_scale_idx, max_rot_idx, cx, cy in max_coordinates
        ])
