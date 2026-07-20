
import numpy as np
import cv2


COCO_WEIGHTS = np.array([
    0.5,0.3,0.3,0.3,0.3,
    1,1,
    1.5,1.5,
    1.8,1.8,
    2,2,
    2.5,2.5,
    3,3
])


def normalize_pose(kp):

    if kp is None:
        return None

    kp = kp.copy()


    # hip center

    center = (
        kp[11,:2] +
        kp[12,:2]
    ) / 2


    kp[:,:2] -= center


    # shoulder scale

    scale = np.linalg.norm(
        kp[5,:2] -
        kp[6,:2]
    )


    if scale == 0:
        return None


    kp[:,:2] /= scale


    return kp



def affine_transform(source, target):

    points = [
        5,6,
        11,12,
        13,14,
        15,16
    ]


    src = source[points,:2].astype(
        np.float32
    )

    dst = target[points,:2].astype(
        np.float32
    )


    matrix,_ = cv2.estimateAffinePartial2D(
        src,
        dst
    )


    if matrix is None:
        return None


    aligned = cv2.transform(
        np.array([source[:,:2]]),
        matrix
    )[0]


    return aligned



def cosine_similarity(a,b):

    a = a.flatten()
    b = b.flatten()


    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )


    if denominator == 0:
        return 0


    return np.dot(a,b) / denominator



def weighted_distance(a,b):

    distances = np.linalg.norm(
        a-b,
        axis=1
    )


    return np.mean(distances)
