

import numpy as np


def cosine_similarity(a,b):

    a=np.array(a)
    b=np.array(b)

    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


def weighted_distance(a,b):

    a=np.array(a)
    b=np.array(b)

    return np.mean(
        np.linalg.norm(a-b,axis=1)
    )


def calculate_pose_score(similarity,distance):

    score = (
        0.7*((similarity+1)/2)
        +
        0.3*(1/(1+distance))
    )

    return score*100

