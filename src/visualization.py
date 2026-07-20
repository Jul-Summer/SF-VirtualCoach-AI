
import cv2


COCO_CONNECTIONS = [

    (5,7),
    (7,9),

    (6,8),
    (8,10),

    (5,6),

    (5,11),
    (6,12),

    (11,12),

    (11,13),
    (13,15),

    (12,14),
    (14,16)

]



def draw_skeleton(
        frame,
        keypoints,
        score=None):


    output = frame.copy()



    for x,y,c in keypoints:

        if c > 0.3:

            cv2.circle(
                output,
                (int(x),int(y)),
                5,
                (0,255,0),
                -1
            )



    for p1,p2 in COCO_CONNECTIONS:


        x1,y1,c1 = keypoints[p1]

        x2,y2,c2 = keypoints[p2]


        if c1 > 0.3 and c2 > 0.3:

            cv2.line(
                output,
                (int(x1),int(y1)),
                (int(x2),int(y2)),
                (255,0,0),
                3
            )



    if score is not None:

        cv2.putText(
            output,
            f"Score: {score:.1f}%",
            (30,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0,0,255),
            3
        )


    return output
