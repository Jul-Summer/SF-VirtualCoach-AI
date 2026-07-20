
import cv2
import os

from tqdm import tqdm



def extract_frames(
        video_path,
        save_dir=None,
        max_frames=None):


    cap = cv2.VideoCapture(
        video_path
    )


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    frames = []

    frame_id = 0


    while True:

        ret, frame = cap.read()


        if not ret:
            break


        frames.append(frame)


        if save_dir:

            os.makedirs(
                save_dir,
                exist_ok=True
            )


            cv2.imwrite(
                os.path.join(
                    save_dir,
                    f"frame_{frame_id}.jpg"
                ),
                frame
            )


        frame_id += 1


        if max_frames:

            if frame_id >= max_frames:
                break


    cap.release()


    return frames, fps




def create_output_video(
        frames,
        output_path,
        fps):


    height,width,_ = frames[0].shape


    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width,height)
    )


    for frame in frames:

        writer.write(frame)


    writer.release()




def extract_pose_video(
        frames,
        predict_pose):


    poses=[]


    for frame in tqdm(frames):

        pose = predict_pose(frame)

        poses.append(pose)


    return poses
