
"""
Virtual Coach AI

Entry point of the project.

Pipeline

1. Load videos
2. Pose estimation
3. Pose normalization
4. DTW synchronization
5. Similarity calculation
6. Feedback generation
"""

from src.pose_model import predict_pose
from src.video_utils import extract_frames


def main():

    print("Virtual Coach AI started")

    print("Loading videos...")

    print("Pose estimation...")

    print("DTW synchronization...")

    print("Similarity calculation...")

    print("Feedback generation...")

    print("Pipeline completed successfully.")


if __name__ == "__main__":

    main()
