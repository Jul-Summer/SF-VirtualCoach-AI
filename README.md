
# Virtual Coach AI

## Project Goal

Virtual Coach AI is a Computer Vision project for human exercise analysis.

The system compares a user's exercise video with a reference coach video using human pose estimation and provides a similarity score.


## Dataset

Input data:

- Coach reference video
- User exercise video

The system processes videos frame-by-frame and extracts 17 human body keypoints.


## Pipeline

Input Video

↓

Keypoint R-CNN

↓

Pose Keypoints Extraction

↓

Pose Normalization

↓

DTW Synchronization

↓

Similarity Calculation

↓

Feedback Score


## Model

Used model:

- Keypoint R-CNN
- torchvision detection model
- pretrained on MS COCO dataset


## Validation

Steps:

1. Extract video frames
2. Detect body keypoints
3. Normalize skeleton coordinates
4. Align sequences using DTW
5. Calculate similarity metrics


## Experiments

Implemented methods:

- Pose normalization
- Affine transformation
- Cosine similarity
- Weighted distance
- DTW alignment


## Results

Generated outputs:

- Comparison video
- CSV reports
- Similarity plot


Results folder:

results/

    video/

    csv/

    plots/


## How to Run

Install dependencies:

pip install -r requirements.txt


Run:

python scripts/main.py


For full reproduction:

notebooks/Sprint_4_Virtual_Coach_Final.ipynb


## Repository Structure

Virtual_Coach/

    src/

    notebooks/

    scripts/

    images/

    results/

    data/



## Google Colab Notebooks

### Sprint 1 — Keypoint R-CNN Pose Detection
[Open in Colab]([https://colab.research.google.com/drive/1uO8dPtjoPPQrlK56rTnrXjnB039K6FVa?usp=sharing])    

### Sprint 2 — Pose Matching
[Open in Colab]([https://colab.research.google.com/drive/1uO8dPtjoPPQrlK56rTnrXjnB039K6FVa?usp=sharing])

### Sprint 3 — Video Validation
[Open in Colab]([https://colab.research.google.com/drive/17p9zufjnnKboHQIFD1W0ykvZgGs8lzfa?usp=sharing])

### Sprint 4 — Final Virtual Coach Pipeline
[Open in Colab]([https://colab.research.google.com/drive/1tJ11RvSZnuBXVcdXbk3p2lolMxy6SIBn?usp=sharing])
