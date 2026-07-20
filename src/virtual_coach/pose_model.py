
import torch
import torchvision
import torchvision.transforms as T


def load_pose_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
        weights="DEFAULT"
    )

    model.to(device)

    model.eval()

    return model, device



transform = T.Compose([
    T.ToTensor()
])



def predict_pose(frame, model, device):

    image = transform(frame).to(device)

    with torch.no_grad():

        output = model([image])


    keypoints = output[0]["keypoints"]
    scores = output[0]["scores"]


    if len(scores) == 0:
        return None


    person = torch.argmax(scores)


    pose = (
        keypoints[person]
        .cpu()
        .numpy()
    )


    return pose
