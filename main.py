import torch
from ultralytics import YOLO
import torch_directml

if __name__ == "__main__":

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch_directml.is_available():
        device = "dml"

    print(f"Using device: {device}")

    data_yaml_path = "data.yaml"
    model_path = "yolo11n-seg.pt"

    model = YOLO(model_path)

    results = model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=980,
        augment=True,
        device=device,
        batch=8,
    )

    val_results = model.val(data=data_yaml_path, split="val")
