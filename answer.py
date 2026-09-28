from __future__ import annotations

import csv
from pathlib import Path

import cv2 as cv
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import DatasetMaker
from model import TextOrientationCNN


PROJECT_ROOT = Path(__file__).resolve().parent


class TestImageDataset(Dataset):
    def __init__(self, image_paths: list[Path]) -> None:
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.image_paths[index]
        image = cv.imread(str(path))
        processed = DatasetMaker.preprocess(image)
        return DatasetMaker.to_tensor(processed), path.stem


def read_submission_ids() -> list[str]:
    sample_sub_path = PROJECT_ROOT / "test" / "sample_submission.csv"
    with sample_sub_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [row["image_id"] for row in rows]

# составляем список с путями к файлам
def resolve_image_paths(image_ids: list[str]) -> list[Path]:
    paths = []
    image_dir = PROJECT_ROOT / "test" / "test" / "images"
    for image_id in tqdm(image_ids):
        paths.append(image_dir / (image_id+".png"))
    return paths

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_ids = read_submission_ids()
    image_paths = resolve_image_paths(image_ids)

    model = TextOrientationCNN().to(device)
    state_dict = torch.load((PROJECT_ROOT / "orientation_cnn.pt"), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(1)
    loader = DataLoader(
        TestImageDataset(image_paths),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    
    probabilities = []
    with torch.inference_mode():
        for images, _ in tqdm(loader, desc="predicting test", unit="batch"):
            logits = model(images.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())

    with (PROJECT_ROOT / "submission.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "p_180"])
        writer.writerows(zip(image_ids, probabilities))


if __name__ == "__main__":
    main()
