import cv2 as cv
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm




class DatasetMaker(Dataset):

    def __init__(
        self, type, samples_per_source=None
    ):
        self.current_dir = Path.cwd()
        self.type = type
        self.samples_per_source = samples_per_source
        self.cache_dir = self.current_dir / "cache" / "data" / type
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.preprocess_data_direct()

    # прочитать пути файлов для датасета на английском
    def read_list(self):
        list_path = self.current_dir / "dataset" / f"{self.type}.txt"
        with open(list_path, "r") as file:
            return ["/".join(line.strip().split("/")[-2:]) for line in file]
        
    # если кроп меньше килобайта, то он отбрасывается        
    def is_valid_crop(self, crop):
        success, encoded = cv.imencode(".jpg", crop)
        return success and encoded is not None and encoded.nbytes >= 1024

    # препроцессинг
    def preprocess(self, img):
        resized = cv.resize(img, (256, 128), interpolation=cv.INTER_CUBIC)
        gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)

        binary = cv.threshold(
            blurred,
            0,
            255,
            cv.THRESH_BINARY + cv.THRESH_OTSU
        )[1]

        # получает пиксели границы, чтобы узнать цвет фона
        border = np.concatenate([
            binary[0, :],
            binary[-1, :],
            binary[:, 0],
            binary[:, -1]
        ])

        # если фон белый, то инфертируем, чтобы на всех образцах
        # был черный фон и белый текст
        background_value = 255 if np.mean(border) > 127 else 0
        if background_value == 255:
            binary = cv.bitwise_not(binary)
            
        return binary

    # приводить np массив к тензоры и нормализирует его 
    def to_tensor(self, image):
        tensor = torch.from_numpy(image.copy()).float().div(255).unsqueeze(0)
        # 0.5, т.к. изображение черно-белое
        return (tensor - 0.5) / 0.5
    
    def write_cached_image(self, image, index):
        for label, transformed in (
            (0, image),
            (1, cv.rotate(image, cv.ROTATE_180)),
        ):
            processed = self.preprocess(transformed)
            path = self.cache_dir / f"{index}_{label}.jpg"
            cv.imwrite(str(path), processed)

    def append_direct_image(self, image):
        for label, transformed in (
            (0, image),
            (1, cv.rotate(image, cv.ROTATE_180)),
        ):
            processed = self.preprocess(transformed)
            tensor = self.to_tensor(processed)
            self.dataset.append((tensor, torch.tensor(label, dtype=torch.long)))

    def preprocess_data_direct(self):
        self.dataset = []
        samples = self.read_list()
        if self.samples_per_source is not None:
            samples = samples[:self.samples_per_source]

        for sample in tqdm(samples, desc="preprocessing dataset"):
            image_path = self.current_dir / "dataset" / sample
            label_path = image_path.with_suffix(".txt")
            with open(label_path) as file:
                bboxes = [map(float, line.split()[1:]) for line in file]

            image = cv.imread(str(image_path))
            image_height, image_width = image.shape[:2]
            # вырезаем кропы из изображений, приводя ббоксы от YOLO формата
            # к левом верхнему углу и ширине/высоте
            for x_center, y_center, width, height in bboxes:
                x_min = max(0, int(x_center * image_width - width * image_width / 2))
                y_min = max(0, int(y_center * image_height - height * image_height / 2))
                crop_width = int(width * image_width)
                crop_height = int(height * image_height)
                crop = image[y_min:y_min + crop_height, x_min:x_min + crop_width]
                
                # удоляет кроп, если он слишком длинный или маленький (защита от единичных букв и т.д.)
                if (
                    crop.size == 0
                    or crop.shape[0] > crop.shape[1] * 2
                    or not self.is_valid_crop(crop)
                ):
                    continue
                self.append_direct_image(crop)

        russian_images = sorted(
            (self.current_dir / "rus_dataset" / "images").glob("*.jpg")
        )
        if self.samples_per_source is not None:
            russian_images = russian_images[:self.samples_per_source]

        for image_path in tqdm(russian_images, desc="preprocessing rus_dataset"):
            image = cv.imread(str(image_path))
            quarter_width = max(1, image.shape[1] // 4)
            for quarter in range(4):
                start = quarter * quarter_width
                end = image.shape[1] if quarter == 3 else (quarter + 1) * quarter_width
                crop = image[:, start:end]
                # удоляет кроп, если он слишком маленький
                if not self.is_valid_crop(crop):
                    continue
                self.append_direct_image(crop)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
