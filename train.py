import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data import DatasetMaker
from model import TextOrientationCNN


EPOCHS = 10
BATCH_SIZE = 32



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    samples = 0

    with torch.set_grad_enabled(training):
        progress = tqdm(
            loader,
            desc="training" if training else "validation",
            leave=False,
        )
        for images, labels in progress:
            images = images.to(device)
            labels = labels.float().to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += ((torch.sigmoid(logits) >= 0.5) == labels.bool()).sum().item()
            samples += batch_size
            progress.set_postfix(
                loss=f"{total_loss / samples:.4f}",
                accuracy=f"{correct / samples:.3f}",
            )

    return total_loss / samples, correct / samples


def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = DatasetMaker(
        "train",
        samples_per_source=100
    )
    
    validation_size = int(len(cache) * 0.2)
    train_size = len(cache) - validation_size
    
    train_data, validation_data = random_split(
        cache,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE)

    model = TextOrientationCNN().to(device)
    criterion = BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.2)

    output_path = Path("orientation_cnn.pt")
        
    print(f"device={device} train={train_size} validation={validation_size}")
    for epoch in tqdm(range(1, EPOCHS + 1), desc="epochs"):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, criterion, device
        )
        print(
            f"epoch {epoch}/{EPOCHS} "
            f"train_loss={train_loss:.4f} train_accuracy={train_accuracy:.3f} "
            f"val_loss={validation_loss:.4f} val_accuracy={validation_accuracy:.3f}"
        )
        torch.save(model.state_dict(), output_path)
        print(f"saved={output_path}")


if __name__ == "__main__":
    train()
