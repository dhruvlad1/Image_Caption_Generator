import argparse
import json
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.combined_model import CombinedModel
from utils.dataset import CocoDataset, collate_fn
from utils.vocab import Vocabulary


def save_checkpoint(model, optimizer, epoch, loss, path):
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]


def train_model(model, dataloader, criterion, optimizer, device, num_epochs=1, checkpoint_path=None):
    model.to(device)
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        loop = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)
        for images, captions in loop:
            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions[:, :-1])
            targets = captions[:, 1:]

            min_len = min(outputs.size(1), targets.size(1))
            outputs = outputs[:, :min_len, :]
            targets = targets[:, :min_len]

            outputs = outputs.reshape(-1, outputs.shape[2])
            targets = targets.reshape(-1)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(len(dataloader), 1)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Avg Loss: {avg_loss:.4f}")

        if checkpoint_path is not None:
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, marker="o", color="blue")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    return train_losses


def build_vocab_and_dataloader(dataset_dir, batch_size=32, subset_size=60000, freq_threshold=5):
    dataset_dir = Path(dataset_dir)
    train_images_dir = dataset_dir / "train2017"
    annotations_dir = dataset_dir / "annotations"
    captions_path = annotations_dir / "captions_train2017.json"

    if not captions_path.exists():
        raise FileNotFoundError(f"COCO captions file not found: {captions_path}")

    with captions_path.open("r", encoding="utf-8") as f:
        captions_json = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in captions_json["images"]}
    data_pairs = []
    for ann in captions_json["annotations"]:
        img_id = ann["image_id"]
        if img_id in id_to_filename:
            image_path = train_images_dir / id_to_filename[img_id]
            data_pairs.append((str(image_path), ann["caption"]))

    if not data_pairs:
        raise ValueError(f"No image-caption pairs found in dataset directory: {dataset_dir}")

    vocab = Vocabulary(freqthreshold=freq_threshold)
    all_captions = [caption for _, caption in data_pairs]
    vocab.buildvocabulary(all_captions)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = CocoDataset(data_pairs, vocab, transform=transform)
    subset_size = min(int(subset_size), len(dataset)) if subset_size is not None else len(dataset)
    subset_dataset = Subset(dataset, list(range(subset_size)))

    dataloader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, vocab=vocab),
    )

    return vocab, dataloader


def main():
    parser = argparse.ArgumentParser(description="Train an image captioning model.")
    parser.add_argument("--dataset-dir", type=str, default="data/coco2017", help="Path to the COCO dataset root containing train2017/ and annotations/")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--subset-size", type=int, default=60000, help="Number of training samples to use. Set to 0 to use all samples.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--freq-threshold", type=int, default=5, help="Minimum word frequency for the vocabulary.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory where checkpoints are stored.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint_img_caption.pth"

    vocab, dataloader = build_vocab_and_dataloader(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        freq_threshold=args.freq_threshold,
    )

    embed_size = 256
    hidden_size = 512
    vocab_size = len(vocab)
    model = CombinedModel(embed_size, hidden_size, vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_model(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        checkpoint_path=str(checkpoint_path),
    )


if __name__ == "__main__":
    main()
