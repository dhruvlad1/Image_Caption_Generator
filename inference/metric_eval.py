import argparse
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from models.combined_model import CombinedModel
from utils.dataset import CocoDataset, collate_fn
from utils.evaluation import evaluate_bleu
from utils.vocab import Vocabulary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a captioning model with BLEU scores.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Root directory for the COCO dataset.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained checkpoint path.")
    parser.add_argument("--vocab-size", type=int, default=256, help="Vocabulary size for the model.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    captions_path = dataset_dir / "annotations" / "captions_train2017.json"
    if not captions_path.exists():
        raise FileNotFoundError(f"COCO captions file not found: {captions_path}")

    with captions_path.open("r", encoding="utf-8") as f:
        captions_json = __import__("json").load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in captions_json["images"]}
    data_pairs = []
    train_dir = dataset_dir / "train2017"
    for ann in captions_json["annotations"]:
        image_id = ann["image_id"]
        if image_id in id_to_filename:
            data_pairs.append((str(train_dir / id_to_filename[image_id]), ann["caption"]))

    vocab = Vocabulary(freqthreshold=1)
    vocab.buildvocabulary([caption for _, caption in data_pairs])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = CocoDataset(data_pairs, vocab, transform=transform)
    subset_size = min(args.subset_size, len(dataset))
    val_loader = DataLoader(
        Subset(dataset, list(range(subset_size))),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, vocab=vocab),
    )

    model = CombinedModel(256, 512, len(vocab)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    from utils.checkpoint import load_checkpoint
    load_checkpoint(model, optimizer, args.checkpoint)

    bleu1, bleu4 = evaluate_bleu(model, val_loader, vocab, args.device)
    print(f"BLEU-1: {bleu1:.4f}, BLEU-4: {bleu4:.4f}")


if __name__ == "__main__":
    main()