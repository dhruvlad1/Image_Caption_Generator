import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as transforms

from models.combined_model import CombinedModel
from utils.vocab import Vocabulary
from utils.visualization import generate_caption_with_attention, visualize_attention


def parse_args():
    parser = argparse.ArgumentParser(description="Render attention maps for a generated caption.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--vocab", type=str, default=None, help="Optional vocab file.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    vocab = Vocabulary(freqthreshold=1)
    if args.vocab:
        vocab_path = Path(args.vocab)
        if vocab_path.exists():
            with vocab_path.open("r", encoding="utf-8") as f:
                entries = [line.strip().split(":", 1) for line in f if line.strip()]
                vocab.itos = {int(k): v for k, v in entries}
                vocab.stoi = {v: k for k, v in vocab.itos.items()}

    model = CombinedModel(256, 512, len(vocab)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    from utils.checkpoint import load_checkpoint
    load_checkpoint(model, optimizer, str(checkpoint_path))

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image)
    caption, alphas = generate_caption_with_attention(model, image_tensor, vocab, device=args.device)
    print("Predicted:", " ".join(caption))
    visualize_attention(image_tensor, caption, alphas)


if __name__ == "__main__":
    main()