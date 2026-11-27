"""Visual search ("more like this"): find visually similar images with a pretrained CNN.

Real images: the CIFAR-10 test set (10 classes such as airplane, cat, ship), downloaded by torchvision.
A good "more like this" result shares the query's class. Compares three embeddings:

1. raw pixels (no learning)
2. ResNet-18 with random weights (same architecture, no training): shows what the architecture alone gives
3. ResNet-18 pretrained on ImageNet (transfer learning, no fine-tuning)

Metrics: precision@10 (share of the 10 nearest neighbours with the query's class) and mean average precision.

Needs internet the first time (about 170 MB of images and 45 MB of weights). Runs in GitHub Actions.
Usage: python scripts/visual_search.py --n-per-class 200
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent


def load(n_per_class: int, data_dir: Path):
    ds = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True)
    labels = np.array(ds.targets)
    idx = np.concatenate([np.where(labels == c)[0][:n_per_class] for c in range(10)])
    images = [ds[i][0] for i in idx]
    return images, labels[idx], ds.classes


def embed_pixels(images) -> np.ndarray:
    x = np.stack([np.asarray(im, dtype=np.float32).ravel() / 255.0 for im in images])
    return x - x.mean(0)


@torch.no_grad()
def embed_resnet(images, pretrained: bool, batch: int = 64) -> np.ndarray:
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    torch.manual_seed(0)
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()   # keep the 512-dimension image embedding, drop the classifier
    model.eval()
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    out = []
    for s in range(0, len(images), batch):
        out.append(model(torch.stack([tf(im) for im in images[s:s + batch]])).numpy())
    return np.concatenate(out)


def evaluate(emb: np.ndarray, labels: np.ndarray, k: int = 10) -> dict:
    z = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)                 # never return the query itself
    order = np.argsort(-sim, axis=1)
    same = labels[order] == labels[:, None]
    p_at_k = same[:, :k].mean()
    n_rel = same.sum(1)
    hits = np.cumsum(same, 1)
    ranks = np.arange(1, same.shape[1] + 1)
    ap = ((hits / ranks) * same).sum(1) / n_rel
    return {f"precision_at_{k}": float(p_at_k), "mean_average_precision": float(ap.mean()),
            "per_class_precision": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=200)
    ap.add_argument("--data-dir", default=str(ROOT / ".data"))
    args = ap.parse_args()
    images, labels, classes = load(args.n_per_class, Path(args.data_dir))
    rows = {}
    for name, fn in (("Raw pixels", embed_pixels),
                     ("ResNet-18, random weights", lambda im: embed_resnet(im, pretrained=False)),
                     ("ResNet-18, pretrained on ImageNet", lambda im: embed_resnet(im, pretrained=True))):
        t = time.time()
        e = fn(images)
        m = evaluate(e, labels)
        z = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        sim = z @ z.T
        np.fill_diagonal(sim, -np.inf)
        nn10 = np.argsort(-sim, axis=1)[:, :10]
        m["per_class_precision"] = {classes[c]: float((labels[nn10[labels == c]] == c).mean()) for c in range(10)}
        m["seconds"] = round(time.time() - t, 1)
        m["dim"] = int(e.shape[1])
        rows[name] = m
        print(name, {k: v for k, v in m.items() if k != "per_class_precision"}, flush=True)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "visual_search.json").write_text(json.dumps(rows, indent=2))
    best = rows["ResNet-18, pretrained on ImageNet"]["per_class_precision"]
    worst = sorted(best, key=best.get)[:2]
    lines = [
        "# Visual search results (real images: CIFAR-10 test set)", "",
        f"{len(images):,} images, {args.n_per_class} per class, 10 classes. Each image is a query; a result is correct if it shares the query's class.",
        "Chance level is about 10%.", "",
        "| Embedding | Dimensions | Precision@10 | Mean average precision |", "|---|---|---|---|",
    ]
    for n, m in rows.items():
        lines.append(f"| {n} | {m['dim']} | {m['precision_at_10']:.3f} | {m['mean_average_precision']:.3f} |")
    lines += ["", "Precision@10 by class (pretrained ResNet-18):", "",
              "| Class | Precision@10 |", "|---|---|"]
    lines += [f"| {c} | {best[c]:.3f} |" for c in sorted(best, key=best.get, reverse=True)]
    lines += ["", f"Hardest classes: {worst[0]} and {worst[1]}, which look alike at 32 x 32 pixels.",
              "The pretrained network was never trained on CIFAR-10; its ImageNet features transfer.", ""]
    (out / "VISUAL_SEARCH.md").write_text("\n".join(lines))
    print("\n".join(lines))   # also in the job log


if __name__ == "__main__":
    main()
