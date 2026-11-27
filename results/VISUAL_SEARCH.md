# Visual search results (real images: CIFAR-10 test set)

From the `visual-search` GitHub Actions run [45930121393](https://github.com/Snehachowdary56/pin-recommendations/actions/runs/45930121393)
(`python scripts/visual_search.py --n-per-class 200`).

2,000 images, 200 per class, 10 classes. Each image is a query; a result is correct if it shares the query's class.
Chance level is about 10%.

| Embedding | Dimensions | Precision@10 | Mean average precision | Embedding time (CPU) |
|---|---|---|---|---|
| Raw pixels | 3,072 | 0.242 | 0.152 | 0.4 s |
| ResNet-18, random weights | 512 | 0.223 | 0.144 | 68.8 s |
| **ResNet-18, pretrained on ImageNet** | 512 | **0.633** | **0.391** | 69.7 s |

- The pretrained network finds a same-class image 2.6 times as often as raw pixels, with a 6 times smaller embedding.
- The same network with random weights is no better than pixels: the gain comes from what it learned on ImageNet,
  not from the architecture. It was never trained on CIFAR-10 (transfer learning, no fine-tuning).
- Next step: fine-tune the last block on the target images, or use a CLIP model so text and image share one space
  (useful for new pins, which have an image and a title but no saves).
