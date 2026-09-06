# WSNet official YOLO detection dataset

Source: https://huggingface.co/datasets/subbareddyoota/wseg_dataset
Revision: bdcc1e1cef607bfd7a4f98fc28f31a638a0e0ca9
License: CC BY-NC 4.0

This source-specific dataset is generated from official WSNet image-mask pairs. Masks are converted to single-class wound bounding boxes using external contours with area >= 50 pixels. Exact image-hash groups are assigned deterministically with seed 42; no clinical, Kaggle, or Roboflow files are included.

Counts: train=1894, val=412, test=380; boxes: train=2681, val=609, test=533.

This dataset is for non-commercial research validation with attribution and remains subject to the source licence.
