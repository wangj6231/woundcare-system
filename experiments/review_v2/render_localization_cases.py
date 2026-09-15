"""Render square, undistorted case panels from saved DEVELOPMENT predictions."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .audit_project import read_json, sha
from .localization_benchmark import BUNDLE, output_path, safe_path


def render(out):
    target = out / "worst_crop_cases_square.png"
    if target.exists():
        raise FileExistsError("keep existing scientific figure")
    cohort = {r["image_id"]: r for r in read_json(out / "cohort.json")}
    cases = read_json(out / "error_cases.json")
    sheet = Image.new("RGB", (1024, 4 * 382 + 35), "white")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    ImageDraw.Draw(sheet).text((12, 8), "Development only | Green: GT | Red: predicted box | Cyan: padded crop | conf=.10, NMS=.70", font=font, fill="black")
    for i, r in enumerate(cases):
        row = cohort[r["image_id"]]
        if row["split"] != "val":
            raise ValueError("not development validation")
        path = safe_path(BUNDLE, row["image"])
        if sha(path) != row["image_sha256"]:
            raise ValueError("image identity changed")
        with Image.open(path) as source:
            im = source.convert("RGB")
        if im.size != (512, 512):
            raise ValueError("unexpected aspect ratio")
        draw = ImageDraw.Draw(im)
        for box in r["gt_boxes"]:
            draw.rectangle(box, outline="#00ff00", width=3)
        for box in r["pred_boxes"]:
            draw.rectangle(box, outline="#ff2222", width=3)
        if r["crop"]:
            draw.rectangle(r["crop"], outline="#00ddff", width=3)
        x, y = (i % 3) * 341, (i // 3) * 382 + 35
        sheet.paste(im.resize((330, 330), Image.Resampling.LANCZOS), (x, y))
        ImageDraw.Draw(sheet).text((x + 3, y + 334),
            f"{r['image_id']}  TP/FP/FN {r['tp']}/{r['fp']}/{r['fn']}\n"
            f"crop {r['crop_coverage']:.1%}  mask IoU {r['mask_iou']:.1%}", font=font, fill="black")
    sheet.save(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    render(output_path(parser.parse_args().output))
