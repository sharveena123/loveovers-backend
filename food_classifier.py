"""
Food image classification — pretrained MobileNetV2 (ImageNet), no training.
"""

from __future__ import annotations

import os
import time
from io import BytesIO
from typing import Dict, List, Optional

import numpy as np

FOOD_CATEGORIES = [
    "Bakery",
    "Pastries",
    "Bread",
    "Desserts",
    "Meals",
    "Beverages",
    "Other",
]

IMG_SIZE = 224
TOP_K = 20
# Downscale phone photos before inference (model only needs 224px; saves decode/resize time)
MAX_IMAGE_SIDE = int(os.getenv("FOOD_IMAGE_MAX_SIDE", "640"))

EXPLICIT_LABEL_MAP: Dict[str, str] = {
    "bagel": "Bread",
    "pretzel": "Bread",
    "French loaf": "Bread",
    "banana": "Other",
    "orange": "Other",
    "lemon": "Other",
    "fig": "Other",
    "pineapple": "Other",
    "croissant": "Pastries",
    "dough": "Pastries",
    "cheeseburger": "Meals",
    "hotdog": "Meals",
    "pizza": "Meals",
    "burrito": "Meals",
    "carbonara": "Meals",
    "consomme": "Meals",
    "guacamole": "Meals",
    "mashed potato": "Meals",
    "french fries": "Meals",
    "ice cream": "Desserts",
    "ice lolly": "Desserts",
    "trifle": "Desserts",
    "chocolate sauce": "Desserts",
    "red wine": "Beverages",
    "espresso": "Beverages",
    "cup": "Beverages",
    "eggnog": "Beverages",
    "milk can": "Beverages",
    "water bottle": "Beverages",
    "beer glass": "Beverages",
    "coffee mug": "Beverages",
    "teapot": "Beverages",
    "waffle iron": "Bakery",
    "frying pan": "Other",
    "potpie": "Meals",
    "meat loaf": "Meals",
    "head cabbage": "Other",
}

_model = None
_imagenet_labels: List[str] = None
_index_to_category: List[str] = None
_transform = None
_device = None
_warmed_up = False


def _get_device():
    global _device
    if _device is None:
        import torch
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def format_food_label(imagenet_label: str) -> str:
    return imagenet_label.strip().lower().replace("_", " ")


def _label_to_category(label: str) -> str:
    if label in EXPLICIT_LABEL_MAP:
        return EXPLICIT_LABEL_MAP[label]

    text = label.lower()

    beverages = (
        "espresso", "coffee", "tea", "wine", "beer", "juice", "water",
        "drink", "beverage", "mug", "teapot", "bottle", "eggnog",
    )
    bread = (
        "bread", "bagel", "pretzel", "loaf", "bun", "roll", "toast",
        "pita", "naan", "baguette", "crumpet",
    )
    pastries = (
        "croissant", "danish", "doughnut", "donut", "pastry", "eclair",
        "profiterole", "strudel", "turnover", "scone", "dough",
    )
    desserts = (
        "cake", "pie", "tart", "cookie", "brownie", "ice cream", "pudding",
        "custard", "mousse", "cupcake", "cheesecake", "trifle", "sundae",
        "candy", "chocolate", "macaroon", "lolly", "fudge",
    )
    bakery = (
        "muffin", "biscuit", "cracker", "waffle", "pancake", "french toast",
        "bakery", "waffle iron",
    )
    meals = (
        "pizza", "burger", "sandwich", "hot dog", "hotdog", "steak", "salad",
        "soup", "pasta", "spaghetti", "lasagna", "rice", "sushi", "taco",
        "burrito", "fried", "grilled", "roast", "chicken", "beef", "pork",
        "fish", "lobster", "crab", "egg", "omelet", "curry", "stew", "nachos",
        "cheeseburger", "meatball", "dumpling", "ramen", "potpie", "meat loaf",
        "french fries", "mashed potato", "guacamole", "consomme", "carbonara",
    )

    for kw in beverages:
        if kw in text:
            return "Beverages"
    for kw in bread:
        if kw in text:
            return "Bread"
    for kw in pastries:
        if kw in text:
            return "Pastries"
    for kw in desserts:
        if kw in text:
            return "Desserts"
    for kw in bakery:
        if kw in text:
            return "Bakery"
    for kw in meals:
        if kw in text:
            return "Meals"
    return "Other"


def _get_transform():
    global _transform
    if _transform is None:
        from torchvision import transforms
        _transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _transform


def _load_model():
    global _model, _imagenet_labels, _index_to_category
    if _model is not None:
        return _model

    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    t0 = time.perf_counter()
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    _imagenet_labels = list(weights.meta["categories"])
    _index_to_category = [_label_to_category(name) for name in _imagenet_labels]

    _model = mobilenet_v2(weights=weights)
    _model.to(_get_device())
    _model.eval()
    _get_transform()
    ms = (time.perf_counter() - t0) * 1000
    print(f"Food classifier ready ({ms:.0f} ms load): MobileNetV2 ImageNet")
    return _model


def warm_up() -> None:
    """Load model at server start so the first seller request is not slow."""
    global _warmed_up
    if _warmed_up:
        return
    t0 = time.perf_counter()
    _load_model()
    import torch
    from PIL import Image

    dummy = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
    tensor = _preprocess_tensor(dummy)
    with torch.inference_mode():
        _model(tensor)
    _warmed_up = True
    ms = (time.perf_counter() - t0) * 1000
    print(f"Food classifier warmed up ({ms:.0f} ms)")


def classifier_status() -> dict:
    return {
        "mode": "pretrained",
        "model": "MobileNetV2-ImageNet1K",
        "trainingRequired": False,
        "warmedUp": _warmed_up,
        "categories": FOOD_CATEGORIES,
        "framework": "pytorch",
        "responseFields": ["foodLabel", "category", "confidence"],
        "maxImageSide": MAX_IMAGE_SIDE,
    }


def compress_image_bytes(image_bytes: bytes, target_max_kb: int = 1200) -> bytes:
    """Re-encode large JPEGs so upload + decode stay fast (AI still uses 224px)."""
    from PIL import Image, ImageOps

    target_bytes = target_max_kb * 1024
    if len(image_bytes) <= target_bytes:
        return image_bytes

    with Image.open(BytesIO(image_bytes)) as img:
        img = ImageOps.exif_transpose(img.convert("RGB"))
        if max(img.size) > MAX_IMAGE_SIDE:
            img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.BILINEAR)

        for quality in (85, 75, 65, 55, 45):
            out = BytesIO()
            img.save(out, format="JPEG", quality=quality, optimize=True)
            data = out.getvalue()
            if len(data) <= target_bytes:
                return data
        return data


def _open_image(image_bytes: bytes):
    """Decode and shrink large camera photos before inference."""
    from PIL import Image, ImageOps

    with Image.open(BytesIO(image_bytes)) as img:
        img.draft("RGB", (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        img = ImageOps.exif_transpose(img.convert("RGB"))
        if max(img.size) > MAX_IMAGE_SIDE:
            img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.BILINEAR)
        return img.copy()


def _preprocess_tensor(image) -> "torch.Tensor":
    return _get_transform()(image).unsqueeze(0).to(_get_device())


def _predict_probs(image_bytes: bytes) -> np.ndarray:
    import torch

    image = _open_image(image_bytes)
    tensor = _preprocess_tensor(image)
    model = _load_model()

    with torch.inference_mode():
        return torch.softmax(model(tensor), dim=1)[0].cpu().numpy()


def _result_from_probs(probs: np.ndarray) -> dict:
    top_indices = np.argsort(probs)[::-1][:TOP_K]
    top_idx = int(top_indices[0])

    # Prefer a food-like label in top-K (skip "plate", "table", etc. -> Other)
    food_label = format_food_label(_imagenet_labels[top_idx])
    for idx in top_indices:
        label = _imagenet_labels[int(idx)]
        mapped = _index_to_category[int(idx)]
        if mapped != "Other":
            food_label = format_food_label(label)
            break

    category_scores = {cat: 0.0 for cat in FOOD_CATEGORIES}
    for idx in top_indices:
        category_scores[_index_to_category[int(idx)]] += float(probs[idx])

    total = sum(category_scores.values())
    category = max(category_scores, key=category_scores.get)
    confidence = category_scores[category] / total if total > 0 else 0.0

    # Only fall back to Other if nothing food-like scored
    if category == "Other" or confidence < 0.08:
        for idx in top_indices:
            mapped = _index_to_category[int(idx)]
            if mapped != "Other":
                category = mapped
                confidence = float(probs[idx])
                break

    return {
        "foodLabel": food_label,
        "category": category,
        "confidence": round(confidence, 4),
    }


def classify_image_bytes(image_bytes: bytes) -> dict:
    probs = _predict_probs(image_bytes)
    return _result_from_probs(probs)


def classify_image_bytes_timed(image_bytes: bytes) -> dict:
    """Same as classify_image_bytes with timing breakdown (for logs / debug)."""
    t0 = time.perf_counter()
    probs = _predict_probs(image_bytes)
    inference_ms = (time.perf_counter() - t0) * 1000
    result = _result_from_probs(probs)
    result["_timing"] = {
        "inferenceMs": round(inference_ms, 1),
        "imageBytes": len(image_bytes),
    }
    return result


def classify_image_bytes_detailed(image_bytes: bytes) -> dict:
    probs = _predict_probs(image_bytes)
    result = _result_from_probs(probs)
    top_indices = np.argsort(probs)[::-1][:TOP_K]

    hits = []
    for idx in top_indices:
        label = _imagenet_labels[int(idx)]
        hits.append({
            "imagenetLabel": label,
            "foodLabel": format_food_label(label),
            "mappedCategory": _index_to_category[int(idx)],
            "probability": round(float(probs[idx]), 4),
        })

    return {**result, "topImagenetPredictions": hits}
