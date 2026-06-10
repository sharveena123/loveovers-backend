"""
Optional: custom fine-tuning (not required for LoveOvers).

The app uses pretrained MobileNetV2 + category mapping by default.
Run this only if you want a dedicated 7-class model instead of ImageNet mapping.
"""

print(
    "LoveOvers uses pretrained MobileNetV2 (no training).\n"
    "POST /classify-food works out of the box after: pip install torch torchvision pillow\n"
    "Optional debug: POST /classify-food?debug=1"
)
import sys
sys.exit(0)
