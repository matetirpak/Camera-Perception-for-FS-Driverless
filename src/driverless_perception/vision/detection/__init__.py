"""2D cone detection with YOLO26, fine-tuned on FSOCO.

The paper used YOLOv2 with three classes (blue, yellow, orange) and anchors tuned for tall boxes. Here it is
YOLO26 n, s and m with the five FSOCO classes, whose ids are the ``ConeColor`` values. ``fsoco`` prepares the
dataset, ``train`` runs the fine-tuning.
"""
