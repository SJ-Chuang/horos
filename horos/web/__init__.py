"""horos Web API (Flask).

R2: routes are thin — parameter validation and dispatch into horos.api only.
R1: nothing here may import torch/rfdetr/transformers, and per the layer rule
this package must not import horos.core directly (only horos.api / horos.errors).
"""
