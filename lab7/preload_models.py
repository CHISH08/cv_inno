import os
from transformers import pipeline

depth_model = os.getenv("DEPTH_MODEL_ID", "depth-anything/Depth-Anything-V2-Small-hf")

print(f"Preloading depth model: {depth_model}")
pipeline("depth-estimation", model=depth_model, device=-1)

print("Model preload finished.")