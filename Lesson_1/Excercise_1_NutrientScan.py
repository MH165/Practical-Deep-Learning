from fastcore.all import *
import time, json
from fastdownload import download_url
from fastai.vision.all import *
import torch

# ─────────────────────────────────────────────
#  GPU SETUP
# ─────────────────────────────────────────────
if torch.cuda.is_available():
    print(f"✅ GPU found: {torch.cuda.get_device_name(0)}")
    print(f"✅ Fastai is using device: {default_device()}")
else:
    print("⚠️ WARNING: No GPU found by PyTorch. Training will fall back to CPU!")



path = Path(__file__).parent / 'Images'


# ─────────────────────────────────────────────
# Data Block 
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)
plt.show()
# ─────────────────────────────────────────────
# Learner
model_file = path / 'Nutrient_model.pkl'

if model_file.exists():
    print("✅ Loading existing model, skipping training...")
    learn = load_learner(model_file)
else:
    print("⚙️ Training new model...")
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(3)
    learn.export(model_file)

# ─────────────────────────────────────────────
# Inference / Testing
test_image = get_image_files(path/'Bad')[0]
predicted_category, category_index, probabilities = learn.predict(test_image)
print(f"\n--- INFERENCE TEST ---")
print(f"Predicted category: {predicted_category}")
print(f"Confidence: {probabilities[category_index]:.4f}")
