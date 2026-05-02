from ddgs import DDGS #DuckDuckGo has changed the api so we need to update 
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


def search_images(keywords, max_images=200): 
    return L(DDGS().images(keywords, max_results=max_images)).itemgot('image')


# glass can Image From ddg search
# download_url(search_images('bird photos', max_images=1)[0], 'bird.jpg', show_progress=False)
# Image.open('bird.jpg').to_thumb(256,256).show()
# # Forest Image From ddg search
# download_url(search_images('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
# Image.open('forest.jpg').to_thumb(256,256).show()

search = 'glass can','plastic can', 'aluminium can', 'paper can'
path = Path(__file__).parent / 'Can_Images'

if not path.exists():
    for s in search:
        dest = path/s
        dest.mkdir(parents=True, exist_ok=True)
        download_images(dest, urls=search_images(f'{s} photo'))
        time.sleep(5)
        resize_images(path/s, max_size=400, dest=path/s)

    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    print(f" Deleted {len(failed)}")
else:
    print("Images already downloaded, skipping download step.")


can = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=RandomResizedCrop(224, min_scale=0.5), # Presizing: resize larger first on CPU
    # batch_tfms=aug_transforms(mult=2) # Augment and do final crop on GPU
).dataloaders(path)

# Open a few sample images to verify
can.show_batch(max_n=8, nrows=2)
plt.show()

# ─────────────────────────────────────────────
# Save the model
model_file = path / 'canClassifier_model.pkl'
if model_file.exists():
    print("✅ Loading existing model, skipping training...")
    learner = load_learner(model_file)
    learner.dls = can
else:
    print("⚙️ Training new model...")
    learner = vision_learner(can, resnet18, metrics=error_rate)
    learner.fine_tune(9)
    learner.export(model_file)

# ─────────────────────────────────────────────
# Inference / Testing
test_image = get_image_files(path/'glass')[0]
predicted_category, category_index, probabilities = learner.predict(test_image)
print(f"\n--- INFERENCE TEST ---")
print(f"Predicted category: {predicted_category}")
print(f"Confidence: {probabilities[category_index]:.4f}")
#


# ─────────────────────────────────────────────
# plot confusion matrix
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix()
plt.show()


# plot top losses
interp.plot_top_losses(9, nrows=3)
plt.show()
