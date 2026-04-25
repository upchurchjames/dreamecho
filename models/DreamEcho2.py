from transformers import pipeline
from PIL import Image
import os

# Image file name
image_path = "test_image.png"

print("Checking file...")
print("Current working folder:", os.getcwd())
print("Exists:", os.path.exists(image_path))

# Try opening image
# img = Image.open(image_path)
img = Image.open('../data/text_training/test_image.png')
img.verify()
print("Image opened successfully")

# Load classifier
print("Loading classifier...")
classifier = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32"
)
print("Classifier loaded successfully")

# Prompts
scene_prompts = [
    "a dog on a dirt road",
    "a comfortable living room in a dream",
    "a sandy beach with ocean waves in a dream",
    "a deep dark pit in a dream",
    "a high school campus in a dream",
    "a quiet forest path leading to a stream in a dream"
]

# Run classification
print("Running classification...")
result = classifier('https://dogagingproject.org/_next/image?url=https%3A%2F%2Fcontent.dogagingproject.org%2Fwp-content%2Fuploads%2F2020%2F11%2Fhelena-lopes-S3TPJCOIRoo-unsplash-scaled.jpg&w=1200&q=75', candidate_labels=scene_prompts)

# Output
print("\nClassification Results:")
for item in result:
    print(f"{item['label']} -> {item['score']:.4f}")