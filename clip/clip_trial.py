#%%
import clip
import torch
from PIL import Image

#%%
model, preprocess = clip.load("ViT-B/32", device="cpu")

image = preprocess(Image.open("robot.png")).unsqueeze(0)
texts = clip.tokenize(["a robot", "a cat", "a car"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(texts)

# Compare which text matches the image
logits_per_image, logits_per_text = model(image, texts)
probs = logits_per_image.softmax(dim=-1).detach().numpy()
print("Label probs:", probs)

image2 = preprocess(Image.open("cat.png")).unsqueeze(0)
# images = [preprocess(Image.open(f"image_{i}.jpg")).unsqueeze(0) for i in range(5)]
images = [image, image2]
images = torch.cat(images, dim=0)

text = clip.tokenize(["a photo of a cat"]).to("cpu")

with torch.no_grad():
    image_features = model.encode_image(images)
    text_features = model.encode_text(text)

# Calculate similarity
similarities = (image_features @ text_features.T).squeeze()
best_match_idx = similarities.argmax().item()
print(f"Best matching image is image_{best_match_idx}.jpg")