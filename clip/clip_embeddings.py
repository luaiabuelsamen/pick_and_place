import torch
import clip
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.nn as nn
import torchvision.transforms as T

# Load CLIP model and preprocess pipeline with ViT-B/16 (16x16 patches)
device = "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
model.eval()

# Define a projection head for image patch embeddings
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, output_dim=512):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        x = self.proj(x)
        return x / x.norm(dim=-1, keepdim=True)  # Normalize after projection

# Instantiate projection head
image_head = ProjectionHead(input_dim=768, output_dim=512).to(device)

# Function to extract patch-level features from the image
def get_patch_features(model, image_tensor):
    with torch.no_grad():
        x = model.visual.conv1(image_tensor)  # (1, 3, 224, 224) -> (1, 768, 14, 14)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (1, 196, 768)
        x = torch.cat([
            model.visual.class_embedding.to(x.dtype).unsqueeze(0).unsqueeze(0)
            + torch.zeros(x.shape[0], 1, x.shape[2], dtype=x.dtype),
            x
        ], dim=1)
        x = x + model.visual.positional_embedding.to(x.dtype)
        x = model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # (L, B, D)
        x = model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # (B, L, D)
        return x[:, 1:]  # (B, 196, 768), exclude CLS

# Load text prompts
text_prompts = ["a peg", "top of peg", "bottom of peg"]

# Define image directory and get image list
image_dir = "/home/jetson3/luai/pick_and_place/traj/visual_observations/realsensecameracolorimage_raw"
image_files = [name for name in os.listdir(image_dir) if name.endswith(".png")][:5]

# Prepare plotting grid
fig, axes = plt.subplots(len(text_prompts), len(image_files), figsize=(len(image_files) * 5, len(text_prompts) * 5))

# Loop over images and text prompts
for i, image_name in enumerate(image_files):
    image_path = os.path.join(image_dir, image_name)
    img = preprocess(Image.open(image_path)).unsqueeze(0).to(device)  # Preprocessed for CLIP

    for j, task_text in enumerate(text_prompts):
        text = clip.tokenize([task_text]).to(device)

        # Encode and normalize text features
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # (1, 512)

        # Get and normalize patch features
        patches = get_patch_features(model, img).squeeze(0)  # (196, 768)
        patches = patches / patches.norm(dim=-1, keepdim=True)

        # Project to latent space using interoperability-style head
        patches_projected = image_head(patches)  # (196, 512)

        # Compute cosine similarity to text feature
        similarity_map = (patches_projected @ text_features.T).squeeze()  # (196,)
        similarity_map = similarity_map.reshape(14, 14).detach().cpu().numpy()

        # Rescale similarity map for visualization
        img_full = Image.open(image_path)
        similarity_map_resized = np.interp(similarity_map, (similarity_map.min(), similarity_map.max()), (0, 255))
        similarity_map_resized = np.array(Image.fromarray(similarity_map_resized.astype(np.uint8)).resize(img_full.size))

        # Plot
        ax = axes[j, i]
        ax.imshow(img_full)
        ax.imshow(similarity_map_resized, cmap='jet', alpha=0.5)
        ax.set_title(f"Heatmap: '{task_text}'")
        ax.axis("off")

plt.tight_layout()
plt.show()
