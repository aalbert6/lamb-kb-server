from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
import os


class CLIPEmbeddingFunction:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def __call__(self, inputs):
        """Genera els embeddings per a un conjunt de textos, taules i imatges.
        
           Args:
                inputs (dict): Un diccionari que pot contenir claus 'text', 'tables' i 'images'.
            
           Returns:
                embeddings (dict): Un diccionari amb els embeddings per a cada tipus d'input.
        """
        embeddings = []
        
        for item in inputs:
            if isinstance(item, str) and os.path.exists(item):
                image = Image.open(item).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                features = self.model.get_image_features(**inputs)
            else:
                inputs = self.processor(text=item, return_tensors="pt", paddindg=True).to(self.device)
                features = self.model.get_text_features(**inputs)

            features = features.detach().cpu().numpy().tolist()
            embeddings.append(features[0])

        return embeddings
