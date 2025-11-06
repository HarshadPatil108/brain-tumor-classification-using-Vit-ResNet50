
# !pip install gradio transformers

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
# We only need the ViT model from transformers now
from transformers import ViTForImageClassification
from PIL import Image




# --- 2. App Configuration ---

MODEL_PATH = 'vit_base_best.pth' 
NUM_CLASSES = 4
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. Image Transforms ---

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 4. Load Model ---
def load_model(model_path):
    print(f"Loading model from {model_path}...")
    try:
        # Load the base ViT model structure
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True
        ).to(DEVICE)
        
        # Load your fine-tuned weights
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        print("Please make sure 'vit_base_best.pth' is uploaded to your Colab session.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load the model once when the script starts
model = load_model(MODEL_PATH)

# --- 5. Prediction Function ---
def predict(input_image: Image.Image):
    if model is None:
        return {"Error": "Model not loaded. Please check Colab logs."}

    # Convert to RGB if it's not (e.g., grayscale or RGBA)
    if input_image.mode != "RGB":
        input_image = input_image.convert("RGB")

    # Apply transforms
    image_tensor = inference_transform(input_image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(DEVICE)

    # Perform inference
    with torch.no_grad():
        # Get the logits from the model's output object
        outputs = model(image_tensor).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Create a dictionary of {label: confidence}
    confidences = {CLASS_LABELS[i]: float(probabilities[i]) for i in range(NUM_CLASSES)}
    
    return confidences

# --- 6. Create Gradio Interface ---
print("Starting Gradio interface...")

title = "Brain Tumor Detection (Vision Transformer - ViT)"
description = "Upload an MRI scan to classify the tumor type. This model is a fine-tuned Vision Transformer (ViT)."

if model is not None:
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload MRI Image"),
        outputs=gr.Label(num_top_classes=NUM_CLASSES, label="Prediction Results"),
        title=title,
        description=description,
        examples=[

        ]
    )
    

    demo.launch(share=True, debug=True)
else:
    print("Gradio interface could not be started because the model failed to load.")