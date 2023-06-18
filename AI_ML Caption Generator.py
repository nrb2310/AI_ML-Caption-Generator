#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Install required packages
get_ipython().system('pip install transformers')
get_ipython().system('pip install torch')
get_ipython().system('pip install pillow')
get_ipython().system('pip install torchvision')


# In[12]:


# Import necessary libraries

# Transformers library for vision encoder-decoder model
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

# PyTorch for deep learning framework
import torch

# PIL (Python Imaging Library) for image processing
from PIL import Image

# Flask for creating web application
from flask import Flask, render_template, request

# OS module for file operations
import os

# TorchVision for image transformations
from torchvision.transforms import ToTensor


# In[15]:


# Initialize Flask app
app = Flask(__name__, template_folder='C:/Users/Computer/Desktop/Ass1', static_folder='C:/Users/Computer/Desktop/Ass1')

# Load pre-trained vision encoder-decoder model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load pre-trained ViT (Vision Transformer) feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load pre-trained tokenizer for caption generation
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set parameters for caption generation
max_length = 16  # Maximum length of the beam
total_beams = 5  # Maximum number of beams to consider while generating captions
generate_parameters = {"max_length": max_length, "num_beams": total_beams}

# Function to generate captions for an image
def generate_captions(image_path, num_captions):
    try:
        # Open and preprocess the image
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        images = [image]
        # Extract image features using the ViT feature extractor
        image_features = feature_extractor(images=images, return_tensors="pt").pixel_values
        image_features = image_features.to(device)
        # Generate captions using the vision encoder-decoder model
        output_captions = model.generate(image_features, **generate_parameters, num_return_sequences=num_captions)
        # Decode and post-process the generated captions
        generated_captions = tokenizer.batch_decode(output_captions, skip_special_tokens=True)
        generated_captions = [caption.strip() for caption in generated_captions]
        return generated_captions
    except Exception as e:
        print(f"An error occurred while generating captions: {str(e)}")
        return []

# Define a Flask route for uploading an image and generating captions
@app.route("/", methods=["GET", "POST"])
def upload_and_generate_caption():
    if request.method == 'POST':
        try:
            # Retrieve the uploaded image file and the number of captions to generate
            image_file = request.files['file']
            num_captions = int(request.form['num_captions'])
            if image_file:
                # Define the folder to save the uploaded image
                app.config['UPLOAD_FOLDER'] = 'C:/Users/Computer/Desktop/Ass1'
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
                # Save the uploaded image to the specified folder
                image_file.save(image_path)
                # Generate captions for the uploaded image
                captions = generate_captions(image_path, num_captions)
                # Render the result template with the uploaded image filename and generated captions
                return render_template("result.html", filename=image_file.filename, captions=captions)
        except Exception as e:
            print(f"An error occurred while processing the image: {str(e)}")
    # Render the index template for uploading the image
    return render_template("index.html")

# Run the Flask app
if __name__ == "__main__":
    app.run()


# In[ ]:




