# AI_ML-Caption-Generator

#### The Code allows users to upload an image and generates captions for the uploaded image using the ViT-GPT2 vision encoder-decoder model. It provides an easy-to-use interface for caption generation
#

#### --> The different parts of the code are :

1. Installation: To run the web application, you need to install the required packages. You can use the following command to install them:

```python
!pip install transformers
!pip install torch
!pip install pillow
!pip install torchvision
```

2. Importing Libraries: Next, we import the required libraries for our application :

```python
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
```

3. Flask App: The Flask application is created using Flask(__name__), where __name__ represents the name of the current module.

```python
app = Flask(__name__)
```

4. Loading Pre-trained Models and Tokenizer: We load the pre-trained models and tokenizer needed for image caption generation. Here's what each step does:

    i) We load a pre-trained vision encoder-decoder model using ```VisionEncoderDecoderModel```. The model we use is called ```vit-gpt2-image-captioning```.

    ii) The ```ViTFeatureExtractor``` is loaded using ```ViTFeatureExtractor```. This class extracts features from images using the ```Vision Transformer (ViT)``` architecture.

    iii) The ```AutoTokenizer``` is used to load the pre-trained tokenizer for caption generation.

5. Setting Device and Generation Parameters: We set the device for execution to use the ```GPU``` if available, otherwise the ```CPU```. This ensures efficient processing of the image and caption generation. We also define some parameters for caption generation, such as the maximum length of the caption and the number of beams to consider while generating captions. Here’s what parameters are there:

    i) ```max_length```: This object sets the maximum length of the generated captions. Captions longer than this length will be truncated.

    ii) ```total_beams```: This object sets the number of beams (alternative captions) to consider during caption generation. More beams can result in a wider exploration of possible captions but may also increase computation time.

    iii) ```generate_parameters = {"max_length": max_length, "num_beams": total_beams}```: This line creates a dictionary containing the parameters for caption generation. The dictionary is passed as an argument when calling the ```model.generate()``` method.

6. Caption Generation Function: Next, we define the ```generate_captions``` function. This function takes an image path and the number of captions to generate as input. Here's a breakdown of what happens inside this function:

    i) We open and preprocess the image using the ```PIL``` library. If the image is not in RGB mode, we convert it to RGB.

    ii) We extract image features using the ViT feature extractor by passing the preprocessed image to ```feature_extractor```. The extracted features are then transferred to the specified device (GPU or CPU).

    iii) We generate captions for the image using the vision ```encoder-decoder``` model. The generate method is called on the model, passing the image features and generation parameters. The method returns output captions in tensor form.

    iv) We decode and post-process the generated captions using the ```tokenizer```. The batch_decode method is used to convert the tensor of captions to humanreadable text, skipping special tokens. We also remove any leading or trailing white spaces.

    v) Finally, the function returns the generated
captions.

7. Uploading and Generating Captions: We define a Flask route ```("/")``` for handling the uploading of images and generating captions. Here's how it works:

    i) If a ```POST``` request is received, we retrieve the uploaded image file and the number of captions to generate from the form data.

    ii) We define a folder to save the uploaded image.

    iii) The image file is saved to the specified folder using ```save()``` method.

    iv) We call the ```generate_captions``` function with the image path and the number of captions as arguments.

    v) The result template is rendered with the uploaded image filename and the generated captions.

    vi) If any errors occur during the image processing or caption generation, they are caught and logged.

8. HTML Templates: We have two HTML templates, namely ```index.html``` and ```result.html```.

    i) ```Index.html``` template is the landing page of the application. It includes a form where users can upload an image and specify the number of captions to generate.

    ii) ```Result.html``` template is displayed after the image is uploaded and captions are generated. It shows the uploaded image and the generated captions.

10. Main Execution: The code runs the Flask application on a specific host and port if the module is executed directly.

```python
app.run(host="0.0.0.0", port=8000)
```












