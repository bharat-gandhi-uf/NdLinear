# NdLinear
This Repo shocases usage of NdLinear on a CLIP type Vision Language Deep Learning Model, with the use of "vit-base-patch16-224" Vision transformer from Google and "distilbert-base-uncased" for Language understanding.
# Process
# CLIP (Contrastive Language-Image Pre-training) Implementation

This repository contains a PyTorch implementation of a CLIP-like model for image-text matching. CLIP models are trained to learn a joint embedding space for images and text, enabling zero-shot image classification and cross-modal retrieval.

## Overview

The CLIP (Contrastive Language-Image Pre-training) model architecture was introduced by OpenAI and allows for learning transferable visual representations from natural language supervision. This implementation creates a simplified version of CLIP that can be trained on image-caption pairs.

## Model Architecture

The implementation consists of two main components:

1. **Image Encoder**: Uses a Vision Transformer (ViT) from Hugging Face's Transformers library to encode images into embedding vectors.

2. **Text Encoder**: Uses DistilBERT from Hugging Face's Transformers library to encode text into embedding vectors.

Both encoders project their respective outputs to a shared embedding space of the same dimension (configurable, default is 256). The model is trained using a contrastive loss function that maximizes the similarity between matching image-text pairs while minimizing the similarity for non-matching pairs.

## Dataset

The model is designed to be trained on the Flickr8k dataset, which contains 8,000 images, each with 5 different captions. The dataset is loaded using a custom `ImageTextDataset` class that handles:

- Image loading and preprocessing
- Caption tokenization
- Returning batches of image-text pairs

## Training Process

The training process includes:

1. **Data Preparation**: 
   - Loading the Flickr8k dataset
   - Splitting into training and validation sets
   - Creating data loaders with batching and shuffling

2. **Model Training**:
   - Initializing the CLIP model with ViT and DistilBERT components
   - Training for a specified number of epochs
   - Computing contrastive loss between image and text embeddings
   - Optimizing model parameters with Adam optimizer

3. **Evaluation**:
   - Evaluating the model on the validation set
   - Computing image-to-text and text-to-image retrieval accuracy (top-1 and top-5)
   - Tracking and visualizing training metrics

4. **Checkpointing**:
   - Saving model checkpoints at regular intervals
   - Storing final trained model

## Zero-Shot Capabilities

The trained model can be used for zero-shot tasks including:

1. **Zero-Shot Classification**: 
   - Classifying images into categories without specific training for those categories
   - Using text prompts like "a photo of a dog" to create classifiers

2. **Cross-Modal Retrieval**:
   - Image-to-text: Finding the most relevant caption for an image
   - Text-to-image: Finding the most relevant image for a caption

## Usage

The notebook contains the following core functionality:

1. **Training**: 
   ```python
   main()
   ```
   This function handles the complete training pipeline, from data loading to model evaluation.

2. **Testing**:
   ```python
   test_random_image(checkpoint_path, image_dir)
   ```
   This function tests the trained model on new images for zero-shot classification and caption matching.

## Components

The implementation includes the following classes and functions:

- `ImageTextDataset`: Handles loading and preprocessing of the Flickr8k dataset
- `TextEncoder`: BERT-based encoder for textual content
- `ImageEncoder`: ViT-based encoder for visual content
- `CLIPModel`: Combines both encoders and implements the contrastive learning objective
- `train_epoch`: Handles one epoch of training
- `evaluate`: Evaluates model performance on validation data
- `zero_shot_prediction`: Performs zero-shot prediction using the trained model
- `train_and_evaluate`: Orchestrates the training and evaluation process
- `visualize_results`: Creates visualizations of training metrics
- `test_random_image`: Tests the model on new images

## Requirements

The implementation requires the following Python libraries:
- PyTorch
- torchvision
- transformers (Hugging Face)
- PIL
- numpy
- matplotlib
- scikit-learn
- tqdm

## Results

The model is evaluated based on:
- Training loss
- Image-to-text retrieval accuracy (top-1 and top-5)
- Text-to-image retrieval accuracy (top-1 and top-5)

These metrics are saved as plots in the results directory.

## References

This implementation is inspired by the original CLIP paper:
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) by Alec Radford, Jong Wook Kim, et al.
