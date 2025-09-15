# Text-to-Image-Generation.
Text-to-Image Generation using Custom DCGAN

 Overview

This project focuses on generating images from natural language descriptions using a Custom Deep Convolutional Generative Adversarial Network (DCGAN). The goal is to explore how conditional GANs can be applied for text-to-image synthesis while experimenting with architectural modifications, dataset choice, and optimization strategies for improved stability and quality of generated images.

 Objectives
	•	Implement a custom DCGAN for text-to-image generation.
	•	Experiment with different dataset (Oxford 102 Flowers) instead of generic datasets like CIFAR or COCO.
	•	Enhance training stability with architectural modifications and regularization.
	•	Evaluate results using both quantitative metrics and visual quality.

 Dataset

The project uses the Oxford 102 Flowers dataset, which contains:
	•	8,189 images of flowers across 102 categories.
	•	Each image is paired with natural language descriptions (captions).
	•	Dataset is preprocessed by:
	•	Resizing and normalizing images.
	•	Tokenizing captions and converting them into word embeddings.

 Methodology

1. Text Preprocessing
	•	Tokenization of captions.
	•	Embedding generation (Word2Vec/GloVe).
	•	Alignment of text features with latent noise vectors.

2. Model Architecture

Generator:
	•	Input: concatenation of random noise vector (z=128) + text embedding.
	•	Layers: ConvTranspose2D, BatchNorm, Dropout, GELU activations.
	•	Output: Synthetic image conditioned on the caption.

Discriminator:
	•	Input: real/fake image + text embedding.
	•	Layers: Conv2D, Spectral Normalization, LeakyReLU.
	•	Output: Probability (real/fake) with respect to text.

3. Training Setup
	•	Optimizer: RMSProp instead of Adam (for smoother convergence).
	•	Loss function: Binary Cross Entropy with label smoothing.
	•	Stabilization techniques:
	•	Data augmentation (random flips, crops).
	•	Spectral Normalization.
	•	Gradual learning rate decay.

4. Hyperparameter Tuning
	•	Used Optuna for automated search.
	•	Tuned parameters: learning rate, batch size, embedding dimension, latent vector size.

📊 Results
	•	Generated images show clear alignment with textual descriptions.
	•	Visual examples:
	•	Input: “A purple flower with five petals” → Generated image resembles correct petal shape and color.
	•	Input: “A yellow sunflower with a dark center” → Generated image reflects distinct sunflower structure.
	•	Evaluation Metrics:
	•	FID (Fréchet Inception Distance) used to measure quality and diversity.
	•	Lower FID observed after applying spectral normalization and label smoothing.

⚙️ Technologies Used
	•	Python 3.x
	•	PyTorch / Torchvision
	•	Numpy, Pandas
	•	Matplotlib, Seaborn
	•	Optuna (hyperparameter tuning)
	•	NLTK / SpaCy (text preprocessing)
