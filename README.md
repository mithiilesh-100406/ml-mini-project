# Sentiment Classification Using Transformer Models

## 📌 Overview
This project focuses on building a sentiment classification system using Transformer-based models to classify input text as **Positive** or **Negative**. The system leverages pre-trained language models (such as BERT) and fine-tunes them for binary sentiment analysis. The goal is to demonstrate the effectiveness of Transformer architectures in understanding and classifying textual sentiment with high accuracy and interpretability.

## 🚀 Features
- Binary sentiment classification (Positive/Negative)
- Fine-tuning of BERT-based Transformer models
- Preprocessing pipeline for cleaning and tokenizing text
- Support for GPU-accelerated training and evaluation
- Streamlit-based interactive web application for predictions

## 🧠 Model Architecture
- **Base Model:** Pre-trained BERT from Hugging Face Transformers
- **Classifier Head:** Fully connected layers with dropout and softmax activation
- **Loss Function:** Cross-entropy loss
- **Optimizer:** AdamW
- **Scheduler:** Learning rate scheduler with warm-up steps

## 🗃️ Dataset
- **IMDb Movie Reviews Dataset**
  - 50,000 labeled reviews (25k train, 25k test)
  - Balanced dataset: 50% positive, 50% negative
  - Download: [IMDb Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## ⚙️ Installation

Clone the repository:
```bash
git clone https://github.com/your-username/sentiment-transformer.git
cd sentiment-transformer
