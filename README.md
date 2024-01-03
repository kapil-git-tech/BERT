# Sentiment Analysis on IMBD dataset

## Overview

This project aims to perform sentiment analysis on movie reviews using the IMDB dataset. The dataset contains movie reviews labeled with sentiments (positive or negative), making it suitable for training and evaluating sentiment analysis models.

## Contents

- [Dataset](#dataset)
- [Model](#model)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Dataset

### Overview

The IMDB dataset consists of movie reviews with binary sentiment labels (positive/negative). The dataset is widely used in natural language processing (NLP) and machine learning tasks, especially for sentiment analysis.

### Data Structure

- **Format:** CSV or tabular format
- **Columns:**
  - `text`: Movie review text
  - `sentiment`: Binary sentiment label (1 for positive, 0 for negative)

## Model

### Sentiment Analysis Model

- **Architecture:** BERT (Bidirectional Encoder Representations from Transformers)
- **Implementation:** [fine_tune_bert.py](fine_tune_bert.py)
- **Training:** The model is fine-tuned on the IMDB dataset using the Hugging Face Transformers library.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/kapil-git-tech/BERT.git
   cd BERT
