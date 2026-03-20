# 🌀 Naive Bayes from Scratch

An end-to-end project implementing Gaussian and Multinomial Naive Bayes classifiers entirely from scratch using NumPy and Pandas. This project focuses on understanding probabilistic machine learning from first principles.

![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)

## 🧠 Overview

This project implements two core Naive Bayes variations:

- **Gaussian Naive Bayes (GNB)**: Designed for continuous numerical data, such as predicting Abalone age groups.
- **Multinomial Naive Bayes (MNB)**: Designed for discrete text data, specifically for IMDB sentiment analysis.

Both classifiers utilize log-probabilities to ensure numerical stability and prevent floating-point underflow. This project is strictly educational, demonstrating all computations without the help of Scikit-Learn or other ML libraries.

## 📊 Datasets

| Dataset | Task | Type | Source |
|---------|------|------|--------|
| Abalone | Age Group Prediction | Numerical / Continuous | [Kaggle](https://www.kaggle.com/) |
| IMDB Reviews | Sentiment Classification | Text / Discrete | [Stanford AI](https://ai.stanford.edu/) |

## 🧮 Mathematical Foundation

### 1. Parameter Estimation

We calculate the sample mean ($\mu$) and variance ($\sigma^2$) for Gaussian distributions:

$$\mu = \frac{1}{n} \sum_{i=1}^{n} x_i$$

$$\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2$$

### 2. Gaussian Probability Density Function (PDF)

Used to compute the likelihood for continuous features:

$$P(x_i \mid C) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x_i - \mu)^2}{2\sigma^2} \right)$$

### 3. Log-Space Optimization

To maintain numerical precision and avoid multiplying many small probabilities, we use the sum of logarithms:

$$\ln(P(C \mid x)) \propto \ln(P(C)) + \sum_{i=1}^{n} \ln(P(x_i \mid C))$$


## 👥 Team & Responsibilities

### Numerical Stream (Gaussian)

| Member | Responsibility | Deliverables |
|--------|---------------|--------------|
| Ahmed | Data & Utilities | Mean, variance, preprocessing |
| Maya | GNB Training | Class priors, mean/variance matrices |
| Malak | GNB Inference | Gaussian PDF, predict, plots |

### Text Stream (Multinomial)

| Member | Responsibility | Deliverables |
|--------|---------------|--------------|
| Sama | Feature Engineering | Tokenization, vocabulary, Bag of Words, log-probabilities |
| Seif | MNB Logic | Laplace smoothing, likelihoods |

## ⚙️ Installation

```
# Clone the repository
git clone https://github.com/Sam-Gyu/naive-bayes-from-scratch
cd naive-bayes-from-scratch

# Install required dependencies
pip install -r requirements.txt

▶️ Usage
Configure the model choice (Gaussian or Multinomial) in config.py:

python
# config.py
MODEL_TYPE = "gaussian"  # or "multinomial"
DATASET_PATH = "data/"

Run the pipeline:
python main.py
The pipeline will automatically preprocess the data, train the chosen model, and display the prediction results.
