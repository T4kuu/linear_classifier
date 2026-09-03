# Linear Classifier From Scratch

A binary logistic regression classifier from scratch using only Python and NumPy.

The main goal of this project is not to use a machine learning framework, but to understand how a binary classifier works mathematically and implement every core component manually.

---

## Overview

This project implements logistic regression without:

- scikit-learn
- pandas
- scipy
- PyTorch
- TensorFlow
- ready-made optimization algorithms

The only external numerical library used is **NumPy**.

The classifier is trained using gradient descent and Binary Cross-Entropy loss.

---

## Mathematical Pipeline

The complete model can be represented as:

```text
Input features X
       │
       ▼
Linear model
       │
       ▼
Sigmoid
       │
       ▼
Predicted probability
       │
       ▼
Binary Cross-Entropy
       │
       ▼
Gradient
       │
       ▼
Gradient Descent
       │
       ▼
Updated w and b
```

### Linear Model
$z = Xw + b$

### Sigmoid
${\sigma}(z) = \frac{1}{1 + e^{-z}}$

$0 < p < 1$

$p = P(y=1|x)$

### Likelihood and ln-likelihood
$p^y(1-p)^{1-y}; y \in \{0; 1\}$

$ln(ab) = ln(a) + ln(b)$ and $ln(a^b)=bln(a)$

$$
L = \prod_{i=1}^{N} p_i^{y_i}(1-p_i)^{1-y_i}
\quad\rightarrow\quad
\ln(L) = \sum_{i=1}^{N}
\left[
y_i\ln(p_i) + (1-y_i)\ln(1-p_i)
\right]
$$

### BCE
$min(-ln(L))$

$BCE = -\frac{1}{2}{\Sigma}_{i=1}^N[y_iln(p_i) + (1-y_i)ln(1-p_i)]$

### Gradient and Gradient Descent
$\frac{\partial L}{\partial w} = \frac{1}{N}X^T(p-y)$

$\frac{\partial L}{\partial b} = \frac{1}{N}{\Sigma}_{i=1}^N=(p_i-y_i)$

$w_{new} = w - {\eta}{\nabla}_wL$

$b_{new} = b - {\eta} \frac{\partial L}{\partial b}$

## Dataset
dataset used - https://www.kaggle.com/datasets/alyabdelnaby/data-banknote-authentication
