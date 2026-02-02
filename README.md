# 📊 Learning Probability Density Functions using Roll-Number Parameterized Non-Linear Model

## 📌 Assignment Overview

This project learns a probability density function (PDF) from NO₂ air-quality data using a roll-number-parameterized non-linear transformation.

---

## 🧠 Objective

1. Transform the NO₂ feature using a non-linear function  
2. Learn parameters of a Gaussian-shaped PDF  
3. Estimate distribution parameters from real data

---

## 📂 Dataset

**India Air Quality Dataset**  
Source: Kaggle  
https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data

Feature used:


---

## 🔢 Step-1: Non-Linear Transformation

Each NO₂ value (x) is transformed into (z):

\[
z = x + a_r * sin(b_r * x)
\]

Where:

\[
a_r = 0.05 * (r % 7)
\]

\[
b_r = 0.3 * ((r % 5) + 1)
\]

`r` = University roll number

---

## 📈 Step-2: PDF Learning

We model:

\[
p^​(z)=c * exp(−λ * (z−μ)^2)
\]

Parameters to learn:

- μ (mean)
- λ (precision parameter)
- c (normalization constant)

---

## 🧮 Parameter Estimation

Using statistical estimation:

\[
mu = mean(z)
\]

\[
var = var(z)
\]

\[
lambda_est = 1/(2*var)
\]

\[
c_est = sqrt(lambda_est/pi)
\]

---

## ⚙️ Implementation

### Load Data

```python
df = pd.read_csv("/kaggle/input/india-air-quality-data/data.csv",encoding="latin1")
x = df["no2"].dropna().values
