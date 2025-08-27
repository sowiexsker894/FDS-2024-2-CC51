# YouTube Trends Analytics - International Marketing Insights

## 📊 Project Overview

An international consulting firm based in Lima has commissioned the development of an analytics project to understand YouTube video trends across seven key countries. This project addresses the needs of their client, a leading digital marketing company, seeking strategic insights to enhance their marketing campaigns.

## 🎯 Predictive Analysis: Factors Influencing Video View Count

This project explores an extensive and diverse dataset to predict video view counts using machine learning techniques. Multiple variables are analyzed and results are evaluated using regression models with advanced statistical metrics.

---

## 📋 Table of Contents

- [Introduction](#-introduction)
- [Objectives](#-objectives)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Regression Model](#-regression-model)
- [Results](#-results)
- [Conclusions](#-conclusions)
- [Recommendations](#-recommendations)
- [References](#-references)

---

## 🚀 Introduction

This project analyzes an extensive dataset containing detailed information about videos published on a digital platform. The goal is to identify patterns and relationships between key variables to predict the number of views a video can achieve.

### Key Analysis Components:
- ✅ Data processing and cleaning
- ✅ Predictive modeling using advanced techniques
- ✅ Model evaluation using RMSE, MAE, and R² metrics

---

## 🎯 Objectives

### 🎪 General Objective
Predict video view count based on various video characteristics and engagement metrics.

### 🔍 Specific Objectives
- **Exploratory Analysis:** Identify relationships between variables through comprehensive data exploration
- **Regression Modeling:** Train predictive models to forecast video view performance
- **Performance Evaluation:** Assess model effectiveness using standard statistical metrics

---

## 📈 Dataset

The dataset contains thousands of records with relevant variables for video performance analysis.

| Variable | Type | Description |
|----------|------|-------------|
| `video_id` | String | Unique video identifier |
| `title` | String | Video title |
| `channel_title` | String | Publishing channel name |
| `category_id` | Categorical | Video category (numeric ID) |
| `views` | Numeric | Total view count |
| `likes` | Numeric | Number of likes received |
| `dislikes` | Numeric | Number of dislikes received |
| `comment_count` | Numeric | Total comment count |
| `published_at` | DateTime | Publication date and time |
| `tags` | String | Associated video tags |
| `ratings_disabled` | Boolean | Rating functionality status |
| `comments_disabled` | Boolean | Comment functionality status |
| `description` | String | Video description |

### Dataset Characteristics
- **High Dimensionality:** Multiple feature types (numeric, categorical, text)
- **Heterogeneous Data:** Various data formats requiring advanced preprocessing
- **Scale Variability:** Wide range in target variable distribution

---

## ⚙️ Methodology

### 📊 Analysis Pipeline

#### 1. Exploratory Data Analysis (EDA)
- Pattern identification in dataset
- Outlier detection and missing value analysis
- Variable correlation assessment

#### 2. Data Preprocessing
- Data cleaning and validation
- Categorical and text variable encoding
- Feature scaling and normalization

#### 3. Feature Selection
- **Predictor Variables:** `category_id`, `comment_count`, `likes`, `dislikes`, `ratings_disabled`
- **Target Variable:** `views`

#### 4. Model Training
- Random Forest Regressor implementation
- Hyperparameter optimization

#### 5. Model Evaluation
- Performance metrics: RMSE, MAE, R²
- Cross-validation analysis

---

## 🤖 Regression Model

### Model Architecture
**Random Forest Regressor** was selected to handle non-linear data relationships and capture complex variable interactions.

### Implementation

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Feature selection and target variable
X = df[['category_id', 'comment_count', 'likes', 'dislikes', 'ratings_disabled']]
y = df['views']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation metrics
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

---

## 📊 Results

### Model Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | 5,509,100 | High prediction deviation due to target variable variability |
| **R²** | [Calculated Value] | Model explains ~[percentage]% of data variability |
| **MAE** | [Calculated Value] | Average absolute prediction error |

### Key Findings

#### ✅ Strong Predictors
- `comment_count`, `likes`, and `dislikes` show significant correlation with view count
- `category_id` demonstrates clear influence on video popularity patterns

#### ⚠️ Model Limitations
- **Performance Variance:** Better accuracy for moderate view counts
- **Extreme Value Handling:** Consistent under/over-estimation for viral or low-performing videos
- **Feature Completeness:** Current variables don't fully capture popularity complexity

#### 📈 Category Insights
- Certain video categories show higher propensity for view accumulation
- Audience behavior varies significantly across content types

---

## 🎯 Conclusions

### 📊 Data Insights
- **Variable Importance:** Engagement metrics (`comment_count`, `likes`, `dislikes`) are crucial but insufficient for complete prediction
- **Category Influence:** Video categories significantly impact view potential due to audience differences
- **Rating Functionality:** `ratings_disabled` shows minimal correlation with view count

### 📈 Distribution Analysis
- **Asymmetric Distribution:** Most videos achieve moderate views; few reach viral status
- **Long-tail Effect:** Presence of "long-tail" phenomenon in video popularity distribution

### 🔧 Model Limitations
- **Missing Variables:** Lack of key explanatory features (publication timing, title keywords)
- **Complexity Capture:** Current model cannot fully represent data complexity

---

## 💡 Recommendations

### 🔧 Model Enhancement

#### Feature Engineering
- **Temporal Variables:** Publication hour, day, and seasonal patterns
- **Content Analysis:** Video duration and quality metrics
- **Text Mining:** Keyword extraction from titles, descriptions, and tags

#### Data Transformation
- **Log Transformation:** Apply to `views` variable to reduce extreme value impact
- **Feature Scaling:** Normalize variables for improved model stability

### 🎯 Advanced Modeling

#### Algorithm Exploration
- **XGBoost:** Enhanced gradient boosting for complex relationships
- **Neural Networks:** Deep learning for non-linear pattern recognition
- **Ensemble Methods:** Combine multiple models for improved accuracy

#### Segmentation Strategy
- **Category-Specific Models:** Train specialized models for different video categories
- **Audience Segmentation:** Develop models for distinct viewer demographics

### 📊 Analysis Extensions

#### Temporal Analysis
- **Growth Patterns:** Evaluate view evolution over time
- **Trend Identification:** Identify seasonal and cyclical patterns

#### Dataset Enhancement
- **Sample Size:** Increase dataset size for better representation
- **Feature Diversity:** Include additional metadata and engagement metrics

---

## 📚 References

1. **Brownlee, J.** (2021). *How to Evaluate Machine Learning Algorithms*. Machine Learning Mastery. [https://machinelearningmastery.com/evaluate-machine-learning-algorithms/](https://machinelearningmastery.com/evaluate-machine-learning-algorithms/)

2. **Aggarwal, C. C.** (2015). *Data Mining: The Textbook*. Springer. (Available in academic repositories)

3. **Tibshirani, R.** (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society: Series B (Methodological)*, 58(1), 267–288.

4. **Python Software Foundation** (2023). *Python Documentation*. [https://docs.python.org/3/](https://docs.python.org/3/)

---

**Project Contributors:** Data Science Team | **Last Updated:** [Current Date] | **Version:** 1.0
