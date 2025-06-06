# Healthcare Stroke Prediction with PySpark

A comprehensive machine learning project using Apache Spark (PySpark) to predict stroke risk based on healthcare data.

## Project Overview

This project implements a complete data science pipeline for predicting stroke occurrence using patient health data. The analysis includes data preprocessing, feature engineering, machine learning model training, and performance evaluation.

## Dataset

The project uses a healthcare stroke dataset (`HealthCareStroke_Dataset.csv`) containing patient information including:

- **Demographics**: Age, gender, marital status
- **Health Conditions**: Hypertension, heart disease
- **Lifestyle Factors**: Work type, residence type, smoking status
- **Medical Metrics**: Average glucose level, BMI
- **Target Variable**: Stroke occurrence (binary classification)

## Technical Stack

- **Apache Spark (PySpark)**: Distributed data processing and machine learning
- **PySpark MLlib**: Machine learning algorithms and pipelines
- **Matplotlib**: Data visualization
- **NumPy**: Numerical computations

## Key Features

### 1. Data Preprocessing
- **Missing Value Handling**: Imputation using mode values for categorical variables
- **Data Type Conversion**: Ensuring consistent data types across features
- **Data Quality**: Removal of completely null records

### 2. Feature Engineering
- **Categorical Encoding**: StringIndexer and OneHotEncoder for categorical variables
- **Class Imbalance Handling**: Weighted sampling to address stroke prediction imbalance
- **Feature Assembly**: VectorAssembler for ML pipeline compatibility

### 3. Machine Learning Pipeline
- **Algorithm**: Random Forest Classifier
- **Pipeline Architecture**: Integrated preprocessing and modeling stages
- **Train-Test Split**: 80-20 split with random seed for reproducibility

### 4. Model Evaluation
Comprehensive evaluation using multiple metrics:
- **AUC (Area Under Curve)**: 0.7091
- **Accuracy**: 77.51%
- **Precision**: 0.9382
- **Recall**: 0.7751
- **F1-Score**: 0.8382

### 5. Visualization
Performance metrics visualization using matplotlib bar charts for:
- Precision scores
- Recall scores  
- F1-scores

## Project Structure

```
├── HealthCareStroke_Dataset.csv    # Input dataset
├── stroke_prediction.ipynb         # Main Jupyter notebook
├── cleaned_HealthCareStroke2.csv   # Processed dataset output
└── README.md                       # Project documentation
```

## Key Results

The Random Forest model achieved strong performance with:
- **High Precision (93.82%)**: Low false positive rate
- **Good Recall (77.51%)**: Reasonable true positive detection
- **Balanced F1-Score (83.82%)**: Good overall performance
- **Moderate AUC (70.91%)**: Decent discriminative ability

## Data Processing Highlights

1. **Missing Data Strategy**: Mode imputation for age, BMI, and glucose levels
2. **Categorical Variables**: Proper encoding of gender, marital status, work type, residence type, and smoking status
3. **Class Balancing**: Weighted approach to handle stroke prediction imbalance
4. **Feature Selection**: Comprehensive feature set including demographics, health conditions, and lifestyle factors

## Usage

1. **Environment Setup**: Ensure PySpark is installed and configured
2. **Data Loading**: Place the healthcare stroke dataset in the specified path
3. **Execution**: Run the Jupyter notebook cells sequentially
4. **Output**: Review model performance metrics and visualizations

## Model Performance Interpretation

- The model shows strong precision, indicating reliable positive predictions
- Moderate recall suggests some stroke cases may be missed
- The balanced F1-score indicates good overall performance
- Results suggest the model is suitable for healthcare screening applications

## Future Enhancements

- **Feature Engineering**: Additional derived features from existing variables
- **Model Comparison**: Testing multiple algorithms (Logistic Regression, Gradient Boosting)
- **Hyperparameter Tuning**: Grid search for optimal model parameters
- **Cross-Validation**: More robust model validation techniques
- **Feature Importance**: Analysis of most predictive variables

## Dependencies

```python
pyspark
matplotlib
numpy
pandas (for data exploration)
