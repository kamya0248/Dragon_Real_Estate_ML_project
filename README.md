# Dragon Real Estate - House Price Predictor

This project leverages machine learning techniques to predict house prices for **Dragon Real Estates** using the **Boston Housing Dataset**. By analyzing various housing features, the model provides data-driven insights to assist in pricing strategies.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Overview](#dataset-overview)
3. [Workflow](#workflow)
4. [Model Selection](#model-selection)
5. [Technologies Used](#technologies-used)
6. [How to Run the Project](#how-to-run-the-project)
7. [Results](#results)
8. [Conclusion](#conclusion)

---

## Introduction
The **Dragon Real Estate - House Price Predictor** project is a beginner-level machine learning task that uses regression models to predict housing prices in the suburbs of Boston. The primary goal is to identify the most suitable machine learning model for this task.

---

## Dataset Overview
### Source
- **Origin**: StatLib library, Carnegie Mellon University
- **Creator**: Harrison, D. and Rubinfeld, D.L., "Hedonic prices and the demand for clean air", 1978.

### Key Details
- **Number of Instances**: 506  
- **Number of Attributes**: 13 continuous attributes and 1 binary-valued attribute  
- **Target Variable**: `MEDV` (Median value of owner-occupied homes in $1000s)

### Attributes
1. **CRIM**: Per capita crime rate by town  
2. **ZN**: Proportion of residential land zoned for large lots  
3. **INDUS**: Proportion of non-retail business acres per town  
4. **CHAS**: Charles River dummy variable (= 1 if tract bounds river, 0 otherwise)  
5. **NOX**: Nitric oxides concentration  
6. **RM**: Average number of rooms per dwelling  
7. **AGE**: Proportion of owner-occupied units built before 1940  
8. **DIS**: Weighted distances to employment centers  
9. **RAD**: Accessibility index to radial highways  
10. **TAX**: Full-value property tax rate  
11. **PTRATIO**: Pupil-teacher ratio  
12. **B**: 1000(Bk - 0.63)^2, where Bk is the proportion of black population  
13. **LSTAT**: % lower status of the population  
14. **MEDV**: Median value of owner-occupied homes  

---

## Workflow
1. **Data Analysis**: Examined the dataset for patterns and correlations using:
   - Scatter plots
   - Correlation matrix
2. **Data Preprocessing**:
   - Handled missing values using the median strategy.
   - Applied data standardization through pipelines.
3. **Model Training**:
   - Trained and compared three models:  
     - Decision Tree Regressor  
     - Linear Regression  
     - Random Forest Regressor  
4. **Evaluation**:
   - Used metrics like **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)**.
   - Employed cross-validation for reliable performance estimation.

---

## Model Selection
Three models were evaluated for performance:
1. **Decision Tree Regressor**:  
   - Mean: 4.28  
   - Standard Deviation: 0.83  
2. **Linear Regression**:  
   - Mean: 5.03  
   - Standard Deviation: 1.06  
3. **Random Forest Regressor** (Best Model):  
   - Mean: 3.30  
   - Standard Deviation: 0.67  

The **Random Forest Regressor** emerged as the most suitable model for this dataset.

---

## Technologies Used
- **Programming Language**: Python  
- **Libraries**: 
  - Pandas
  - NumPy
  - Matplotlib
  - Scikit-learn  
- **Environment**: Jupyter Notebook  

---

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone <repository_link>
   cd <repository_folder>
   
2. Install the required libraries:
   pip install -r requirements.txt

3. Run the jupyter notebook:
   jupyter notebook

4. Load the data.csv file in the working directory

## Results
   The Random Forest Regressor achieved the lowest RMSE and proved to be the most reliable model for predicting house prices.
   The model was saved using joblib for future deployment.

## Conclusion
  This project demonstrates the end-to-end implementation of a machine learning pipeline for regression tasks. By comparing models and using advanced evaluation 
  techniques, we achieved a robust predictor for housing prices.

