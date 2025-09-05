# Random Forest 

## Introduction
This project demonstrates the use of the **Random Forest algorithm** for classification tasks using two datasets:
1. A **salary prediction dataset** (`dataset1.csv`), which predicts whether an employee earns more than 100k based on company, job position, and degree.  
2. The **Titanic dataset** (`dataset2.csv`), which predicts passenger survival during the Titanic disaster based on demographic and travel details.  

Random Forest is an **ensemble learning algorithm** that builds multiple decision trees and combines their predictions. This approach reduces overfitting and improves accuracy compared to a single decision tree.

---

## Random Forest: Brief Overview
- **Definition**: Random Forest is an ensemble method that constructs multiple decision trees and aggregates their results (majority voting for classification, averaging for regression).  
- **Key Idea**: Instead of relying on a single tree (which may overfit), Random Forest combines the output of many trees trained on random subsets of data and features.  
- **Advantages**:
  - High accuracy and robustness.  
  - Handles both categorical and numerical data.  
  - Resistant to overfitting compared to a single decision tree.  
  - Provides feature importance scores.  
- **Limitations**:
  - Slower training with very large datasets.  
  - Less interpretable compared to a single tree.  

---

## Datasets

### 1. Salary Prediction Dataset (`dataset1.csv`)
- **Shape**: (100, 4)  
- **Columns**:
  - `company`: Name of the company (e.g., Google, Amazon, Apple).  
  - `job_position`: Job role (e.g., Software Engineer, Data Scientist).  
  - `degree`: Education qualification (e.g., Bachelors, Masters, PhD).  
  - `salary_more_than_100k`: Target variable (1 = Yes, 0 = No).  

**Sample Data**:
| company   | job_position        | degree    | salary_more_than_100k |
|-----------|---------------------|-----------|------------------------|
| Google    | Software Engineer   | Masters   | 1                      |
| Amazon    | Data Scientist      | PhD       | 1                      |
| Microsoft | System Administrator| Associates| 0                      |

**Objective**: Predict whether an employee earns more than 100k.  

---

### 2. Titanic Dataset (`dataset2.csv`)
- **Shape**: (891, 12)  
- **Columns**:
  - `PassengerId`: Unique identifier.  
  - `Survived`: Target variable (1 = Survived, 0 = Did not survive).  
  - `Pclass`: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd).  
  - `Name`: Passenger’s name.  
  - `Sex`: Gender.  
  - `Age`: Age in years.  
  - `SibSp`: Number of siblings/spouses aboard.  
  - `Parch`: Number of parents/children aboard.  
  - `Ticket`: Ticket number.  
  - `Fare`: Ticket fare.  
  - `Cabin`: Cabin number (many missing values).  
  - `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).  

**Sample Data**:
| PassengerId | Survived | Pclass | Sex    | Age | Fare  | Embarked |
|-------------|----------|--------|--------|-----|-------|----------|
| 1           | 0        | 3      | male   | 22  | 7.25  | S        |
| 2           | 1        | 1      | female | 38  | 71.28 | C        |
| 3           | 1        | 3      | female | 26  | 7.92  | S        |

**Objective**: Predict passenger survival based on available features.  

---

## Workflow in the Notebook

### 1. Data Loading
- Import CSV files using Pandas (`read_csv`).  
- Explore datasets using `.head()`, `.info()`, and `.describe()`.  

### 2. Data Preprocessing
- Convert categorical variables into numerical form (Label Encoding / One-Hot Encoding).  
- Handle missing values in the Titanic dataset (e.g., filling missing `Age` with median).  
- Split data into **features (X)** and **target (y)**.  

### 3. Train-Test Split
- Use `train_test_split` from Scikit-learn to split data into training and testing sets (commonly 70-30 or 80-20).  

### 4. Model Training
- Import `RandomForestClassifier` from `sklearn.ensemble`.  
- Initialize model with parameters (e.g., `n_estimators=100`, `max_depth=None`).  
- Train the model using `.fit(X_train, y_train)`.  

### 5. Model Prediction
- Use `.predict(X_test)` to generate predictions.  
- Compare predictions against actual test labels.  

### 6. Model Evaluation
- Metrics used:
  - **Accuracy**: `(correct predictions) / (total predictions)`.  
  - **Confusion Matrix**: Shows TP, FP, TN, FN.  
  - **Classification Report**: Precision, Recall, F1-score.  
- Feature importance is also extracted to understand which features contributed most to predictions.  

---

## Results and Interpretation

### Salary Dataset:
- Model successfully predicts whether an employee earns more than 100k.  
- Features like **job position** and **degree** have high importance.  
- Achieves **very high accuracy**, since the dataset is small and simple.  

### Titanic Dataset:
- Random Forest achieves higher accuracy compared to a single Decision Tree.  
- Important features:
  - **Sex**: Females had a higher survival chance.  
  - **Pclass**: 1st class passengers were more likely to survive.  
  - **Age**: Younger passengers had better survival rates.  
- Typical accuracy: **~80%** (depending on preprocessing and hyperparameters).  

---

##  Advantages, Limitations, and Conclusion

### Advantages
- Outperforms a single decision tree by reducing overfitting.  
- Works well with both categorical (e.g., sex, company) and numerical (e.g., age, fare) features.  
- Provides **feature importance**, which helps interpret the model.  
- Can handle large datasets and complex patterns effectively.  

### Limitations
- Model training may take longer for very large datasets.  
- Harder to interpret compared to a simple decision tree.  
- Hyperparameter tuning (e.g., `n_estimators`, `max_depth`) is often necessary to achieve optimal performance.  

### Conclusion
- **Salary Dataset**: Random Forest accurately predicts whether an employee earns more than 100k based on company, job, and degree.  
- **Titanic Dataset**: Random Forest improves prediction accuracy of passenger survival by capturing complex relationships in the data.  
- Overall, Random Forest is a **robust, reliable algorithm** that balances bias and variance, making it highly suitable for real-world applications in both structured business data and historical survival prediction tasks.  

---
👉 Random Forest is a **robust, reliable algorithm** that balances bias and variance, making it highly suitable for real-world applications.