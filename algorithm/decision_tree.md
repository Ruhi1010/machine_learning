# Decision Tree 

## Overview
This project demonstrates the application of **Decision Tree Classifiers** on two datasets:
1. **Dataset 1** – A synthetic dataset of employees and their salaries.
2. **Dataset 2** – The Titanic survival dataset.

The notebook (`decision_tree.ipynb`) walks through the preprocessing, model training, and evaluation steps for both datasets. The primary objective is to show how decision trees can be used to classify categorical and numerical data, understand feature importance, and visualize the reasoning behind predictions.

---

## What is a Decision Tree?
A **Decision Tree** is a supervised machine learning algorithm used for **classification** and **regression** tasks. It splits data into subsets based on feature values, creating a **tree-like structure** where each internal node represents a decision rule, and each leaf node represents an outcome.

### Structure:
- **Root Node**: The first split based on the most important feature.
- **Decision Nodes**: Intermediate nodes that represent feature-based questions.
- **Leaf Nodes**: The final output labels or values.

### Example:
A simple tree for predicting salary might look like:


---

## How Decision Trees Work
Decision trees use measures of **impurity** to decide how to split data:

1. **Entropy**  
   - Measures the randomness in data.  
   - Formula:  
     Entropy = - Σ p_i log2(p_i)  
   - Lower entropy = purer split.

2. **Information Gain**  
   - The reduction in entropy after a split.  
   - Higher information gain = better split.

3. **Gini Index**  
   - Another impurity measure (used by default in scikit-learn).  
   - Formula:  
     Gini = 1 - Σ p_i²

4. **Pruning**  
   - Trees can become too complex and overfit.  
   - Pruning techniques (e.g., max depth, min samples per leaf) are used to reduce complexity.

---
## Background and Real-World Applications

Decision Trees are widely used because they are **interpretable** and can handle **categorical and numerical features** without heavy preprocessing. Below are some real-world domains where decision trees play a key role:

### 1. Finance
- **Credit Risk Analysis**: Banks use decision trees to evaluate loan applicants based on income, employment history, credit score, and other attributes.  
- **Fraud Detection**: Transactions can be flagged as fraudulent or legitimate using branching rules.  

### 2. Healthcare
- **Disease Diagnosis**: Decision trees can predict whether a patient is likely to have a disease based on symptoms, test results, and demographic factors.  
- **Treatment Recommendations**: Medical decision-making systems suggest treatment options based on patient conditions.  

### 3. Marketing & Sales
- **Customer Segmentation**: Marketers use decision trees to segment customers into categories such as "likely buyer" or "not interested".  
- **Churn Prediction**: Telecom companies can identify customers at risk of leaving and create targeted retention strategies.  

### 4. Human Resources
- **Hiring Decisions**: Decision trees help HR teams identify the best candidates based on qualifications, skills, and experience.  
- **Attrition Prediction**: Companies predict which employees are likely to leave the organization and take preventive actions.  

### 5. Manufacturing & Operations
- **Quality Control**: Trees can classify products as "defective" or "non-defective" based on manufacturing parameters.  
- **Supply Chain Optimization**: Helps in deciding stocking strategies based on demand patterns.  

### 6. Transportation
- **Predicting Delays**: Airlines and logistics companies use decision trees to predict delays based on weather, traffic, and scheduling data.  
- **Route Optimization**: Decision-making for choosing cost-effective and time-efficient routes.  

### Advantages in Practice:
- **Interpretability**: Managers and non-technical staff can understand decisions.  
- **Speed**: Quick to train and predict, suitable for real-time systems.  
- **Versatility**: Applicable across industries with both structured and unstructured data.  

### Limitations in Practice:
- **Overfitting**: Pure decision trees can perform poorly on unseen data.  
- **Bias**: If training data is biased, predictions will reflect that bias.  
- **Scalability**: Very deep trees can be computationally expensive and hard to manage.  

---

## Datasets

### Dataset 1: Employee Salaries
- **File:** `dataset1.csv`
- **Shape:** 100 rows × 4 columns
- **Columns:**
  - `company`: Employer (e.g., Google, Amazon, Microsoft)
  - `job_position`: Job title (e.g., Software Engineer, Data Scientist)
  - `degree`: Education level (Bachelors, Masters, PhD, etc.)
  - `salary_more_than_100k`: Target variable (1 = Yes, 0 = No)

**Sample Data:**

| company   | job_position         | degree     | salary_more_than_100k |
|-----------|----------------------|-----------|------------------------|
| Google    | Software Engineer    | Masters   | 1 |
| Amazon    | Data Scientist       | PhD       | 1 |
| Microsoft | System Administrator | Associates| 0 |
| Facebook  | Product Manager      | Bachelors | 1 |
| Apple     | Marketing Specialist | Bachelors | 0 |

---

### Dataset 2: Titanic Survival
- **File:** `dataset2.csv`
- **Shape:** 891 rows × 12 columns
- **Columns:**
  - `PassengerId`: Unique passenger identifier
  - `Survived`: Target variable (1 = Survived, 0 = Did not survive)
  - `Pclass`: Passenger class (1 = First, 2 = Second, 3 = Third)
  - `Name`: Passenger name
  - `Sex`: Gender
  - `Age`: Age in years
  - `SibSp`: Number of siblings/spouses aboard
  - `Parch`: Number of parents/children aboard
  - `Ticket`: Ticket number
  - `Fare`: Ticket fare
  - `Cabin`: Cabin number (missing for many passengers)
  - `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

**Sample Data:**

| PassengerId | Survived | Pclass | Name                               | Sex    | Age | SibSp | Parch | Fare  | Embarked |
|-------------|----------|--------|------------------------------------|--------|-----|-------|-------|-------|----------|
| 1           | 0        | 3      | Braund, Mr. Owen Harris            | male   | 22  | 1     | 0     | 7.25  | S |
| 2           | 1        | 1      | Cumings, Mrs. John Bradley (Florence Briggs Thayer) | female | 38  | 1     | 0     | 71.28 | C |
| 3           | 1        | 3      | Heikkinen, Miss. Laina             | female | 26  | 0     | 0     | 7.92  | S |

---

## Methodology

### Step 1: Data Loading
- Import datasets using `pandas`.
- Preview structure, dimensions, and missing values.

---

### Step 2: Preprocessing
- **Dataset 1**:
  - Encode categorical variables (`company`, `job_position`, `degree`) using `LabelEncoder`.
- **Dataset 2 (Titanic)**:
  - Handle missing values (`Age` filled with mean, `Cabin` dropped, `Embarked` filled with mode).
  - Encode categorical variables (`Sex`, `Embarked`) using encoding techniques.

---

### Step 3: Splitting Data
- Separate inputs (features `X`) and target (`y`).
- Split into training and testing sets using `train_test_split`.

---

### Step 4: Model Training
- Train using **DecisionTreeClassifier** from scikit-learn.
- Specify parameters like `criterion`, `max_depth`, and `random_state`.

---

### Step 5: Evaluation
- Measure performance with **accuracy score**.
- Visualize decision tree with `plot_tree`.
- Analyze **feature importance**.

---

## Results

### Dataset 1 (Employee Salaries)
- The decision tree achieved **perfect accuracy** because the rules were clear and categorical:
  - Certain companies and degrees strongly influenced salaries above $100k.

### Dataset 2 (Titanic Survival)
- Accuracy ranged **75–80%**, depending on preprocessing.
- Key findings:
  - **Sex** was the strongest predictor (female passengers had higher survival rates).
  - **Passenger Class (Pclass)** played an important role.
  - **Age** and **Fare** were also relevant features.

---

## Limitations
- **Overfitting**: Decision trees can memorize training data, especially with small datasets.
- **Bias in Data**: Imbalanced datasets may skew predictions.
- **Interpretability in Large Trees**: Small trees are easy to interpret, but large trees become complex.
- **Handling Missing Data**: Preprocessing steps (imputation, dropping columns) influence accuracy.

---

## Future Improvements
- Use **Random Forests** or **Gradient Boosted Trees** for better generalization.
- Perform **cross-validation** to ensure stability of results.
- Tune hyperparameters (`max_depth`, `min_samples_split`) for optimization.
- Engineer new features (e.g., family size in Titanic dataset).
- Compare decision trees with other models like Logistic Regression, SVMs, or Neural Networks.

---

## How to Run
1. Clone or download the project files.
2. Install dependencies:
3. Open the Jupyter Notebook:
4. Ensure both `dataset1.csv` and `dataset2.csv` are in the same folder as the notebook.
5. Run all cells to reproduce results.

---

## Key Takeaways
- Decision Trees are interpretable models that work well with both categorical and numerical data.
- Dataset 1 shows a **clear and simple classification task** with deterministic rules.
- Dataset 2 demonstrates the **real-world challenge** of missing values and mixed feature types.
- While decision trees are powerful, **ensemble methods** (Random Forest, XGBoost) often achieve superior performance.
