# 📘 Label Encoding in Machine Learning

## 🔹 What is Label Encoding?
Label Encoding is a preprocessing technique used to convert **categorical values** (text/labels) into **numerical values** so that machine learning algorithms can interpret them.

For example:


---

## 🔹 When to Use Label Encoding
- ✅ **Ordinal Data (ordered categories)**
  - Example: `["Low", "Medium", "High"]`
  - Encoded as: `Low=0, Medium=1, High=2`
  - Preserves the order.

- ✅ **Binary Data (two categories)**
  - Example: `["Yes", "No"]`
  - Encoded as: `Yes=1, No=0`

- ✅ **Tree-based Models (Decision Tree, Random Forest, XGBoost)**
  - These models don’t assume mathematical relationships between labels, so label encoding works fine.

---

## 🔹 When *Not* to Use Label Encoding
- ❌ For **Nominal Data** (no natural order) with **many categories**
  - Example: `["Red", "Blue", "Green"]`
  - Encoding as `Red=0, Blue=1, Green=2` may confuse models (e.g., thinking Green > Red).
  - In such cases, use **One-Hot Encoding** instead.

---

## 🔹 Example in Python
```python
from sklearn.preprocessing import LabelEncoder

# Sample data
colors = ["Red", "Blue", "Green", "Blue", "Red"]

# Apply Label Encoding
encoder = LabelEncoder()
encoded_colors = encoder.fit_transform(colors)

print("Original:", colors)
print("Encoded:", encoded_colors)
print("Mapping:", dict(zip(encoder.classes_, range(len(encoder.classes_)))))

# Output:
Original: ['Red', 'Blue', 'Green', 'Blue', 'Red']
Encoded: [2 0 1 0 2]
Mapping: {'Blue': 0, 'Green': 1, 'Red': 2}
```

### 🔹 Label Encoding vs One-Hot Encoding

| Feature Type        | Preferred Encoding |
| ------------------- | ------------------ |
| **Ordinal**         | Label Encoding     |
| **Binary**          | Label Encoding     |
| **Nominal (multi)** | One-Hot Encoding   |


### 🔹 Key Takeaways

* Use Label Encoding for ordered or binary categorical features.

* Use One-Hot Encoding when categories are unordered and many.

* Be careful: improper use of label encoding may introduce false orderings into the model.
