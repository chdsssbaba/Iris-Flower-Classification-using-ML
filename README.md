# 🌸 Iris Flower Classification

<p align="center">
  <img src="assets/images/iris_flower_classification.png" alt="Iris Flower Classification Banner" width="100%">
</p>

Welcome! This project is a friendly, step-by-step guide to teaching a computer how to identify different species of **Iris flowers** using **Machine Learning**. 

This guide is written in very simple English, so even if you have no background in computer science or programming, you will easily understand everything!

---

## 👨‍💻 Author
*   **Name:** CH D S S S BABA
*   **Project:** Machine Learning with Python Internship Project

---

## 📚 What is this Project About?

Imagine you have three different types of Iris flowers: **Setosa**, **Versicolor**, and **Virginica**. They look very similar, but they have different sizes.

We want to build a computer program that can look at the size of a new flower and instantly guess its correct type.

To do this, we measure four parts of the flower:
1.  **Sepal Length** (the length of the green outer leaves)
2.  **Sepal Width** (the width of the green outer leaves)
3.  **Petal Length** (the length of the colorful inner petals)
4.  **Petal Width** (the width of the colorful inner petals)

---

## 📈 The Visual Workflow (How it Works)

Here is a simple flow chart showing exactly how the computer learns and makes guesses:

```mermaid
flowchart TD
    Start([1. Start Project]) --> Load[2. Load the Flower Data]
    Load --> Explore[3. Explore & Draw Graphs]
    Explore --> Split[4. Split into Study & Exam Sets]
    Split --> Train[5. Train the AI Model]
    Train --> Test[6. Test & Check Accuracy]
    Test --> Custom[7. Predict on Brand New Flowers]
    Custom --> End([8. Complete!])

    style Start fill:#ffe3e3,stroke:#ff8585,stroke-width:2px,color:#000000
    style Load fill:#e3f2fd,stroke:#42a5f5,stroke-width:2px,color:#000000
    style Explore fill:#e8f5e9,stroke:#66bb6a,stroke-width:2px,color:#000000
    style Split fill:#fff3e0,stroke:#ffb74d,stroke-width:2px,color:#000000
    style Train fill:#e0f7fa,stroke:#26c6da,stroke-width:2px,color:#000000
    style Test fill:#fffde7,stroke:#ffee58,stroke-width:2px,color:#000000
    style Custom fill:#ede7f6,stroke:#7e57c2,stroke-width:2px,color:#000000
    style End fill:#e0f2f1,stroke:#26a69a,stroke-width:2px,color:#000000
```

---

## 📂 Project Structure
```text
.
├── assets/
│   ├── images/
│   │   └── iris_flower_classification.png                  <-- Our minimalist banner
│   └── screenshots/                                        <-- Visual graphs and cell outputs
│       ├── Dataset_description.png
│       ├── Dataset_info.png
│       ├── accuracy and confusion matrix.png
│       ├── classification report.png
│       ├── confusion matrix heatmap.png
│       ├── df_head.png
│       ├── feature correlation heatmap.png
│       ├── iris_data.png
│       ├── library_test.png
│       ├── model prediction with unknown data.png
│       ├── model prediction.png
│       ├── model training.png
│       ├── pairplot.png
│       ├── scatterplot.png
│       └── training and test data.png
├── docs/
│   └── Iris_Flower_Classification_Report.pdf               <-- Detailed project report document
├── src/
│   ├── classify_iris.py                                     <-- The main Python program
│   └── Iris_Flower_Classification.ipynb                     <-- Interactive Jupyter Notebook
├── .gitignore
├── README.md                                               <-- This instruction guide
└── requirements.txt                                        <-- Libraries to install
```

---

## 🚀 How to Run it Yourself

### Option A: Run the Python script
1.  **Install the Libraries:**
    Open your command prompt or terminal and run:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the script:**
    ```bash
    python src/classify_iris.py
    ```

### Option B: Run the Interactive Jupyter Notebook
1.  Open the file **`src/Iris_Flower_Classification.ipynb`** in VS Code, Jupyter Lab, or Google Colab.
2.  Run the cells step-by-step to view the interactive plots and predictions instantly!

---

## 📊 Step-by-Step Walkthrough with Screenshots

Below is the complete execution pipeline showing what the program does at every stage, accompanied by the generated screenshot output:

### 1. Library Testing (`library_test.png`)
Verifies that all required packages (`pandas`, `matplotlib`, `seaborn`, `sklearn`) are installed properly.
<p align="center">
  <img src="assets/screenshots/library_test.png" alt="Library Test" width="50%">
</p>

### 2. Raw Dataset (`iris_data.png`)
Displays a sample of the raw numerical measurements loaded directly from the Iris dataset.
<p align="center">
  <img src="assets/screenshots/iris_data.png" alt="Iris Data" width="50%">
</p>

### 3. Data Frame Head (`df_head.png`)
Shows the first few rows of the cleaned table with the column names and categorical species labels.
<p align="center">
  <img src="assets/screenshots/df_head.png" alt="Data Frame Head" width="70%">
</p>

### 4. Dataset Information (`Dataset_info.png`)
Shows details about the columns, row count, data types, and check for missing/null values.
<p align="center">
  <img src="assets/screenshots/Dataset_info.png" alt="Dataset Info" width="70%">
</p>

### 5. Dataset Description (`Dataset_description.png`)
Generates descriptive statistical summaries (mean, standard deviation, min, max) of the features.
<p align="center">
  <img src="assets/screenshots/Dataset_description.png" alt="Dataset Description" width="80%">
</p>

### 6. Scatterplot (`scatterplot.png`)
Plots sepal length against sepal width to visually group the three flower types.
<p align="center">
  <img src="assets/screenshots/scatterplot.png" alt="Scatterplot" width="60%">
</p>

### 7. Pairplot (`pairplot.png`)
Displays scatter plots for every combination of measurements.
<p align="center">
  <img src="assets/screenshots/pairplot.png" alt="Pairplot" width="80%">
</p>

### 8. Feature Correlation Heatmap (`feature correlation heatmap.png`)
Highlights strong relationships between different features using warm and cool colors.
<p align="center">
  <img src="assets/screenshots/feature correlation heatmap.png" alt="Feature Correlation Heatmap" width="60%">
</p>

### 9. Training and Test Data Splitting (`training and test data.png`)
Shows the dimensions of the split training set (120 samples) and testing set (30 samples).
<p align="center">
  <img src="assets/screenshots/training and test data.png" alt="Training and Test Data Split" width="50%">
</p>

### 10. Model Training (`model training.png`)
Confirms successful training of the Logistic Regression model.
<p align="center">
  <img src="assets/screenshots/model training.png" alt="Model Training" width="50%">
</p>

### 11. Model Prediction on Test Set (`model prediction.png`)
Shows the predicted species classifications for the first 10 unseen test samples.
<p align="center">
  <img src="assets/screenshots/model prediction.png" alt="Model Prediction" width="50%">
</p>

### 12. Accuracy Score (`accuracy and confusion matrix.png`)
Calculates the final accuracy percentage of the model, which is 100%.
<p align="center">
  <img src="assets/screenshots/accuracy and confusion matrix.png" alt="Accuracy Score" width="50%">
</p>

### 13. Confusion Matrix Heatmap (`confusion matrix heatmap.png`)
Shows the counts of correct predictions versus incorrect predictions.
<p align="center">
  <img src="assets/screenshots/confusion matrix heatmap.png" alt="Confusion Matrix Heatmap" width="50%">
</p>

### 14. Classification Report (`classification report.png`)
Shows precision, recall, and f1-score performance metrics for all three species.
<p align="center">
  <img src="assets/screenshots/classification report.png" alt="Classification Report" width="60%">
</p>

### 15. Predictions with Unknown/New Data (`model prediction with unknown data.png`)
Shows the predicted species for three brand new flower samples:
*   Sample 1 `[5.1, 3.5, 1.4, 0.2]`: **Setosa**
*   Sample 2 `[6.0, 2.9, 4.5, 1.5]`: **Versicolor**
*   Sample 3 `[6.7, 3.0, 5.8, 2.2]`: **Virginica**
<p align="center">
  <img src="assets/screenshots/model prediction with unknown data.png" alt="Model Prediction with Unknown Data" width="50%">
</p>