import os
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def save_text_as_image(text, filepath, title=None, figsize=(8, 5)):
    """Helper to render console text as a clean PNG image."""
    plt.figure(figsize=figsize)
    plt.axis('off')
    if title:
        plt.text(0.01, 0.98, title, fontsize=12, fontweight='bold', va='top', family='sans-serif')
        plt.text(0.01, 0.88, text, fontsize=10, family='monospace', va='top', ha='left')
    else:
        plt.text(0.01, 0.98, text, fontsize=10, family='monospace', va='top', ha='left')
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()

def save_df_as_image(df_to_save, filepath, title=None, figsize=(10, 4)):
    """Helper to render a Pandas DataFrame as a clean PNG table."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    if title:
        plt.title(title, fontsize=12, fontweight='bold', pad=10, loc='left')
    
    # Check if we should include row labels (index)
    include_index = not isinstance(df_to_save.index, pd.RangeIndex)
    cell_text = df_to_save.round(3).values
    col_labels = list(df_to_save.columns)
    row_labels = list(df_to_save.index) if include_index else None
    
    table = ax.table(
        cellText=cell_text, 
        colLabels=col_labels, 
        rowLabels=row_labels, 
        loc='center', 
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.3)
    
    # Style the headers
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2f2f2')
            
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()

def main():
    print("=" * 60)
    print("        IRIS FLOWER CLASSIFICATION PIPELINE")
    print("=" * 60)

    # Set up directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    plots_dir = os.path.join(project_root, "assets", "screenshots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. library_test.png
    save_text_as_image(
        "All libraries imported successfully!", 
        os.path.join(plots_dir, "library_test.png"), 
        title="Library Import Verification"
    )

    # 2. Loading dataset
    print("\n[Step 1] Loading Iris Dataset...")
    iris = load_iris()
    
    # Save raw iris data array text output
    raw_data_sample = str(iris.data[:15]) + "\n ..."
    save_text_as_image(
        raw_data_sample, 
        os.path.join(plots_dir, "iris_data.png"), 
        title="Raw Iris Target Data Sample"
    )

    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print(f"Dataset successfully loaded with {df.shape[0]} rows and {df.shape[1] - 1} features.")

    # 3. df_head.png
    save_df_as_image(
        df.head(), 
        os.path.join(plots_dir, "df_head.png"), 
        title="df.head() - First 5 rows of Dataset"
    )

    # 4. Dataset_info.png
    buffer = io.StringIO()
    df.info(buf=buffer)
    save_text_as_image(
        buffer.getvalue(), 
        os.path.join(plots_dir, "Dataset_info.png"), 
        title="Dataset Information (df.info())"
    )

    # 5. Dataset_description.png
    save_df_as_image(
        df.describe(), 
        os.path.join(plots_dir, "Dataset_description.png"), 
        title="Dataset Description Statistics (df.describe())",
        figsize=(11, 5)
    )

    # 6. Scatterplot (using default Seaborn colors to match friend's image output)
    print("Generating Sepal Length vs Sepal Width Scatterplot...")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species', data=df)
    plt.title('Sepal Length vs Sepal Width', fontsize=14, pad=15)
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.savefig(os.path.join(plots_dir, "scatterplot.png"), bbox_inches='tight', dpi=150)
    plt.close()

    # 7. Pairplot (using default Seaborn colors to match friend's image output)
    print("Generating Pairplot...")
    pairplot = sns.pairplot(df, hue='species')
    pairplot.fig.suptitle("Iris Feature Relationships by Species", y=1.02, fontsize=16)
    pairplot.savefig(os.path.join(plots_dir, "pairplot.png"), bbox_inches='tight', dpi=150)
    plt.close()

    # 8. Correlation Heatmap
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(8, 6))
    corr_matrix = df.drop('species', axis=1).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix of Iris Features", fontsize=14, pad=15)
    plt.savefig(os.path.join(plots_dir, "feature correlation heatmap.png"), bbox_inches='tight', dpi=150)
    plt.close()

    # 9. Split Data
    print("\n[Step 3] Splitting Dataset...")
    X = df.drop('species', axis=1)
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    split_info = (
        f"X_train shape: {X_train.shape}\n"
        f"X_test shape: {X_test.shape}\n"
        f"y_train shape: {y_train.shape}\n"
        f"y_test shape: {y_test.shape}"
    )
    save_text_as_image(
        split_info, 
        os.path.join(plots_dir, "training and test data.png"), 
        title="Training and Test Set Dimensions"
    )

    # 10. Model Training
    print("\n[Step 4] Training Model...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    save_text_as_image(
        "Model trained successfully!", 
        os.path.join(plots_dir, "model training.png"), 
        title="Model Training Verification"
    )

    # 11. Model Prediction on test set
    y_pred = model.predict(X_test)
    pred_sample = (
        "First 10 Predictions on test set:\n"
        f"{list(y_pred[:10])}"
    )
    save_text_as_image(
        pred_sample, 
        os.path.join(plots_dir, "model prediction.png"), 
        title="Predictions on secret Exam Set"
    )

    # 12. Accuracy & Confusion Matrix text
    accuracy = accuracy_score(y_test, y_pred)
    acc_text = (
        f"Accuracy: {accuracy:.4f}\n"
        f"accuracy Percentage: {accuracy * 100:.2f} %"
    )
    save_text_as_image(
        acc_text, 
        os.path.join(plots_dir, "accuracy and confusion matrix.png"), 
        title="Model Accuracy Performance"
    )

    # 13. Confusion Matrix heatmap (matching friend's label casing and figsize)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        confusion_matrix(y_test, y_pred), 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=['setosa', 'versicolor', 'virginica'], 
        yticklabels=['setosa', 'versicolor', 'virginica']
    )
    plt.title("Confusion Matrix Heatmap", fontsize=14, pad=15)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(plots_dir, "confusion matrix heatmap.png"), bbox_inches='tight', dpi=150)
    plt.close()

    # 14. Classification Report
    report = classification_report(y_test, y_pred)
    save_text_as_image(
        report, 
        os.path.join(plots_dir, "classification report.png"), 
        title="Model Classification Report"
    )

    # 15. Predictions on unknown data
    custom_samples = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2],
         [6.0, 2.9, 4.5, 1.5],
         [6.7, 3.0, 5.8, 2.2]],
        columns=X.columns
    )
    predictions = model.predict(custom_samples)
    custom_preds_text = ""
    for i, pred_name in enumerate(predictions):
        custom_preds_text += f"Flower {i+1}: {pred_name}\n"
        
    save_text_as_image(
        custom_preds_text, 
        os.path.join(plots_dir, "model prediction with unknown data.png"), 
        title="Model predictions on new/unknown flowers"
    )

    print("\n" + "=" * 60)
    print("              PIPELINE EXECUTION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
