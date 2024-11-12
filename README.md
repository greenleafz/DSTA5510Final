# Wisconsin Breast Cancer Clustering Analysis

This repository contains an unsupervised learning project on the **Wisconsin Breast Cancer Dataset**. We aim to apply various clustering algorithms to identify patterns within the dataset, specifically distinguishing between benign and malignant cases. The notebook leverages multiple clustering models and an ensemble approach to optimize performance and evaluate clustering accuracy.

## Dataset Overview

The **Wisconsin Breast Cancer Dataset** contains features derived from fine needle aspirate (FNA) images of breast masses. Key features include radius, texture, perimeter, area, and other cell nucleus characteristics. This dataset is commonly used in machine learning, particularly for studies related to medical diagnosis.

## Project Structure

This project is organized into several key sections:

1. **Data Preprocessing**  
   - Standardizes features for accurate clustering.
   - Prepares data for dimensionality reduction and visualization.

2. **Exploratory Data Analysis (EDA)**  
   - Descriptive statistics and visualizations.
   - Correlation analysis between features.

3. **Dimensionality Reduction**  
   - Applies PCA and t-SNE for visualization in 2D space.

4. **Clustering Models**  
   - **K-Means**: A distance-based clustering algorithm.
   - **Spectral Clustering**: Captures complex structures in data using eigenvalues of a similarity matrix.
   - **Gaussian Mixture Model (GMM)**: Assumes clusters follow a Gaussian distribution.

5. **Model Evaluation**  
   - Uses the **Adjusted Rand Score (ARS)** to measure similarity between predicted clusters and true labels.
   - Evaluates models based on precision, recall, F1-score, and ARS.

6. **Ensemble Clustering**  
   - Combines individual model results for improved clustering accuracy.

## Key Findings

- **Spectral Clustering** achieved the highest recall, performing well at identifying malignant cases.
- **Gaussian Mixture Model** demonstrated the highest precision, minimizing false positives.
- The **Ensemble Method** balanced precision and recall effectively, achieving the highest ARS.

## Future Directions

Future studies could explore larger diagnostic datasets or investigate deep learning clustering techniques for more complex data. This approach may further improve the clustering accuracy necessary for critical medical diagnoses.

## Requirements

To run the notebook, install the following libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

## Usage

1. Clone this repository.
2. Open the `Exploring Clustering Techniques for the Wisconsin Breast Cancer Dataset.ipynb` notebook in Jupyter Notebook or JupyterLab.
3. Execute cells to preprocess data, run clustering models, and evaluate results.

