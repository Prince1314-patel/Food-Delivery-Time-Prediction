# Food Delivery Time Prediction

A Streamlit-based AI-powered app that predicts food delivery time based on distance, weather, traffic, and other key factors.

## Overview

This project implements a complete machine learning pipeline that includes data preprocessing, exploratory data analysis (EDA), hyperparameter optimization with Optuna, and model evaluation. The best performing model is saved and deployed through a modern Streamlit interface that gracefully handles user inputs and unknown categories.

## Features

- **Comprehensive ML Pipeline:** Data loading, EDA, preprocessing, hyperparameter tuning, and model evaluation.
- **Optimized Model:** Utilizes pipelines and advanced regression models (ElasticNet, Decision Tree, Random Forest, XGBoost, and AdaBoost) with Optuna-based optimization.
- **Streamlit Deployment:** User-friendly interface with a modern UI for real-time prediction.
- **Robust Input Handling:** Incorporates custom CSS and two-column layout for an enhanced user experience, with one-hot encoder adjustments to handle unknown categories.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/food-delivery-time-prediction.git
   cd food-delivery-time-prediction
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   Ensure that your `requirements.txt` includes:
   - `streamlit`
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - `xgboost`
   - `optuna`
   - and any other dependencies used in the project.

## Usage

1. **Train and Save the Model:**

   Run your notebook to train the model and save the best model as `best_model.pkl`.

2. **Deploy the App:**

   With the model saved in the project directory, launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. **Interact with the App:**

   Use the sidebar to input feature values and click the "Predict Delivery Time" button to view the predicted delivery time.

## Project Structure

```
├── app.py                 # Streamlit app for model deployment
├── best_model.pkl         # Saved best model from the training pipeline
├── data/                  # Directory containing the dataset (if applicable)
├── notebooks/             # Jupyter notebooks used for model development and evaluation
├── requirements.txt       # List of project dependencies
└── README.md              # Project documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs, enhancements, or suggestions.