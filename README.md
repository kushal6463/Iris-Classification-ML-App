# Iris-Classification-ML-App


This repository contains a simple machine learning web application built with [Streamlit](https://streamlit.io/). The app classifies iris species using a RandomForestClassifier model from scikit-learn. The app allows users to input flower measurements (sepal length, sepal width, petal length, and petal width) and returns the predicted iris species.

## Features

- **Data Exploration**: Displays the first 10 rows of the Iris dataset.
- **Model Training**: Trains a RandomForestClassifier on the Iris dataset.
- **User Input**: Allows users to provide input for sepal and petal dimensions through sliders in the sidebar.
- **Prediction**: Displays the predicted iris species based on the user input.
- **Model Accuracy**: Shows the accuracy of the trained model on a test set.

### Prerequisites

- Python 3.7+
- streamlit


### Running the App

To run the Streamlit app locally, use the following command:

```bash
streamlit run app.py
```

This will launch the application in your web browser.
