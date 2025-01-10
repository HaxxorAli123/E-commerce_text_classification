# E-commerce text classification


# Overview

This repository contains a project for creating a text classification model designed to predict and categorize items commonly found on e-commerce websites into one of four predefined categories:

* Electronics

* Household

* Books

* Clothing & Accessories

The model analyzes item descriptions to determine the appropriate category.

# Key Features

* Text-based Categorization: The model uses item descriptions to predict the category.

* High Accuracy: Leverages advanced deep learning techniques for better performance.

* Scalable: Can be easily expanded to include additional categories if needed.

# Model Architecture

The core of the classification model is a Bidirectional LSTM (Long Short-Term Memory) neural network, which is particularly effective for text data due to its ability to capture contextual dependencies in sequences.

# Why Bidirectional LSTM?

Bidirectional LSTMs are used because they process data in both forward and backward directions, enabling the model to capture dependencies from both past and future contexts in the text data.

# Technology Stack

* Programming Language: Python

* Libraries Used:

  * TensorFlow & Keras: For building and training the deep learning model.

  * Scikit-learn: For preprocessing and performance evaluation.

  * Pandas and NumPy: For data manipulation and analysis.

  * Matplotlib and Seaborn: For visualizing model performance.

# Workflow

1. Data Preprocessing:

  * Cleaning and tokenizing text data.

  * Vectorizing text using techniques like word embeddings.

  * Splitting data into training, validation, and test sets.

2. Model Development:

  * Building a Bidirectional LSTM model using Keras.

  * Compiling the model with appropriate loss functions and optimizers.

3. Training:

  * Training the model on labeled e-commerce item descriptions.

  * Validating the model using a separate validation set.

4. Evaluation:

  * Evaluating the model's performance using metrics like accuracy, precision, recall, and F1-score.

  * Generating a confusion matrix for detailed analysis.

5. Deployment (Optional):

  * Saving the trained model.

  * Deploying it as a web service or integrating it into an e-commerce platform.

below is an overview of the Bidirectional LSTM model architecture

![image](https://github.com/user-attachments/assets/95cb3320-ab85-4672-acdd-3ac5ae795395)
