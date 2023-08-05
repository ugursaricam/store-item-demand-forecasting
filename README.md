# Store Item Demand Forecasting with LightGBM

![Dataset](https://img.shields.io/badge/dataset-Kaggle-blue.svg) ![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg) ![LightGBM](https://img.shields.io/badge/LightGBM-3.3.5-blue.svg)

This GitHub repository contains the design of a model using the LightGBM algorithm to forecast store item demand based on a time series dataset obtained from a Kaggle competition. Below, you will find the project structure, dataset description, and information about the model used.

## Project Description

The project works with a dataset obtained from a Kaggle competition, consisting of 5 years of store-item sales data. Our goal is to predict the sales of 50 different items at 10 different stores for a period of 3 months. In this kind of time series forecasting problem, using a powerful gradient boosting algorithm like LightGBM can yield successful results.

## Dataset

The dataset is sourced from a Kaggle competition and can be accessed [here](https://www.kaggle.com/something/store-item-sales-time-series-forecasting). It has the following structure:

- `date`: Date of the sale
- `store_id`: Identifier of the store
- `item_id`: Identifier of the item
- `sales`: Daily sales quantity

The dataset has been preprocessed, and missing data has been handled.

## Installation

To run the project on your local machine, follow these steps:

1. Clone the repository:
git clone https://github.com/ugursaricam/store-item-demand-forecasting.git

2. Install the required Python packages:
pip install -r requirements.txt

3. Train the model and make predictions using Jupyter Notebook or Python files.

## Model Design

In this project, a time series forecasting model has been designed using LightGBM, an open-source gradient boosting framework developed by Microsoft. LightGBM is a suitable choice for this kind of problem due to its high performance and fast prediction capabilities on large datasets. Details about model training and hyperparameter tuning are available in the [deman_forecasting.py](./deman_forecasting.py) file.

## Contributions and Feedback

If you would like to contribute to this project or provide feedback, you can open a new issue or participate in existing ones using the "Issues" tab. Additionally, you can fork the repository and submit pull requests with your proposed changes. For communication, you can reach me at ugursrcm[at]gmail[.]com.

---
Please note that the data and descriptions provided above are just examples. You can replace them with information related to your actual project to provide a more comprehensive explanation. This page plays an important role in guiding other users who visit your project and helps them quickly understand the purpose and content. 

Good luck and happy coding!


