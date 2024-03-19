# ML PROJECT - INSURANCE CROSS SELL PREDICTION

## Project Summary:

This machine learning classification project aimed to develop a predictive model to classify instances into binary categories of whether the customer is interested (1) or not interested (0) in purchasing a vehicle insurance policy from the company, based on different features. The project involved several stages, including data loading, preprocessing, exploratory data analysis (EDA), hypothesis testing, feature engineering, model training, evaluation, and selection.

The first step was to **load the dataset** and **handle missing values**. Additionally, we checked for and removed any **duplicate values**, ensuring data integrity and consistency throughout the analysis. **Nominal features were converted into distinct categories** to prepare the data for further analysis.

Extensive **exploratory data analysis (EDA)** was conducted to gain insights into the dataset and understand its characteristics. Through visualization and statistical analysis, important patterns, trends, and relationships within the data were identified. Business insights and recommendations were derived from the EDA, providing valuable information for decision-making.

**Hypothesis testing** was performed using chi-square test for uniform distribution, Shapiro-Wilk test for normality, and t-test for equality of means. These tests helped validate assumptions and guide further analysis.

Feature engineering and data preprocessing techniques were applied to enhance the quality of the data and improve model performance. **Outliers** were handled using the interquartile range **(IQR) method**, and **categorical encoding** was performed using **one-hot encoding**. **Feature selection** was carried out using **Kendall correlation** for numeric features and **Mutual Information** for categorical features.

**Data scaling** was performed using **min-max scaler** to ensure all features were on a similar scale, facilitating better convergence during model training. **Dimensionality reduction** was conducted using **Pearson correlation** to identify highly correlated variables and remove redundancy from the dataset.

**Class imbalance** was addressed using the Synthetic Minority Over-sampling Technique **(SMOTE) oversampler** to create synthetic samples of the minority class, ensuring balanced representation in the dataset.

**Five machine learning models—Decision Tree classifier, Logistic Regression, K-Nearest Neighbors (KNN), ADABoost, and RandomForestClassifier**—were trained on the processed data. **Hyperparameter tuning** was performed using **HalvingRandomSearchCV** to optimize model performance.

**Model evaluation** was conducted using various **metrics including accuracy, precision, recall, F1 score, ROC AUC score, and log loss**. **The RandomForestClassifier model** was selected as the best performing model with default parameters. Feature importance analysis revealed the **top three most important features (Vintage, Previously_Insured, Annual_Premium)** driving the classification task.

## Problem Statement:

**The company, being an insurance entity, aims to predict whether its health insurance policyholders from the previous year would also express interest in the Vehicle Insurance provided by the company.**  

By leveraging predictive modeling techniques, the company seeks to optimize its communication strategy, effectively reach out to potential customers, and enhance its revenue stream. The successful development of a predictive model will enable the company to tailor its marketing efforts, identify potential customers for Vehicle Insurance, and strategically plan its business operations.

## Data Description:

**ID:** (Continuous) - Unique identifier for the Customer.

**Age:** (Continuous) - Age of the Customer.

**Gender:** (Dichotomous) - Gender of the Customer.

**Driving_License:** (Dichotomous) - 0 for customer not having DL, 1 for customer having DL.

**Region_Code:** (Nominal) - Unique code for the region of the customer.

**Previously_Insured:** (Dichotomous) - 0 for customer not having vehicle insurance, 1 for customer having vehicle insurance.

**Vehicle_Age:** (Nominal) - Age of the vehicle.

**Vehicle_Damage:** (Dichotomous) - Customer got his/her vehicle damaged in the past. 0 : Customer didn't get his/her vehicle damaged in the past.

**Annual_Premium:** (Continuous) - The amount customer needs to pay as premium in the year.

**Policy_Sales_Channel:** (Nominal) - Anonymized Code for the channel of outreaching to the customer ie. Different Agents, Over Mail, Over Phone, In Person, etc.

**Vintage:** (Continuous) - Number of Days, Customer has been associated with the company.

**Response (Dependent Feature):** (Dichotomous) - 1 for Customer is interested, 0 for Customer is not interested.

## Challenges Faced:

* Handling Large dataset.
* Already available methods of hyper-parameter tuning were taking a huge amount of time to process.
* Memory Optimization during hyperparameter tuning.

## Conclusion:

Based on the highest average score (84%), "Random Forest Classifier" was chosen as the best model. Even though the time taken by Random Forest Classifier was slightly higher than other models like Decision Tree Classifier and KNN which had a similar average score, we chose Random Forest Classifier as it is an ensemble method, is robust, handles high-dimensional data well and is less prone to overfitting. This model was implemented with its default parameters as the hyperparameter tuning did not significantly improve the model performance. 

