# 🚗 Road Accident Severity Prediction Using Machine Learning

## 📌 Overview

Road accidents are one of the leading causes of injuries and fatalities worldwide. Every year, millions of accidents occur due to factors such as poor weather conditions, low visibility, heavy traffic, driver behavior, and road infrastructure issues. Understanding the severity of accidents and predicting their outcomes can help authorities, emergency services, and transportation agencies take proactive measures to reduce risks and improve road safety.

This project presents a Machine Learning-based Road Accident Severity Prediction System that analyzes accident-related factors and predicts the likely severity of an accident. The system is built using the US Accidents Dataset and employs advanced machine learning algorithms to classify accidents into different severity levels.

The project also includes an interactive Streamlit web application that allows users to input accident conditions and obtain real-time severity predictions along with visual analysis.

---

# 🎯 Problem Statement

Road accidents vary significantly in their impact. Some accidents result in minor injuries, while others lead to severe injuries or fatalities.

The challenge is:

> Can we predict how severe an accident will be based on environmental and traffic-related conditions?

By accurately predicting accident severity, authorities can:

* Improve emergency response planning.
* Identify high-risk conditions.
* Enhance road safety measures.
* Support intelligent transportation systems.
* Reduce accident-related casualties.

---

# 🚦 Severity Classes

The model predicts one of the following severity categories:

| Severity Level | Description     |
| -------------- | --------------- |
| 1              | Minor Injury    |
| 2              | Moderate Injury |
| 3              | Serious Injury  |
| 4              | Fatal Accident  |

For machine learning implementation, these labels are internally encoded and predicted by the XGBoost model.

---

# ❓ Why This Project Exists

Road accidents have significant social and economic consequences.

Traditional accident analysis is generally performed after an accident has occurred. Such approaches provide valuable insights but do not help in predicting future accident severity.

With the availability of large-scale traffic datasets and advancements in Artificial Intelligence, predictive analytics can be used to estimate accident severity before detailed investigations are completed.

This project was developed to:

* Utilize real-world accident data.
* Apply machine learning techniques to transportation safety.
* Demonstrate predictive analytics in a practical domain.
* Build a user-friendly accident severity prediction system.
* Assist decision-makers with data-driven insights.

---

# 🌍 Importance of the Project

Accident severity prediction is an important application of Artificial Intelligence because it contributes to:

## Public Safety

Predicting accident severity can help authorities identify dangerous conditions and take preventive measures.

## Emergency Management

Emergency responders can prioritize incidents based on expected severity levels.

## Traffic Management

Transportation agencies can understand which conditions contribute most to severe accidents.

## Smart Cities

Accident prediction systems can be integrated into intelligent transportation infrastructure.

## Data-Driven Decision Making

Governments and organizations can formulate policies based on insights obtained from accident data.

---

# 📊 Dataset Information

This project uses the US Accidents Dataset.

### Dataset Characteristics

* Over 7.7 million accident records.
* Covers 49 states of the United States.
* Collected between 2016 and 2023.
* Contains weather, traffic, location, and environmental information.

### Key Features Used

* Temperature
* Humidity
* Visibility
* Wind Speed
* Hour of Day
* Day of Week
* Month
* Accident Duration
* Rush Hour Indicator
* Night Indicator

These features were selected because they significantly influence road conditions and accident outcomes.

---

# ⚙️ Project Workflow

The project follows a complete Machine Learning pipeline.

## Step 1: Data Collection

Accident records were obtained from the US Accidents Dataset.

## Step 2: Data Cleaning

The dataset contained missing values and inconsistencies.

Cleaning operations included:

* Handling missing values.
* Removing duplicates.
* Filtering required features.
* Correcting data types.

## Step 3: Feature Engineering

Additional features were created from existing data.

Examples:

### Hour Extraction

Time information was transformed into numerical hour values.

### Day of Week

Weekday information was extracted.

### Month Extraction

Seasonal information was incorporated.

### Rush Hour Detection

Rush-hour traffic periods were identified.

### Night Detection

Accidents occurring during nighttime were flagged.

### Accident Duration

Duration was calculated using start and end timestamps.

Feature engineering improved the predictive power of the model.

---

# ⚖️ Handling Class Imbalance

Accident datasets often contain unequal numbers of samples for different severity classes.

Most accidents belong to moderate severity categories, while fatal accidents are relatively rare.

To address this issue:

* SMOTE (Synthetic Minority Oversampling Technique) was applied.
* Minority classes were balanced.
* Model bias was reduced.

This significantly improved prediction performance.

---

# 🤖 Machine Learning Models Used

Several models were evaluated.

## Logistic Regression

Used as a baseline model.

Advantages:

* Simple
* Fast
* Easy to interpret

---

## Random Forest

An ensemble learning technique.

Advantages:

* Handles non-linearity
* Robust to noise
* Good generalization

---

## XGBoost

Selected as the final model.

Advantages:

* High accuracy
* Fast training
* Handles complex relationships
* Widely used in industry

XGBoost produced the best overall performance for accident severity prediction.

---

# 📈 Model Evaluation

The model was evaluated using multiple metrics.

## Accuracy

Measures overall correctness.

## Precision

Measures correctness of positive predictions.

## Recall

Measures ability to identify true cases.

## F1 Score

Balances Precision and Recall.

## Confusion Matrix

Visualizes prediction performance across classes.

These metrics provide a comprehensive understanding of model effectiveness.

---

# 🔍 Feature Importance Analysis

One of the major goals of this project is interpretability.

The trained XGBoost model identifies which factors influence accident severity the most.

Examples include:

* Visibility
* Accident Duration
* Temperature
* Humidity
* Time of Day
* Wind Speed

Feature importance analysis helps explain the reasoning behind predictions.

---

# 💻 Streamlit Web Application

To make the model accessible and interactive, a Streamlit-based web application was developed.

### Features

* User-friendly interface.
* Real-time severity prediction.
* Confidence score visualization.
* Feature importance analysis.
* Attractive UI with custom themes and background images.

Users can input accident conditions and instantly receive predicted severity levels.

---

# 🛠️ Technologies Used

| Category                | Technology                  |
| ----------------------- | --------------------------- |
| Language                | Python                      |
| Data Analysis           | Pandas, NumPy               |
| Visualization           | Matplotlib, Seaborn, Plotly |
| Machine Learning        | Scikit-Learn                |
| Boosting Model          | XGBoost                     |
| Class Balancing         | SMOTE                       |
| Web Application         | Streamlit                   |
| Development Environment | Jupyter Notebook, VS Code   |

---

# 📁 Project Structure

```text
Road_Accident_Severity/
│
├── app/
│   ├── streamlit_app.py
│   └── images/
│       ├── home.jpg
│       └── prediction.jpg
│
├── data/
│   └── raw/
│
├── models/
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│   └── feature_importance.pkl
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation.ipynb
│
└── README.md
```

---

# 🚀 Future Enhancements

The project can be further improved by:

* Incorporating real-time traffic data.
* Using GPS and geospatial analysis.
* Integrating weather APIs.
* Deploying on cloud platforms.
* Implementing deep learning models.
* Creating mobile applications.
* Developing accident hotspot prediction systems.

---

# 🎓 Learning Outcomes

Through this project, the following concepts were explored:

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Machine Learning Classification
* Class Imbalance Handling
* Model Evaluation
* Model Deployment
* Streamlit Application Development
* Data Visualization
* Explainable AI

---

# 🏁 Conclusion

This project demonstrates how Machine Learning can be applied to real-world transportation safety problems. By analyzing environmental and traffic-related factors, the system predicts accident severity and provides meaningful insights into the causes of severe accidents.

The combination of data preprocessing, feature engineering, class balancing, XGBoost modeling, visualization, and Streamlit deployment results in a complete end-to-end machine learning solution. The project highlights the potential of Artificial Intelligence to support safer roads, better emergency response systems, and smarter transportation infrastructure.
