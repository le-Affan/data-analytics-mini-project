# Fitness Data Analytics Project

## Dataset

This project uses the Fitbit Fitness Tracker dataset:
https://www.kaggle.com/datasets/arashnic/fitbit

The dataset contains activity, sleep, and health-related metrics collected from Fitbit users, including:

* Steps
* Calories burned
* Activity minutes (various intensities)
* Sedentary time
* Sleep duration

---

## Data Preparation

The dataset was preprocessed as follows:

* Selected relevant files: dailyActivity and sleepDay
* Merged datasets on Id and ActivityDate
* Converted date formats into consistent datetime format
* Handled missing values (SleepMinutes filled with 0)
* Removed duplicates and invalid entries
* Created new features:

  * TotalActiveMinutes
  * ActivityRatio

The final dataset contained 936 rows and 11 features.

---

## Experiment 1: Box Plot Analysis

A box plot was created for Calories to analyze distribution and detect outliers.

### Observations:

* Q1 = 1834, Median = 2144, Q3 = 2794.5
* IQR = 960.5
* Outliers exist on both lower and upper ends

### Interpretation:

The data shows variability in calorie expenditure with several extreme values, indicating differences in user activity levels.

---

## Experiment 2: Linear Regression

A linear regression model was built to predict Calories using TotalActiveMinutes.

### Results:

* R² Score ≈ 0.15
* Weak positive relationship observed

### Interpretation:

Calories burned are influenced by activity, but not solely dependent on it. Other factors such as metabolism and user characteristics also play a role.

---

## Experiment 3: Sampling Techniques

Different sampling techniques were applied:

* Simple Random Sampling
* Systematic Sampling
* Stratified Sampling
* Cluster Sampling

### Results:

* Simple Random and Systematic sampling closely matched population mean
* Stratified sampling showed moderate deviation
* Cluster sampling showed significant deviation

### Interpretation:

Random and systematic sampling are more reliable, while cluster sampling can introduce bias depending on the selected cluster.

---

## Experiment 4: Clustering (K-Means)

K-Means clustering was applied using:

* TotalSteps
* TotalActiveMinutes
* SedentaryMinutes
* SleepMinutes

### Results:

* Optimal clusters: 3 (based on elbow method)
* Clear separation of user groups

### Interpretation:

Users can be categorized into:

* Low activity group
* Medium activity group
* High activity group

This helps in identifying different fitness levels.

---

## Experiment 5: Probability Distribution

The probability distribution of Calories was analyzed.

### Observations:

* Distribution is approximately normal
* Slight right skew observed

### Interpretation:

Most users burn calories around the mean, but some users exhibit higher calorie expenditure, creating a positive skew.

---

## Experiment 6: Statistical Analysis

Statistical measures were computed:

* Mean ≈ 2313
* Variance ≈ 494642
* Standard Deviation ≈ 703
* Skewness ≈ 0.55 (positive)
* Kurtosis ≈ 0.41

### Interpretation:

* Data shows moderate variability
* Positive skew confirms higher-end outliers
* Distribution is close to normal with slight deviation

---

## Conclusion

This project demonstrates:

* Data preprocessing and cleaning
* Exploratory data analysis
* Predictive modeling
* Sampling techniques
* Clustering analysis
* Statistical validation

The analysis reveals meaningful insights into user fitness behavior and demonstrates the application of data science techniques on real-world data.

---
