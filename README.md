# Project 7: Medical Appointment No-Shows Prediction

## Table of Contents

1.  [Introduction](https://www.google.com/search?q=%231-introduction)
2.  [Dataset](https://www.google.com/search?q=%232-dataset)
3.  [Problem Formulation](https://www.google.com/search?q=%233-problem-formulation)
4.  [Key Features](https://www.google.com/search?q=%234-key-features)
5.  [Machine Learning Approach](https://www.google.com/search?q=%235-machine-learning-approach)
6.  [Project Process](https://www.google.com/search?q=%236-project-process)
      * [6.1. Data Loading & Initial Inspection](https://www.google.com/search?q=%2361-data-loading--initial-inspection)
      * [6.2. Exploratory Data Analysis (EDA) & Feature Engineering](https://www.google.com/search?q=%2362-exploratory-data-analysis-eda--feature-engineering)
      * [6.3. Data Preprocessing](https://www.google.com/search?q=%2363-data-preprocessing)
      * [6.4. Model Training & Evaluation](https://www.google.com/search?q=%2364-model-training--evaluation)
      * [6.5. Hyperparameter Tuning](https://www.google.com/search?q=%2365-hyperparameter-tuning)
7.  [Key Findings & Feature Importance](https://www.google.com/search?q=%237-key-findings--feature-importance)
8.  [Conclusion & Recommendations](https://www.google.com/search?q=%238-conclusion--recommendations)
9.  [Future Work](https://www.google.com/search?q=%239-future-work)
10. [How to Run (Google Colab)](https://www.google.com/search?q=%2310-how-to-run-google-colab)

-----

## 1\. Introduction

This project aims to build a predictive model to identify patients who are likely to miss their scheduled medical appointments. Medical appointment no-shows are a significant challenge for healthcare providers, leading to wasted resources, decreased efficiency, and reduced patient access to care. By accurately predicting no-shows, clinics can implement proactive strategies to manage their schedules better, reduce patient wait times, and optimize resource allocation.

## 2\. Dataset

  * **Name:** Medical Appointment No Shows
  * **Source:** Kaggle ([Link to Dataset](https://www.kaggle.com/datasets/joniarroba/noshowappointments))
  * **Description:** The dataset contains a large collection of medical appointment records from Brazil, detailing various patient characteristics, appointment specifics, and whether the patient attended or missed their scheduled visit.

## 3\. Problem Formulation

The problem is formulated as a **binary classification** task: predicting whether a patient will "show" or "no-show" for their appointment.

## 4\. Key Features

The dataset includes features such as:

  * `PatientId`, `AppointmentID` (identifiers, dropped after initial use)
  * `Gender`
  * `ScheduledDay` (when the appointment was scheduled)
  * `AppointmentDay` (the actual appointment date)
  * `Age`
  * `Neighbourhood` (location of the hospital)
  * `Scholarship` (whether the patient is enrolled in a welfare program)
  * `Hypertension`, `Diabetes`, `Alcoholism`, `Handicap` (patient's health conditions)
  * `SMS_received` (whether the patient received an SMS reminder)
  * `No-show` (the target variable: 'Yes' for no-show, 'No' for show)

**Engineered Features:**

  * `day_of_week_scheduled`: Day of the week the appointment was scheduled.
  * `day_of_week_appointment`: Day of the week of the appointment.
  * `waiting_time_days`: Number of days between scheduling and appointment.

## 5\. Machine Learning Approach

Given the binary classification nature, we employed various supervised learning algorithms:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Gradient Boosting (LightGBM)
  * Gradient Boosting (CatBoost)

**Evaluation Metrics:**
Due to the significant class imbalance (approx. 80% shows, 20% no-shows), primary evaluation focused on metrics relevant to the minority class ("no-show"):

  * **F1-Score:** Harmonic mean of Precision and Recall.
  * **Recall (Sensitivity):** Ability to correctly identify all actual no-shows.
  * **Precision:** Proportion of positive identifications that were actually correct.
  * **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):** Measures the model's ability to distinguish between classes.
  * Accuracy and Confusion Matrix were also monitored.

## 6\. Project Process

The project followed a structured machine learning workflow:

### 6.1. Data Loading & Initial Inspection

The `KaggleV2-May-2016.csv` dataset was loaded into a Pandas DataFrame. Initial checks confirmed no missing values in the dataset, but a critical error was identified in the `Age` column where one entry had a value of -1.

### 6.2. Exploratory Data Analysis (EDA) & Feature Engineering

  * **Column Renaming:** Columns were standardized to lowercase with underscores (e.g., `ScheduledDay` to `scheduled_day`). `Hipertension` and `Handcap` typos were corrected.
  * **Datetime Conversion:** `ScheduledDay` and `AppointmentDay` were converted to `datetime` objects.
  * **Feature Creation:**
      * `day_of_week_scheduled` and `day_of_week_appointment` were extracted.
      * `waiting_time_days` was calculated as the difference between `AppointmentDay` and `ScheduledDay`. Negative waiting times (indicating same-day or incorrectly logged appointments) were set to 0.
  * **Target Encoding:** The `No-show` column ('Yes'/'No') was converted into a numerical binary `no_show_numeric` (1 for 'Yes', 0 for 'No').
  * **Class Imbalance:** EDA confirmed a class imbalance: \~79.8% showed up (class 0) and \~20.2% no-showed (class 1). This informed our choice of evaluation metrics and modeling strategies (e.g., `class_weight='balanced'`).
  * **Key Observations from EDA:**
      * Longer `waiting_time_days` correlated with higher no-show rates.
      * Certain `day_of_week_appointment` (e.g., Saturday, Friday) showed higher no-show rates.
      * Patients receiving `SMS_received` reminders paradoxically had a higher no-show rate, suggesting either poor reminder effectiveness or targeted reminders to already high-risk patients.
      * `Scholarship` recipients and certain `Neighbourhood`s showed slightly different no-show propensities.

### 6.3. Data Preprocessing

  * **Invalid Age Handling:** The single row with `Age = -1` was removed.
  * **Irrelevant Feature Removal:** `PatientId`, `AppointmentID`, `ScheduledDay`, and `AppointmentDay` were dropped as their direct values are not predictive after new features were extracted.
  * **Feature Definition:** Categorical features (`gender`, `neighbourhood`, `day_of_week_scheduled`, `day_of_week_appointment`) and numerical features (`age`, `waiting_time_days`, `scholarship`, `hypertension`, `diabetes`, `alcoholism`, `handicap`, `sms_received`) were identified.
  * **Train-Test Split:** The dataset was split into 80% training and 20% testing sets using `stratify=y` to maintain the class distribution in both sets.
  * **Preprocessor Pipeline:** A `ColumnTransformer` was set up within pipelines to:
      * Apply `StandardScaler` to numerical features (for models sensitive to scale).
      * Apply `OneHotEncoder` to categorical features (to convert them into a numerical format).

### 6.4. Model Training & Evaluation

We trained Logistic Regression, Decision Tree, Random Forest, LightGBM, and CatBoost models. Each model was wrapped in a `Pipeline` to ensure preprocessing steps were applied consistently. `class_weight='balanced'` (or equivalent for CatBoost) was used to mitigate the class imbalance.

| Model                      | Accuracy | Precision (No-Show) | Recall (No-Show) | F1-Score (No-Show) | ROC-AUC |
| :------------------------- | :------- | :------------------ | :--------------- | :----------------- | :------ |
| Logistic Regression        | 0.6544   | 0.3104              | 0.5822           | 0.4049             | 0.6657  |
| Decision Tree              | 0.7135   | 0.3203              | 0.3730           | 0.3446             | 0.5880  |
| Random Forest              | 0.7695   | 0.3741              | 0.2103           | 0.2693             | 0.7107  |
| Gradient Boosting (LightGBM) | 0.6039   | 0.3129              | **0.8044** | **0.4506** | **0.7380**|
| Gradient Boosting (CatBoost) | 0.6259   | 0.3211              | 0.7652           | **0.4524** | **0.7436**|

**Initial Model Performance Analysis:**
Gradient Boosting models (CatBoost and LightGBM) emerged as the best performers, achieving the highest F1-Scores and ROC-AUC for the "No-Show" class. They demonstrated superior Recall, indicating their effectiveness in identifying actual no-shows.

### 6.5. Hyperparameter Tuning

Hyperparameter tuning was performed on the best-performing model, **CatBoost**, using `GridSearchCV` to optimize its performance, specifically targeting the F1-Score of the "No-Show" class.

**Tuned CatBoost Performance on Test Set:**

  * **Best Parameters:** `{'classifier__depth': 6, 'classifier__iterations': 200, 'classifier__l2_leaf_reg': 1, 'classifier__learning_rate': 0.1}`
  * **F1-Score (Cross-Validation):** 0.4499
  * **Test Set Metrics:**
      * **Accuracy:** 0.6064
      * **Precision (No-Show):** 0.3147
      * **Recall (No-Show):** **0.8060**
      * **F1-Score (No-Show):** **0.4527**
      * **ROC-AUC:** 0.7397

The tuning slightly improved the F1-Score and significantly increased Recall for the "No-Show" class, making the model even better at detecting potential no-shows, though with a slight trade-off in precision.

## 7\. Key Findings & Feature Importance

The tuned CatBoost model revealed the most influential factors contributing to appointment no-shows:

| Feature                   | Importance (%) | Insight                                                                                                 |
| :------------------------ | :------------- | :------------------------------------------------------------------------------------------------------ |
| `waiting_time_days`       | 35.95          | The single most critical factor; longer waits significantly increase no-show probability.               |
| `age`                     | 16.34          | Patient age plays a substantial role.                                                                   |
| `day_of_week_scheduled`/`appointment` | \~15-20 (combined) | Specific days of the week (both when scheduled and the appointment itself) influence attendance.        |
| `sms_received`            | 1.90           | Counter-intuitive: receiving an SMS correlates with *higher* no-show rates. Requires further investigation. |
| `neighbourhood_SANTOS DUMONT`| 1.43           | Specific neighborhoods can be strong indicators, suggesting localized socio-economic or access barriers.|
| `scholarship`             | 1.29           | Patients on welfare programs show a slightly higher tendency to no-show.                               |

## 8\. Conclusion & Recommendations

This project successfully developed and evaluated machine learning models for predicting medical appointment no-shows. The **CatBoost Classifier** demonstrated the best performance, effectively balancing the identification of actual no-shows (high Recall) with overall accuracy, achieving an F1-Score of **0.4527** for the minority "No-Show" class.

**Actionable Recommendations for Clinics:**

1.  **Prioritize Shortening Waiting Times:** The most impactful intervention would be to reduce the time between scheduling and appointment, as this is the primary driver of no-shows. Clinics should review their scheduling practices to minimize wait durations.
2.  **Optimize Scheduling by Day of Week:** Clinics could implement more aggressive reminder systems or specific scheduling adjustments for appointments on Fridays and Saturdays, which show higher no-show rates.
3.  **Rethink SMS Reminder Strategy:** The paradoxical correlation between SMS reminders and no-shows warrants deeper investigation. Clinics should:
      * Analyze if SMS is primarily sent to high-risk groups (and thus reflects existing risk, not an effective deterrent).
      * Experiment with different reminder timings (e.g., multiple reminders, closer to the appointment time).
      * Consider interactive reminders (e.g., "Reply Y to confirm") or supplementing SMS with phone calls for high-risk patients.
4.  **Implement Targeted Interventions:** Leverage the predictive model to identify patients with a high likelihood of no-showing. For these high-risk individuals, clinics can apply more intensive and personalized interventions, such as direct phone calls, offering flexible rescheduling options, or providing detailed pre-appointment information.
5.  **Address Neighborhood-Specific Issues:** For neighborhoods identified as having high no-show rates (e.g., SANTOS DUMONT), clinics might consider community outreach programs, transportation assistance, or local partnerships to address underlying barriers to attendance.

## 9\. Future Work

  * **Expanded Hyperparameter Tuning:** Explore a wider range of hyperparameters and more advanced tuning techniques (e.g., Bayesian Optimization).
  * **Advanced Feature Engineering:** Investigate interaction terms between features or time-series analysis for patterns in scheduling/appointment times.
  * **Cost-Benefit Analysis:** Integrate the model's predictions into a real-world cost analysis to quantify the financial impact of no-shows and the return on investment of implementing interventions.
  * **Deployment:** Develop a system for integrating the model into a clinic's existing patient management software for real-time predictions.

## 10\. How to Run (Google Colab)

1.  **Open in Google Colab:** Upload the `.ipynb` notebook file to Google Colab.
2.  **Upload Dataset:** Download the `KaggleV2-May-2016.csv` dataset from the Kaggle link provided above and upload it to your Colab environment (e.g., drag and drop into the files pane, or mount Google Drive).
3.  **Install Libraries:** Ensure all necessary libraries (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `lightgbm`, `catboost`) are installed. Colab typically has most of these pre-installed, but a `!pip install ...` command might be needed for some.
4.  **Run Cells Sequentially:** Execute the notebook cells sequentially from top to bottom. Each step (Data Loading, EDA, Preprocessing, Training, Tuning) is presented in distinct code blocks.

-----
