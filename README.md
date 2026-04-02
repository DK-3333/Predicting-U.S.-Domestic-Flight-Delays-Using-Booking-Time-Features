# Predicting U.S. Domestic Flight Delays Using Booking-Time Features

This project explores one practical question:

**Can we estimate the risk of a flight arriving late at the time of booking, using only information that is known before departure?**

Using U.S. domestic flight data from **January and February 2024**, this project builds a booking-time delay prediction workflow using **AWS S3, AWS Sagemaker and Tableau**. The final result is a calibrated model that produces a **delay probability**, not just a yes/no label.

## Table of Contents

- [Why This Project Matters](#why-this-project-matters)
- [Project Snapshot](#project-snapshot)
- [The Problem Statement](#the-problem-statement)
- [Dataset and Scope](#dataset-and-scope)
- [Project Pipeline](#project-pipeline)
- [What Was Built](#what-was-built)
- [Modeling Workflow](#modeling-workflow)
- [Final Model Performance](#final-model-performance)
- [Results & Model Behaviour Snapshot](#results--model-behaviour-snapshot)
- [Threshold Tradeoff](#threshold-tradeoff)
- [Classification Report at Threshold 0.25](#classification-report-at-threshold-025)
- [Detailed Resource Benchmarking](#detailed-resource-benchmarking)
- [What the Model Learned](#what-the-model-learned)
- [Visual Insights](#visual-insights)
- [Key Insights From the Data](#key-insights-from-the-data)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Tools and Technologies](#tools-and-technologies)
- [How I Would Explain This Project to a Recruiter](#how-i-would-explain-this-project-to-a-recruiter)
- [Repository Contents](#repository-contents)
- [Final Takeaway](#final-takeaway)


## Why This Project Matters

Most flight delay prediction work uses information that becomes available close to departure, such as weather, airport congestion, or upstream network delays. That may help operations teams, but it is not very useful to a passenger deciding which ticket to buy.

This project takes a different approach. It focuses only on **booking-time features**, such as:

- airline
- origin and destination airports
- scheduled departure and arrival times
- scheduled elapsed time
- distance
- day of week
- weekend indicator

The goal was to build something that could actually help a traveler, product team, or travel platform compare flights based on **structural delay risk** before the trip even begins.

## Project Snapshot

- **Dataset:** U.S. domestic flights, **January 1 to February 29, 2024**
- **Rows used:** about **7.08 million**
- **Target:** `is_late = 1` if arrival delay is **15 minutes or more**, else `0`
- **Best model:** **Calibrated XGBoost model**
- **Core idea:** predict delay risk using only **pre-departure, booking-time information**

## The Problem Statement

Flight delays are common, but travelers usually do not get a flight-specific delay-risk estimate while booking. This project frames that as a binary classification problem:

> Will a flight arrive at least 15 minutes late?

The important constraint is that the model is only allowed to use information available at booking time. That means no weather feeds, no same-day operational signals, and no delay propagation variables.

## Dataset and Scope

The source data comes from the **[Bureau of Transportation Statistics (BTS)](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr) On-Time Performance dataset** for U.S. domestic flights operated during **January and February 2024**.

The raw data included more than **60 variables** per flight. For this project:

- cancelled and diverted flights were filtered out in the final modeling workflow
- leakage-heavy fields were removed
- only booking-time features were retained for modeling

### Final modeling features

**Identifiers and route information**
- `op_unique_carrier`
- `op_carrier_fl_num`
- `origin`
- `dest`
- `carrier_origin_pair`
- `distance`
- `crs_elapsed_time`

**Time-related features**
- `day_of_month`
- `day_of_week`
- `is_weekend`
- `crs_dep_time`
- `dep_hour`
- `dep_hour_bin`
- `crs_arr_time`
- `arr_hour`

**Target**
- `is_late`

## Project Pipeline

The project was built as a cloud-based workflow so it could handle a dataset with more than seven million rows in a practical way. Raw flight data was stored in **Amazon S3**, preprocessing was done through **custom SageMaker notebook pipelines**, cleaned data was written back to **S3**, model training and calibration were performed on the prepared dataset, and the final outputs were explored through **Tableau Desktop**.

![Project Pipeline Workflow](<img width="1075" height="317" alt="image" src="https://github.com/user-attachments/assets/e54e32da-21a8-412f-b9f5-7694abf62229" />)

### Pipeline Overview

- **BTS raw flight data** for January-February 2024 was stored in **Amazon S3**
- **AWS SageMaker notebooks** were used for custom preprocessing instead of a GUI ETL workflow
- preprocessing included:
  - schema checks
  - anomaly checks
  - missingness review
  - categorical cardinality profiling
  - type casting
  - filtering cancelled/diverted flights
  - leakage removal
  - booking-time feature engineering
- the cleaned, model-ready dataset was written back to **S3**
- models were trained and calibrated on the processed data
- the final outputs were visualized in **Tableau Desktop**

## What Was Built

This project was developed as a complete ML workflow, not just a single model notebook.

### 1. Cloud-based preprocessing
The raw flight data was stored in **Amazon S3** and processed in **AWS SageMaker notebooks**. The preprocessing workflow handled:

- schema checks
- type casting
- booking-time feature selection
- target creation
- lightweight profiling and EDA

### 2. Feature engineering
A few simple but useful features were engineered:

- **departure hour** and **arrival hour**
- **weekend indicator**
- **departure hour bins**
- **carrier-origin pair** to capture hub-specific airline behavior

### 3. Model comparison
Five models were evaluated:

- **Dummy Classifier**
- **Logistic Regression**
- **Random Forest**
- **HistGradientBoosting**
- **XGBoost**

### 4. Probability calibration
The final XGBoost model was calibrated using **isotonic regression** so that the output probability is more trustworthy and easier to interpret.

### 5. Visualization layer
The final results were explored and communicated through **Tableau dashboards**.

## Modeling Workflow

The final booking-time modeling pipeline used this split:

- **72%** training
- **8%** calibration
- **20%** test

Categorical variables were encoded using **OrdinalEncoder**, and numerical columns were handled with **median imputation**.

Because the classes were imbalanced, XGBoost used:

- `scale_pos_weight ≈ 3.8`

This helped the model pay more attention to the minority class: delayed flights.

![Train / Calibrate / Test Workflow](<img width="813" height="431" alt="image" src="https://github.com/user-attachments/assets/536adb82-49e9-4055-b08a-581671a398d9" />)

### What this split means

The full dataset was first divided into **80% train/calibration** and **20% test**.  
From that 80%, **10% was held out for calibration**, which becomes **8% of the total dataset**.  
That leaves **72% of the full data** for training the base model.

This setup allowed the project to:

- train the model on a large majority of the data
- keep calibration separate from training
- evaluate final performance on a completely untouched test set

## Final Model Performance

The final selected model was **Calibrated XGBoost**.

### Final performance table

| Model | ROC-AUC | Brier Score | Accuracy at Threshold 0.25 |
|---|---:|---:|---:|
| Uncalibrated XGBoost | 0.6799 | 0.2260 | 0.258 |
| Calibrated XGBoost | 0.6798 | 0.1518 | 0.704 |

### Why calibration mattered

The ranking power of the model stayed almost the same after calibration, but the probability quality improved a lot.

- **Brier Score improved from 0.2260 to 0.1518**
- this made the model's outputs much more reliable as risk estimates

That matters because this project is more about **probability-based decision support** than raw classification accuracy.

## Results & Model Behaviour Snapshot

The below table gives a quick comparison of how the main models behaved in practice.

| Model Name | Calibrated Accuracy | Uncalibrated Accuracy | Fit Time (s) | Avg CPU (%) |
|---|---:|---:|---:|---:|
| Dummy | -- | 79.5% | 0.094 | 71.6% |
| Logistic Regression | 62.2% | 20.6% | 359.38 | 957.31% |
| Random Forest | 70.0% | 67.0% | 243.82 | 962.11% |
| HistGradientBoosting | 70.3% | 70.1% | 19.785 | 897.71% |
| XGBoost | 70.4% | 25.8% | 9.748 | 1013.06% |

### Quick takeaway from this comparison

- **XGBoost** gave the best final calibrated accuracy among the evaluated models
- **HistGradientBoosting** was also strong and very fast
- **Random Forest** performed reasonably well but was more computationally expensive
- **Logistic Regression** was useful as a baseline but could not capture the nonlinear structure as effectively
- the gap between calibrated and uncalibrated performance shows why **probability calibration** was worth doing

## Threshold Tradeoff

The final operating threshold was set to **0.25** after evaluating the precision-recall tradeoff.

| Threshold | Precision | Recall |
|---|---:|---:|
| 0.10 | 0.232 | 0.937 |
| 0.20 | 0.302 | 0.645 |
| **0.25** | **0.341** | **0.476** |
| 0.30 | 0.374 | 0.347 |
| 0.40 | 0.453 | 0.129 |
| 0.50 | 0.516 | 0.031 |

At **0.25**, the model gave a better balance between catching risky flights and avoiding too many false alarms.

## Classification Report at Threshold 0.25

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| On-Time (0) | 0.850 | 0.763 | 0.804 | 1,125,822 |
| Late (1) | 0.341 | 0.476 | 0.397 | 289,994 |
| Accuracy |  |  | 0.704 | 1,415,816 |
| Macro Avg | 0.595 | 0.620 | 0.601 | 1,415,816 |
| Weighted Avg | 0.746 | 0.704 | 0.721 | 1,415,816 |

### Confusion matrix

|  | Predicted On-Time (0) | Predicted Late (1)|
|---|---:|---:|
| **Actual On-Time (0)** | 859,392 (TN) | 266,430 (FP) |
| **Actual Late (1)** | 151,993 (FN) | 138,001 (TP) |

## Detailed Resource Benchmarking

A useful part of this project is that it compares not only predictive performance, but also the computational tradeoffs across models.

| Model | Fit Time (s) | Pred. Time (s) | Avg CPU (%) | Peak CPU (%) | Avg Mem (MB) | Peak Mem (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Dummy (Most Frequent) | 0.028 | 0.005 | 0.0 | 0.0 | 1667.2 | 1667.2 |
| Logistic Regression | 161.111 | 0.044 | 1064.0 | 1089.0 | 1608.4 | 1822.4 |
| Random Forest | 78.188 | 24.171 | 960.9 | 1054.2 | 2010.1 | 2608.1 |
| HistGradientBoosting | 7.500 | 1.277 | 942.4 | 1059.7 | 2951.5 | 3104.2 |
| XGBoost | 9.748 | 0.413 | 1013.6 | 1073.2 | 2700.8 | 2717.9 |

### Main takeaway from benchmarking

- **XGBoost** gave the best balance of predictive quality and speed
- **HistGradientBoosting** trained slightly faster, but XGBoost had stronger final performance and faster inference
- **Random Forest** was expensive at prediction time
- **Logistic Regression** was lightweight at inference but weaker in predictive power
- **Dummy Classifier** showed why plain accuracy is misleading in an imbalanced problem

## What the Model Learned

Even without live operational data, the model still captured useful structural patterns.

### Carrier effects
Some airlines showed clearly different historical late rates. Carrier identity turned out to be one of the strongest sources of signal.

### Time-of-day effects
Morning flights were generally more reliable. Delay risk increased later in the day as disruptions accumulated.

### Route and airport effects
Certain origins, destinations, and carrier-hub combinations carried consistent structural delay patterns.

### Weekend vs weekday signal
Weekend flights showed slightly elevated delay rates, which is why the `is_weekend` feature was included.

## Visual Insights

### 1. Late Rate by Carrier

This chart highlights how much delay risk varies across airlines.

![Late Rate by Carrier](<img width="935" height="492" alt="image" src="https://github.com/user-attachments/assets/3af3e96a-7d24-400f-a785-9c7a0ee25350" />)

### 2. Weekend Late Rate

This view compares delay rates across the days of the week and shows the slight weekend effect.

![Weekend Late Rate](<img width="884" height="468" alt="image" src="https://github.com/user-attachments/assets/a54311e3-b446-4bab-8124-1e85dff3d79d" />)

### 3. Predicted Probabilities by Actual Class

This histogram shows how the calibrated XGBoost model separates on-time and late flights through probability scores.

![Predicted Probabilities by Actual Class](<img width="895" height="475" alt="image" src="https://github.com/user-attachments/assets/0871bae7-99f4-4806-9341-56269af0f149" />)

## Key Insights From the Data

### Late rate by carrier
A few examples from the report:

| Carrier | Late Rate |
|---|---:|
| Frontier (F9) | 0.280 |
| American (AA) | 0.257 |
| JetBlue (B6) | 0.250 |
| Spirit (NK) | 0.234 |
| Alaska (AS) | 0.216 |
| Southwest (WN) | 0.204 |
| United (UA) | 0.195 |
| SkyWest (OO) | 0.187 |
| Delta (DL) | 0.171 |
| Republic (YX) | 0.138 |

### Late rate by scheduled departure hour
A few sample points:

| Hour | Late Rate |
|---|---:|
| 05:00 | 0.089 |
| 08:00 | 0.137 |
| 11:00 | 0.175 |
| 14:00 | 0.230 |
| 17:00 | 0.273 |
| 20:00 | 0.298 |

This pattern supports the idea that delays build up through the day.

## Limitations

This project was intentionally constrained, which is also what makes it interesting.

- it does **not** use weather, ATC restrictions, or network propagation signals
- it focuses only on **January and February 2024**
- it emphasizes **booking-time prediction**, which is a lower-signal problem than same-day operational prediction
- feature interpretation was kept qualitative rather than using SHAP or other large-scale attribution tooling

So the model is not trying to predict every operational detail. It is trying to estimate **structural delay risk** at the time of booking.

## Future Improvements

If this project were extended further, the next steps would be:

- add weather and air traffic data
- expand the training pipeline beyond January and February
- package the model into a real-time inference service
- add feature attribution and model explainability
- compare against LightGBM and CatBoost

## Tools and Technologies

- **Python**
- **Pandas, NumPy, scikit-learn**
- **XGBoost**
- **AWS S3**
- **AWS SageMaker**
- **Tableau**

## Repository Contents

- `notebooks/` for preprocessing and training notebooks
- `data_profiling/` for profiling outputs and summary EDA
- `artifacts/` for selected model outputs and benchmark summaries
- `visualizations/` for Tableau Work
- `docs/` for the final report, presentation, and exported visuals

## Final Takeaway

This project shows that even without real-time operational data, flight delay risk can still be estimated in a meaningful way using booking-time features alone. The final calibrated XGBoost model does not claim perfect prediction, but it does provide a practical and interpretable estimate of structural delay risk, which is exactly the kind of output that is useful at the time a traveler is making a decision.

