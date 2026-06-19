# Auto Insurance Ratemaking Model (GLM)

**📊 [View the Interactive Tableau Dashboard Here](https://public.tableau.com/views/Auto-Insurance-Ratemaking-GLM/Sheet1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)**

## Overview
This project is a predictive ratemaking model that calculates the expected pure premium for a simulated portfolio of 10,000 auto insurance policies. It uses Generalized Linear Models (GLMs) to model claim frequency and claim severity, allowing for an analysis of risk factors such as driver age, vehicle value, and annual mileage.

The goal of this project is to demonstrate raw data simulation and statistical modeling in Python and visualization in Tableau.

## Methodology
The model calculates the expected pure premium using a frequency-severity approach:
`Pure Premium = Expected Claim Frequency × Expected Claim Severity`

### 1. Frequency Model
* **Distribution:** Poisson
* **Link Function:** Log
* **Features:** Age Segment, Miles Driven
* **Rationale:** The Poisson distribution models discrete count data (number of claims over a policy period).

### 2. Severity Model
* **Distribution:** Gamma
* **Link Function:** Log
* **Features:** Vehicle Value
* **Rationale:** The Gamma distribution is chosen because of the right-skewed, strictly positive nature of insurance claim severity, where most claims are small but catastrophic losses occur.

## Tech Stack
* **Data Processing & Simulation:** Python (Pandas, NumPy)
* **Predictive Modeling:** Python (Statsmodels, Scikit-learn)
* **Data Visualization:** Tableau, Matplotlib

## Key Findings
* The model captures the non-linear risk associated with young drivers. Drivers under the age of 25 have significantly higher expected claim frequencies, resulting in a steep vertical premium curve that flattens as drivers age into their 30s and 40s.
* By running the outputs through Tableau, the dashboard shows that high premiums in older demographics are driven primarily by high-mileage exposure rather than age-based risk.

## How to Run
1. Clone the repository: `git clone https://github.com/Andre142/Auto-Insurance-Ratemaking-GLM.git`
2. Install required packages
3. Execute the model: `python ratemaking_model.py`
