# Bayesian Survival Analysis

This project implements a Bayesian survival analysis model using synthetic data. The model takes into account:
- One categorical variable (ManagementType)
- Two numerical variables (LevelsFromTop and Age)

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `data_generator.py`: Contains functions to generate synthetic survival data
- `survival_model.py`: Implements the Bayesian Weibull survival model
- `main.py`: Main script to run the analysis
- `requirements.txt`: List of Python dependencies

## Running the Analysis

To run the analysis, simply execute:
```bash
python main.py
```

This will:
1. Generate synthetic data
2. Fit the Bayesian survival model
3. Print model summary statistics
4. Generate plots of posterior distributions and survival curves

## Model Details

The model uses a Weibull distribution for survival times with:
- Normal priors for regression coefficients
- Half-normal prior for the Weibull shape parameter
- Standardized numerical predictors
- Dummy-coded categorical predictor

## Output

The analysis will generate:
1. A summary of the posterior distributions for all model parameters
2. Plots of the posterior distributions
3. Survival curves for different management types 