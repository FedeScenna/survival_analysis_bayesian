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
4. Calculate the concordance index (C-statistic)
5. Generate plots of posterior distributions and survival curves

## Model Details

The model uses a Weibull distribution for survival times with:
- Normal priors for regression coefficients
- Half-normal prior for the Weibull shape parameter
- Standardized numerical predictors
- Dummy-coded categorical predictor

## Model Evaluation

The model's performance is evaluated using:
1. **Concordance Index (C-statistic)**: Measures the model's discrimination ability, similar to the area under the ROC curve. It represents the probability that for a randomly selected pair of subjects, the subject who died first had a higher predicted risk. Values range from 0.5 (no discrimination) to 1.0 (perfect discrimination).
2. **Posterior Distributions**: Show the uncertainty in the model parameters.
3. **Survival Curves**: Visualize the predicted survival probabilities for different management types.

## Output

The analysis will generate:
1. A summary of the posterior distributions for all model parameters
2. The concordance index with 95% credible interval
3. Plots of the posterior distributions
4. Survival curves for different management types 