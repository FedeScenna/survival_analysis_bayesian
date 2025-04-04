import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def fit_survival_model(df, draws=2000, tune=1000):
    """
    Fit Bayesian Weibull survival model.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    draws : int
        Number of posterior draws
    tune : int
        Number of tuning steps
        
    Returns:
    --------
    arviz.InferenceData
        Posterior samples and model information
    """
    with pm.Model() as survival_model:
        # Priors for coefficients
        mgmt_coef = pm.Normal('mgmt_coef', mu=0, sigma=1, shape=2)  # For 2 dummy variables
        levels_coef = pm.Normal('levels_coef', mu=0, sigma=1)
        age_coef = pm.Normal('age_coef', mu=0, sigma=1)
        
        # Prior for Weibull shape parameter
        shape = pm.HalfNormal('shape', sigma=1)
        
        # Linear predictor
        linear_pred = (
            mgmt_coef[0] * df['Mgmt_Type_B'].values +
            mgmt_coef[1] * df['Mgmt_Type_C'].values +
            levels_coef * df['LevelsFromTop_std'].values +
            age_coef * df['Age_std'].values
        )
        
        # Scale parameter
        scale = pm.Deterministic('scale', pm.math.exp(linear_pred))
        
        # Likelihood
        likelihood = pm.Weibull('likelihood',
                               alpha=shape,
                               beta=scale,
                               observed=df['time'].values)
        
        # Sampling
        trace = pm.sample(draws=draws, tune=tune, return_inferencedata=True)
    
    return trace

def plot_posterior_distributions(trace):
    """Plot posterior distributions of model parameters."""
    az.plot_posterior(trace)
    plt.tight_layout()
    plt.show()

def plot_survival_curves(trace, df):
    """Plot survival curves for different management types."""
    times = np.linspace(0, df['time'].max(), 100)
    management_types = ['Type_A', 'Type_B', 'Type_C']
    
    # Get mean posterior values
    shape_post = trace.posterior['shape'].mean()
    mgmt_coef_post = trace.posterior['mgmt_coef'].mean(dim=['chain', 'draw'])
    levels_coef_post = trace.posterior['levels_coef'].mean()
    age_coef_post = trace.posterior['age_coef'].mean()
    
    plt.figure(figsize=(10, 6))
    
    for mgmt_type in management_types:
        # Create predictor values
        if mgmt_type == 'Type_A':
            mgmt_effect = 0
        elif mgmt_type == 'Type_B':
            mgmt_effect = mgmt_coef_post[0]
        else:
            mgmt_effect = mgmt_coef_post[1]
            
        # Use mean values for other predictors
        linear_pred = (mgmt_effect +
                      levels_coef_post * 0 +  # standardized mean = 0
                      age_coef_post * 0)      # standardized mean = 0
        
        scale = np.exp(linear_pred)
        survival = np.exp(-(times/scale)**shape_post)
        
        plt.plot(times, survival, label=mgmt_type)
    
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title('Survival Curves by Management Type')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_model_summary(trace):
    """Print summary of the posterior distributions."""
    summary = az.summary(trace)
    print("\nModel Summary:")
    print(summary)

def calculate_concordance_index(trace, df):
    """
    Calculate the concordance index (C-statistic) for the survival model.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        Posterior samples from the fitted model
    df : pd.DataFrame
        DataFrame containing the data
        
    Returns:
    --------
    float
        Concordance index value
    dict
        Dictionary with concordance index for each posterior sample
    """
    # Get posterior samples
    shape_samples = trace.posterior['shape'].values
    mgmt_coef_samples = trace.posterior['mgmt_coef'].values
    levels_coef_samples = trace.posterior['levels_coef'].values
    age_coef_samples = trace.posterior['age_coef'].values
    
    # Number of samples and chains
    n_samples = shape_samples.shape[0]
    n_chains = shape_samples.shape[1]
    
    # Initialize array to store concordance indices
    concordance_indices = np.zeros((n_samples, n_chains))
    
    # Calculate linear predictor for each sample
    for i in range(n_samples):
        for j in range(n_chains):
            # Get coefficients for this sample
            shape = shape_samples[i, j]
            mgmt_coef = mgmt_coef_samples[i, j]
            levels_coef = levels_coef_samples[i, j]
            age_coef = age_coef_samples[i, j]
            
            # Calculate linear predictor
            linear_pred = (
                mgmt_coef[0] * df['Mgmt_Type_B'].values +
                mgmt_coef[1] * df['Mgmt_Type_C'].values +
                levels_coef * df['LevelsFromTop_std'].values +
                age_coef * df['Age_std'].values
            )
            
            # Calculate scale parameter
            scale = np.exp(linear_pred)
            
            # Calculate hazard for each subject
            hazard = (shape / scale) * (df['time'].values / scale) ** (shape - 1)
            
            # Calculate concordance index
            concordance_indices[i, j] = _compute_concordance(hazard, df['time'].values, df['censored'].values)
    
    # Calculate mean concordance index
    mean_concordance = np.mean(concordance_indices)
    
    # Calculate 95% credible interval
    ci_lower = np.percentile(concordance_indices.flatten(), 2.5)
    ci_upper = np.percentile(concordance_indices.flatten(), 97.5)
    
    # Create dictionary with results
    results = {
        'mean_concordance': mean_concordance,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'all_samples': concordance_indices.flatten()
    }
    
    return mean_concordance, results

def _compute_concordance(hazard, time, censored):
    """
    Compute concordance index for a single set of hazard values.
    
    Parameters:
    -----------
    hazard : array-like
        Hazard values for each subject
    time : array-like
        Observed times
    censored : array-like
        Censoring indicators (1=censored, 0=event)
        
    Returns:
    --------
    float
        Concordance index
    """
    n = len(time)
    concordant = 0
    comparable = 0
    
    for i in range(n):
        for j in range(i+1, n):
            # Skip if both subjects are censored
            if censored[i] and censored[j]:
                continue
                
            # Determine which subject had the event first
            if time[i] < time[j]:
                if not censored[i]:  # i had event first
                    if hazard[i] > hazard[j]:
                        concordant += 1
                    comparable += 1
            elif time[j] < time[i]:
                if not censored[j]:  # j had event first
                    if hazard[j] > hazard[i]:
                        concordant += 1
                    comparable += 1
            else:  # same time
                if not censored[i] and not censored[j]:  # both had events
                    if hazard[i] > hazard[j]:
                        concordant += 0.5
                    elif hazard[i] < hazard[j]:
                        concordant += 0.5
                    comparable += 1
    
    if comparable == 0:
        return 0.5  # Default value if no comparable pairs
    
    return concordant / comparable 