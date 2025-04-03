import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

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