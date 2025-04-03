import numpy as np
import pandas as pd
from scipy import stats

def generate_synthetic_data(n_subjects=200, seed=42):
    """
    Generate synthetic survival data with predictors.
    
    Parameters:
    -----------
    n_subjects : int
        Number of subjects to generate
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the synthetic data
    """
    np.random.seed(seed)
    
    # Generate predictors
    management_types = ['Type_A', 'Type_B', 'Type_C']
    management = np.random.choice(management_types, size=n_subjects)
    levels_from_top = np.random.randint(0, 6, size=n_subjects)
    age = np.random.normal(45, 10, size=n_subjects)
    
    # Convert management type to numeric for easier computation
    management_numeric = pd.Categorical(management).codes
    
    # Create base hazard that depends on predictors
    hazard = (
        0.1 * management_numeric +  # Management type effect
        0.2 * levels_from_top +    # Levels from top effect
        0.01 * age                 # Age effect
    )
    
    # Generate survival times using Weibull distribution
    shape = 2.0
    scale = np.exp(-hazard)
    survival_times = stats.weibull_min.rvs(shape, loc=0, scale=scale)
    
    # Generate censoring times
    censoring_times = stats.weibull_min.rvs(2.0, loc=0, scale=20, size=n_subjects)
    
    # Create observed times and censoring indicators
    observed_times = np.minimum(survival_times, censoring_times)
    censored = (survival_times > censoring_times).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': observed_times,
        'censored': censored,
        'ManagementType': management,
        'LevelsFromTop': levels_from_top,
        'Age': age
    })
    
    # Create dummy variables for ManagementType
    management_dummies = pd.get_dummies(df['ManagementType'], prefix='Mgmt')
    df = pd.concat([df, management_dummies], axis=1)
    
    # Standardize numerical predictors
    df['LevelsFromTop_std'] = (df['LevelsFromTop'] - df['LevelsFromTop'].mean()) / df['LevelsFromTop'].std()
    df['Age_std'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()
    
    return df 