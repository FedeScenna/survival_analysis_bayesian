from data_generator import generate_synthetic_data
from survival_model import (
    fit_survival_model,
    plot_posterior_distributions,
    plot_survival_curves,
    print_model_summary
)

def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    df = generate_synthetic_data(n_subjects=200, seed=42)
    
    # Fit the model
    print("\nFitting Bayesian survival model...")
    trace = fit_survival_model(df)
    
    # Print model summary
    print_model_summary(trace)
    
    # Plot results
    print("\nGenerating plots...")
    plot_posterior_distributions(trace)
    plot_survival_curves(trace, df)

if __name__ == "__main__":
    main() 