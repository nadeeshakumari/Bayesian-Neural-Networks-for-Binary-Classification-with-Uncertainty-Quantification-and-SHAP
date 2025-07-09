# Bayesian-Neural-Networks-for-Binary-Classification-with-Uncertainty-Quantification-and-SHAP

This repository implements Bayesian Neural Networks (BNNs) using [Pyro](https://pyro.ai) to model binary classification tasks. The framework incorporates **prior sensitivity analysis**, **epistemic and aleatoric uncertainty estimation**, **feature selection**, and **SHAP-based explainability**.

## 📦 Features

- **Bayesian inference** via MCMC using NUTS (No-U-Turn Sampler)
- Support for multiple **prior distributions**:
  - `Normal(0,1)`
  - `Normal(0,10)`
  - `Laplace(0,1)`
  - `Cauchy(0,1)`
  - `Cauchy(0,2.5)`
  - `Horseshoe(1)` 
- **Feature selection** using Boruta algorithm
- **Class imbalance handling** via SMOTE
- **Interpretability** with SHAP (summary, decision, and waterfall plots)
- **Uncertainty quantification** (predictive mean, epistemic, aleatoric)
- **Visualization** of classification error across posterior samples


## 🚀 Workflow
1. Data Preparation
  -Load CSV file into pandas
  -Split into training/testing sets
  -Apply Boruta for feature selection
  -Use SMOTE to handle class imbalance

2. Bayesian Model Training
- Each BNN has:
  - 1 hidden layer with ReLU activation
  - Priors over all weights and biases
  - Models are defined using Pyro modules and trained via:

- Compute:
  - Predictive mean
  - Epistemic uncertainty (variance of model)
  - Aleatoric uncertainty (expected variance from likelihood)

## 📈 SHAP Explainability
This framework integrates SHAP (SHapley Additive exPlanations) to interpret the predictions of the BNN:

How it Works:
- After model training, we extract the mean of posterior weights to define a deterministic prediction function.
- We use shap.KernelExplainer to estimate the contribution of each input feature to the predicted probability.
- SHAP values quantify the impact of each feature, making the model more transparent and interpretable.

SHAP Visualizations Included:
- summary_plot (dot plot): Feature impact across the test set
- summary_plot (bar plot): Mean absolute SHAP values
- decision_plot: How features collectively drive predictions
- waterfall_plot: Feature contributions for a specific prediction

These plots help identify:
- Which features drive positive or negative predictions
- The global importance of features across the test set
- Local explanations for individual predictions

## 📊 Visualizations
- Predictive Mean Plot	Probability estimates
- Uncertainty Bands	Epistemic & Aleatoric Uncertainty
- SHAP Summary Plot	Feature contribution ranking
- Classification Error Plot	Error per MCMC sample






