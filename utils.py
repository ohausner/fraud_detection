
# ===========================================================================
#                                  IMPORTS
# ===========================================================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from IPython.display import display, Markdown
import itertools

import shap

# ===========================================================================
#                            EDA Functions Utils
# ===========================================================================

def analyze_continuous_feature(df: pd.DataFrame, feature: str, target: str = 'Class', outlier_factor: float = 1.5):
    """
    Analyzes a continuous feature against a binary target using visual and statistical methods.
    """
    # --- Data Preparation ---
    # Drop nulls for this specific analysis to avoid errors
    data = df[[feature, target]].dropna()
    
    group0 = data[data[target] == 0][feature] # Non-Fraud
    group1 = data[data[target] == 1][feature] # Fraud

    # --- 1. Visualization (3 Horizontal Plots) ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # Plot A: Overall Density
    sns.kdeplot(data=data, x=feature, fill=True, ax=axes[0], color='gray')
    axes[0].set_title(f'Overall Density: {feature}')
    
    # Plot B: Density by Target
    # common_norm=False is CRITICAL here because classes are imbalanced. 
    # If True, the Fraud curve would be invisible.
    sns.kdeplot(data=data, x=feature, hue=target, common_norm=False, fill=True, ax=axes[1], palette='coolwarm')
    axes[1].set_title(f'Density by {target}')
    # Add vertical lines for means
    axes[1].axvline(group0.mean(), color='blue', linestyle='--', label=f'Mean Class 0 {group0.mean():.2f}')
    axes[1].axvline(group1.mean(), color='red', linestyle='--', label=f'Mean Class 1 {group1.mean():.2f}')
    axes[1].legend()
    
    # Plot C: Boxplot
    sns.boxplot(data=data, x=target, y=feature, hue=target, ax=axes[2], palette='coolwarm')
    axes[2].set_title(f'Boxplot by {target}')
    
    plt.tight_layout()
    plt.show()

    # --- 2. Statistical Tests (T-Test & Effect Size) ---
    # t_stat, p_val = stats.ttest_ind(group0, group1, equal_var=False) # Welch's t-test
    # | **T-Test P-Value** | `{p_val:.4e}` | {'Significant difference' if p_val < 0.05 else 'No significant difference'} |

    # Calculate Cohen's D (Effect Size)
    n0, n1 = len(group0), len(group1)
    var0, var1 = np.var(group0, ddof=1), np.var(group1, ddof=1)
    pooled_se = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2))
    cohens_d = (np.mean(group0) - np.mean(group1)) / pooled_se

    # --- 3. Outlier Analysis ---
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (outlier_factor * IQR)
    upper_bound = Q3 + (outlier_factor * IQR)
    
    outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
    outlier_pct = (len(outliers) / len(data)) * 100

    # --- 4. Render Markdown Report ---
    md_report = f"""
### 📉 Analysis for `{feature}`
| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Cohen's D** | `{abs(cohens_d):.4f}` | {'Large Effect' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small/Negligible'} |
| **Outliers (IQR={outlier_factor})** | `{len(outliers):,}` ({outlier_pct:.2f}%) | Range: [{lower_bound:.2f}, {upper_bound:.2f}] |
    """
    display(Markdown(md_report))

    return {
        'cohens_d': abs(cohens_d),
        'outlier_count': len(outliers),
        'outlier_percentage': outlier_pct}

def table_general_analyses(df: pd.DataFrame) -> pd.DataFrame:

    # General DataFrame Information
    print("="*40)
    print(f"📊 DataFrame Overview: {df.shape[0]} Rows, {df.shape[1]} Columns")
    print("="*40)

    # Overall Duplications Check
    total_duplicates = df.duplicated().sum()
    print(f"Total fully duplicated rows (excluding first occurrence): {total_duplicates}")
    print(f"Percentage of duplicates: {total_duplicates / df.shape[0] * 100:.2f}%")
    print("-" * 40)

    # Missing Values Analysis
    print(" Missing Values Analysis")
    missing_summary = df.isna().sum().to_frame(name='Missing Values')
    missing_summary['% Missing'] = (missing_summary['Missing Values'] / df.shape[0]) * 100
    if len(missing_summary[missing_summary['Missing Values'] > 0]) == 0:
        print("No missing values detected in the dataset.")
    else:
        display(missing_summary[missing_summary['Missing Values'] > 0])

    
def scatter_plot_feature_vs_target(df: pd.DataFrame, feature_1: str, features_2: str, target: str = 'Class'):
    """
    Creates a scatter plot of two features colored by the binary target.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=feature_1, y=features_2, hue=target, palette='coolwarm', alpha=0.6)
    plt.title(f'Scatter Plot of {feature_1} vs {features_2} colored by {target}')
    plt.show()

#ratio plot by hour of day
def plot_fraud_ratio_by_day(df: pd.DataFrame, day_col: str = 'hour_of_day', target_col: str = 'Class'):
    daily_counts = df.groupby(day_col)[target_col].value_counts().unstack(fill_value=0)
    daily_counts['Fraud_Ratio'] = daily_counts[1] / (daily_counts[0] + daily_counts[1])
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily_counts, x=daily_counts.index, y='Fraud_Ratio', marker='o')
    plt.title('Daily Fraud Ratio Over Time')
    plt.ylabel('Fraud Ratio')
    plt.xlabel('Hour of day')
    plt.ylim(0, daily_counts['Fraud_Ratio'].max() * 1.1)
    plt.grid()
    plt.show()


# ===========================================================================
#                        Evaluation Functions Utils
# ===========================================================================

def evaluate_model(model_name: str, data: pd.DataFrame, features: list, target: str = 'Class', 
                   params=None, class_weight=None, k_folds=10):
    
    # 1. Prepare Data and CV
    X = data[features]
    y = data[target]
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # 2. Define Preprocessor (Scale Amount, pass through others)
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', RobustScaler(), features)
        ]
    )

    # 3. Handle Class Weights & Initialize Model
    if params is None:
        params = {}

    if model_name == 'LogisticRegression':
        clf = LogisticRegression(
            solver='liblinear', 
            max_iter=1000, 
            class_weight=class_weight, 
            **params
        )
    elif model_name == 'RandomForest':
        clf = RandomForestClassifier(
            n_jobs=-1, 
            random_state=42, 
            class_weight=class_weight, 
            **params
        )
    elif model_name == 'XGBoost':
        clf = XGBClassifier(
            # use_label_encoder=False, 
            eval_metric='logloss', 
            random_state=42,
            scale_pos_weight=class_weight,
            **params
        )
    else:
        raise ValueError(f"Model {model_name} not supported.")

    # 4. Storage for results
    fold_metrics = []
    feature_importance_list = []

    print(f"🔄 Starting {k_folds}-Fold CV for {model_name} (Weights: {class_weight})...")

    # 5. Cross Validation Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Create Pipeline (Preprocessor + Classifier)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        # Fit
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1] 

        # Calculate Metrics
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics = {
            'Fold': fold + 1,
            'AUPRC': average_precision_score(y_val, y_proba),
            'AUC': roc_auc_score(y_val, y_proba),
            'Accuracy': accuracy_score(y_val, y_pred),
            'Precision': precision_score(y_val, y_pred, zero_division=0),
            'Recall (Sensitivity)': recall_score(y_val, y_pred),
            'Specificity': specificity
        }
        fold_metrics.append(metrics)

        # Extract Feature Importance / Coefficients
        trained_model = pipeline.named_steps['classifier']
        
        # Get feature names 
        feature_names_out = pipeline.named_steps['preprocessor'].get_feature_names_out()
        clean_names = [name.split('__')[-1] for name in feature_names_out]


        if hasattr(trained_model, 'feature_importances_'):
            # Tree based
            imps = trained_model.feature_importances_
            fold_imp = pd.DataFrame({'Feature': clean_names, 'Importance': imps})
        elif hasattr(trained_model, 'coef_'):
            # Linear based
            imps = trained_model.coef_[0]
            fold_imp = pd.DataFrame({'Feature': clean_names, 'Importance': np.abs(imps)})
        else:
            fold_imp = pd.DataFrame()
            
        feature_importance_list.append(fold_imp)

    # 6. Aggregation
    metrics_df = pd.DataFrame(fold_metrics)
    numeric_df = metrics_df.drop(columns='Fold')
    avg_metrics = numeric_df.mean().to_frame(name='Mean')
    avg_metrics['Std'] = numeric_df.std()
    
    if feature_importance_list:
        all_importances = pd.concat(feature_importance_list)
        avg_importance = all_importances.groupby('Feature')['Importance'].mean().sort_values(ascending=False).reset_index()
    else:
        avg_importance = pd.DataFrame()

    print(f"✅ Completed. Average AUPRC: {avg_metrics.loc['AUPRC', 'Mean']:.4f} (std: {avg_metrics.loc['AUPRC', 'Std']:.4f})")
    
    return {
        'metrics_summary': avg_metrics,
        'metrics_per_fold': metrics_df,
        'feature_importance': avg_importance
    }

def plot_model_comparison(model_results: dict, metric_sort_order=None):
    """
    Plots a grouped bar chart comparing multiple models across various metrics,
    including error bars representing standard deviation.

    Args:
        model_results (dict): A dictionary where keys are Model Names (str) 
                              and values are the results dict returned by `evaluate_model`.
                              Example: {'XGBoost': res_xgb, 'LogReg': res_lr}
        metric_sort_order (list): Optional list of metric names to define the order on X-axis.
    """
    
    # 1. Extract and Structure Data
    records = []
    
    for model_name, result_data in model_results.items():
        # Access the summary DataFrame
        summary = result_data['metrics_summary']
        
        for metric_name, row in summary.iterrows():
            records.append({
                'Model': model_name,
                'Metric': metric_name,
                'Mean': row['Mean'],
                'Std': row['Std']
            })
            
    df_plot = pd.DataFrame(records)
    
    # 2. Setup Plot
    if metric_sort_order:
        metrics = metric_sort_order
    else:
        # Default order (try to put AUPRC first as it's the main KPI)
        metrics = df_plot['Metric'].unique().tolist()
        if 'AUPRC' in metrics:
            metrics.remove('AUPRC')
            metrics.insert(0, 'AUPRC')
            
    models = df_plot['Model'].unique().tolist()
    
    # Colors
    colors = sns.color_palette("viridis", len(models))
    
    # Dimensions
    n_metrics = len(metrics)
    n_models = len(models)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Bar width logic
    total_width = 0.8
    bar_width = total_width / n_models
    x_positions = np.arange(n_metrics)
    
    # 3. Plotting Loop
    for i, model in enumerate(models):
        # Filter data for this model
        model_data = df_plot[df_plot['Model'] == model].set_index('Metric')
        
        # Reindex to ensure order matches x_positions
        model_data = model_data.reindex(metrics)
        
        # Calculate offset
        # i - n_models/2 + 0.5 centers the group around the tick
        offset = (i - n_models / 2 + 0.5) * bar_width
        
        ax.bar(
            x_positions + offset, 
            model_data['Mean'], 
            width=bar_width, 
            yerr=model_data['Std'], 
            label=model, 
            color=colors[i],
            capsize=5,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.9
        )

        # Add text labels on top of bars (optional, but helpful)
        for j, (val_mean, val_std) in enumerate(zip(model_data['Mean'], model_data['Std'])):
            if not np.isnan(val_mean):
                ax.text(
                    x_positions[j] + offset, 
                    val_mean + val_std + 0.02, 
                    f'{val_mean:.2f}', 
                    ha='center', 
                    va='bottom', 
                    fontsize=8, 
                    rotation=0,
                    fontweight='bold'
                )

    # 4. Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_ylabel('Score (Mean ± Std)', fontsize=12)
    ax.set_title('Model Performance Comparison across Metrics', fontsize=16, pad=20)
    ax.set_ylim(0, 1.15) # Leave room for labels
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Legend
    ax.legend(title='Models', fontsize=11, title_fontsize=12, loc='lower right')
    
    plt.tight_layout()
    plt.show()

# ===========================================================================
#                      Optimizing Baseline Functions Utils
# ===========================================================================
def feature_selection_heuristics(train_df, target, initial_results, range_search, class_weight=None):
    
    # 1. Initialization
    # Sort features by importance just in case
    sorted_features_df = initial_results['feature_importance'].sort_values(by='Importance', ascending=False)
    all_ranked_features = sorted_features_df['Feature'].tolist()
    
    # Start with the baseline (all features)
    best_subset = all_ranked_features.copy()
    best_auprc = initial_results['metrics_summary'].loc['AUPRC']['Mean']
    feature_bank = [] # Features currently excluded
    
    print(f"🚀 Starting Heuristic Search. Baseline AUPRC: {best_auprc:.4f}")

    # --- PHASE 1: Range Search (Top K Features) ---
    print("\n🔎 Phase 1: Testing Top-K Subsets...")
    phase_1_improved = False
    
    for k in range_search:
        # Select top k
        candidate_subset = all_ranked_features[:k]
        
        # Evaluate
        res = evaluate_model('XGBoost', train_df, candidate_subset, target=target, class_weight=class_weight, k_folds=10)
        curr_auprc = res['metrics_summary'].loc['AUPRC']['Mean']
        
        print(f"   -> Top {k} Features: AUPRC {curr_auprc:.4f}")
        
        if curr_auprc > best_auprc:
            print(f"      ✅ New Best Found! (Improvement: {curr_auprc - best_auprc:.4f})")
            best_subset = candidate_subset
            best_auprc = curr_auprc
            # The bank is everything NOT in the top k
            feature_bank = all_ranked_features[k:]
            phase_1_improved = True

    if not phase_1_improved:
        print("   -> No Top-K subset beat the baseline. Keeping all features for now.")
        feature_bank = []

    # --- PHASE 2: Backward Elimination (Pruning) ---
    print(f"\n✂️ Phase 2: Pruning (Current size: {len(best_subset)})")
    
    # Iterate over a COPY of the list so we can modify the original safely
    features_to_check = list(best_subset)
    
    for feat in features_to_check:
        # Don't prune if we have too few features left (e.g., keep at least 3)
        if len(best_subset) <= 3:
            break
            
        # Create a temp subset without this feature
        test_subset = [f for f in best_subset if f != feat]
        
        res = evaluate_model('XGBoost', train_df, test_subset, target=target, class_weight=class_weight, k_folds=10)
        curr_auprc = res['metrics_summary'].loc['AUPRC']['Mean']
        
        if curr_auprc > best_auprc:
            print(f"   Removing '{feat}' IMPROVED score to {curr_auprc:.4f}")
            best_subset = test_subset
            best_auprc = curr_auprc
            # Add the rejected feature to the bank (we might want it back later)
            feature_bank.append(feat)
        else:
            # print(f"   Removing '{feat}' hurt score ({curr_auprc:.4f}). Keeping it.")
            pass

    # --- PHASE 3: Forward Selection (Recovery) ---
    print(f"\n➕ Phase 3: Recovery (Checking {len(feature_bank)} excluded features)")
    
    # Sort bank to try most important excluded features first? (Optional, but efficient)
    # For now, we just iterate whatever is in the bank
    for feat in feature_bank:
        test_subset = best_subset + [feat]
        
        res = evaluate_model('XGBoost', train_df, test_subset, target=target, class_weight=class_weight, k_folds=10)
        curr_auprc = res['metrics_summary'].loc['AUPRC']['Mean']
        
        if curr_auprc > best_auprc:
            print(f"   Adding back '{feat}' IMPROVED score to {curr_auprc:.4f}")
            best_subset = test_subset
            best_auprc = curr_auprc

    print(f"\n🏁 Final Feature Set ({len(best_subset)} features): {best_subset}")
    print(f"🏆 Final AUPRC: {best_auprc:.4f}")
    
    return best_subset

def tune_xgboost_hyperparameters(train_df, features, target, class_weight=None):
    
    # 1. Define the "Modest" Grid
    # We focus on the most impactful parameters to keep runtime reasonable
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 6],
        'learning_rate': [0.05, 0.1, 0.3],
        'subsample': [0.8, 1.0]
    }
    
    # Generate all combinations
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"🔧 Starting Hyperparameter Tuning. Testing {len(param_combinations)} combinations...")
    
    best_auprc = -1
    best_params = {}
    
    # 2. Iterate through Grid
    for i, params in enumerate(param_combinations):
        print(f"\n[{i+1}/{len(param_combinations)}] Testing: {params}")
        
        # Pass the current params to evaluate_model
        # We use k_folds=5 to speed up the tuning process vs the standard 10
        results = evaluate_model(
            'XGBoost', 
            train_df, 
            features, 
            target=target, 
            params=params, 
            class_weight=class_weight, 
            k_folds=5 
        )
        
        curr_auprc = results['metrics_summary'].loc['AUPRC']['Mean']
        
        print(f"   -> Result: AUPRC = {curr_auprc:.4f}")
        
        if curr_auprc > best_auprc:
            print(f"      ✅ New Best! (Previous: {best_auprc:.4f})")
            best_auprc = curr_auprc
            best_params = params
            
    print(f"\n🏆 Tuning Complete.")
    print(f"Best AUPRC: {best_auprc:.4f}")
    print(f"Best Params: {best_params}")
    
    return best_params


def run_final_evaluation(train_df, X_test, y_test, features, best_params, xgb_scale_weight):
    
    # 1. Prepare Data
    print("⚙️ Preparing and Scaling Data...")
    X_train = train_df[features].copy()
    y_train = train_df['Class'].copy()
    X_test_selected = X_test[features].copy() 

    # 2. Robust Scaling
    scaler = RobustScaler()
    
    # Fit on TRAIN, transform both
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=features, 
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_selected), 
        columns=features, 
        index=X_test_selected.index
    )

    # 3. Train Final XGBoost Model
    print(f"🚀 Training Final XGBoost Model with params: {best_params}...")
    clf = XGBClassifier(
        eval_metric='logloss', 
        random_state=42,
        scale_pos_weight=xgb_scale_weight,
        **best_params
    )
    
    clf.fit(X_train_scaled, y_train)

    # 4. Evaluation
    print("\n📊 --- FINAL TEST SET RESULTS ---")
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        'AUPRC': average_precision_score(y_test, y_proba),
        'AUC': roc_auc_score(y_test, y_proba),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Specificity': tn / (tn + fp)
    }

    print("-" * 35)
    for k, v in metrics.items():
        print(f"{k:<20} | {v:.4f}")
    print("-" * 35)

    # 5. SHAP Analysis
    print("\n🧠 Generating SHAP Analysis...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_scaled)
    
    plt.figure()
    plt.title("SHAP Summary Plot (Test Set)", fontsize=16)
    shap.summary_plot(shap_values, X_test_scaled, show=False)
    plt.show()

    # 6. Prepare Detailed Results DataFrame
    results_df = X_test_selected.copy()
    results_df['True Class'] = y_test
    results_df['Predicted Probability'] = y_proba
    results_df['Predicted Class'] = y_pred

    # Return everything needed for analysis
    return clf, metrics, results_df

def run_final_evaluation(train_df, X_test, y_test, features, best_params, xgb_scale_weight):
    """
    Trains the final model on the full training set, evaluates on the held-out test set,
    and returns the Model, Scaler, Metrics, and a DataFrame with detailed predictions.
    """
    
    # 1. Prepare Data
    print("⚙️ Preparing and Scaling Data...")
    X_train = train_df[features].copy()
    y_train = train_df['Class'].copy()
    X_test_selected = X_test[features].copy() 

    # 2. Robust Scaling
    scaler = RobustScaler()
    
    # Fit on TRAIN, transform both
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=features, 
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_selected), 
        columns=features, 
        index=X_test_selected.index
    )

    # 3. Train Final XGBoost Model
    print(f"🚀 Training Final XGBoost Model with params: {best_params}...")
    clf = XGBClassifier(
        eval_metric='logloss', 
        random_state=42,
        scale_pos_weight=xgb_scale_weight,
        **best_params
    )
    
    clf.fit(X_train_scaled, y_train)

    # 4. Evaluation
    print("\n📊 --- FINAL TEST SET RESULTS ---")
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        'AUPRC': average_precision_score(y_test, y_proba),
        'AUC': roc_auc_score(y_test, y_proba),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Specificity': tn / (tn + fp)
    }

    print("-" * 35)
    for k, v in metrics.items():
        print(f"{k:<20} | {v:.4f}")
    print("-" * 35)

    # 5. SHAP Analysis
    print("\n🧠 Generating SHAP Analysis...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_scaled)
    
    plt.figure()
    plt.title("SHAP Summary Plot (Test Set)", fontsize=16)
    shap.summary_plot(shap_values, X_test_scaled, show=False)
    plt.show()

    # 6. Prepare Detailed Results DataFrame
    results_df = X_test_selected.copy()
    results_df['True Class'] = y_test
    results_df['Predicted Probability'] = y_proba
    results_df['Predicted Class'] = y_pred

    # Return everything needed for analysis
    return clf, scaler, metrics, results_df

def plot_prediction_scatter(results_df, feature_x, feature_y, tn_fraction=0.3):
    """
    Visualizes predictions on a 2D feature plane using Seaborn.
    Downsamples True Negatives to avoid overcrowding.
    
    Args:
        results_df (pd.DataFrame): The dataframe returned by run_final_evaluation
        feature_x (str): Name of feature for X-axis
        feature_y (str): Name of feature for Y-axis
        tn_fraction (float): Fraction of True Negatives to display (default 0.1 for 10%)
    """
    plt.figure(figsize=(12, 7))
    
    # Create temporary columns for plotting logic
    plot_data = results_df.copy()
    plot_data['Prediction Status'] = np.where(
        plot_data['True Class'] == plot_data['Predicted Class'], 
        'Correct', 
        'Missed/Wrong'
    )
    plot_data['Class Label'] = plot_data['True Class'].map({0: 'Normal', 1: 'Fraud'})
    
    # --- Downsampling Logic ---
    # 1. Identify True Negatives (Normal Class + Correct Prediction)
    mask_tn = (plot_data['True Class'] == 0) & (plot_data['Prediction Status'] == 'Correct')
    
    # 2. Split into TN vs Everything Else (TP, FP, FN)
    df_tn = plot_data[mask_tn]
    df_critical = plot_data[~mask_tn]
    
    # 3. Sample the TNs
    if 0 < tn_fraction < 1.0:
        df_tn_sampled = df_tn.sample(frac=tn_fraction, random_state=42)
        print(f"📉 Downsampling True Negatives: Showing {len(df_tn_sampled)} samples ({tn_fraction*100}%) along with {len(df_critical)} critical points (Errors/Fraud).")
    else:
        df_tn_sampled = df_tn
    
    # 4. Recombine for plotting
    final_plot_data = pd.concat([df_tn_sampled, df_critical])
    
    # Plot
    sns.scatterplot(
        data=final_plot_data,
        x=feature_x,
        y=feature_y,
        hue='Class Label',           # Color by Fraud/Normal
        style='Prediction Status',   # Shape by Correct/Incorrect
        palette={'Normal': 'dodgerblue', 'Fraud': 'red'},
        markers={'Correct': 'o', 'Missed/Wrong': 'X'},
        s=80,                        # Size of points
        alpha=0.5
    )
    
    plt.title(f"Prediction Analysis: {feature_x} vs {feature_y} (TN sampled at {tn_fraction:.0%})", fontsize=15)
    plt.xlabel(feature_x, fontsize=12)
    plt.ylabel(feature_y, fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Legend")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def predict_new_row(model, row, scaler, features, threshold=0.5):
    row_features = row[features]
    model_input = pd.DataFrame(scaler.transform(row_features), columns=features)
    if 'Class' in row.columns:
        true_class = row['Class']
        print(f"True Class:   {true_class}")
    proba_fraud = model.predict_proba(model_input)[0][1]
    class_prediction = proba_fraud > threshold
    final_report = {"final_prediction": class_prediction, "probability_of_fraud": proba_fraud}
    print(f"\n🔎 Checking Index: {row.index[0]}")
    print(f"Model Prob:   {proba_fraud:.4f}")
    print(f"Prediction:   {'FRAUD' if class_prediction else 'Normal'}")

    return final_report
