import pandas as pd
import numpy as np


def measure_impact(df: pd.DataFrame, risk_threshold: float = 0.8) -> dict:
    """
    Calculates the real-world "damage" of "ghost" projects by comparing
    funded capacity/loans against actual (audited) and predicted status.

    This is the implementation of Step 4.
    """
    print("\n--- STEP 4: Measuring Real-World Impact ---")

    # --- Data Cleaning ---
    # Ensure numeric columns are numeric, fill NaNs with 0 for aggregation.
    # We assume a project with no loan/capacity data has 0 for this analysis.
    df['total_loan_usd'] = pd.to_numeric(df['total_loan_usd'], errors='coerce').fillna(0)
    df['funded_capacity_mw'] = pd.to_numeric(df['funded_capacity_mw'], errors='coerce').fillna(0)

    # --- 1. Total Portfolio Metrics (The Denominator) ---
    total_portfolio_loan_usd = df['total_loan_usd'].sum()
    total_portfolio_capacity_mw = df['funded_capacity_mw'].sum()
    total_projects = len(df)

    # --- 2. Audited Impact (Based on Step 2 Satellite Audit) ---
    # This is the "ground truth" impact from our audited sample
    audited_ghosts_df = df[df['is_ghost'] == 1.0]
    audited_lost_loan_usd = audited_ghosts_df['total_loan_usd'].sum()
    audited_lost_capacity_mw = audited_ghosts_df['funded_capacity_mw'].sum()

    # --- 3. Predicted Impact (Based on Step 3 Model Score) ---
    # This is the "predicted" impact across the *entire* portfolio
    # using the model's high-risk threshold.
    predicted_ghosts_df = df[df['ghost_risk_score'] >= risk_threshold]
    predicted_at_risk_loan_usd = predicted_ghosts_df['total_loan_usd'].sum()
    predicted_at_risk_capacity_mw = predicted_ghosts_df['funded_capacity_mw'].sum()
    predicted_at_risk_project_count = len(predicted_ghosts_df)

    # --- 4. Compile Metrics ---
    impact_metrics = {
        'risk_threshold': risk_threshold,
        'total_projects': total_projects,

        'total_portfolio_loan_usd': total_portfolio_loan_usd,
        'total_portfolio_capacity_mw': total_portfolio_capacity_mw,

        'audited_lost_loan_usd': audited_lost_loan_usd,
        'audited_lost_capacity_mw': audited_lost_capacity_mw,
        'audited_ghost_project_count': len(audited_ghosts_df),

        'predicted_at_risk_loan_usd': predicted_at_risk_loan_usd,
        'predicted_at_risk_capacity_mw': predicted_at_risk_capacity_mw,
        'predicted_at_risk_project_count': predicted_at_risk_project_count,

        # Calculate percentages for the report
        'pct_loans_at_risk': (
                    predicted_at_risk_loan_usd / total_portfolio_loan_usd) if total_portfolio_loan_usd > 0 else 0,
        'pct_capacity_at_risk': (
                    predicted_at_risk_capacity_mw / total_portfolio_capacity_mw) if total_portfolio_capacity_mw > 0 else 0,
    }

    print(f"Total Portfolio Value: ${total_portfolio_loan_usd:,.0f} ({total_projects} projects)")
    print(
        f"Predicted At-Risk Value (Score > {risk_threshold}): ${predicted_at_risk_loan_usd:,.0f} ({predicted_at_risk_project_count} projects)")
    print(f"Predicted At-Risk Capacity (Score > {risk_threshold}): {predicted_at_risk_capacity_mw:,.1f} MW")
    print(f"Ground Truth (Audited) Lost Capacity: {audited_lost_capacity_mw:,.1f} MW")

    return impact_metrics