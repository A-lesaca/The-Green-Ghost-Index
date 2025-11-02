import pandas as pd
import ee
import os
import sys
import shutil
import webbrowser
from datetime import datetime  # Added for the report date

# --- ABSOLUTE PATH FIX: This ensures the script always finds the project root ---
# 1. Determine the path to the project root (The directory *above* 'src')
# This assumes main.py is in the 'src' folder.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Add the project root to the system path to fix module imports
sys.path.append(PROJECT_ROOT)
# -------------------------------------------------------------------------------

# Now imports should work regardless of execution location (as long as the root is found)
from src.data_pipeline import create_master_data
from src.satellite_audit import run_satellite_audit
from src.model_builder import train_ghost_risk_model, create_final_index
from src.impact_analysis import measure_impact  # <-- NEW IMPORT

# Define paths relative to the PROJECT_ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'master_project_data.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'reports', 'rf_ghost_model.joblib')
FINAL_INDEX_PATH = os.path.join(PROJECT_ROOT, 'reports', 'final_green_ghost_index.csv')
# NEW HTML paths
HTML_TEMPLATE_PATH = os.path.join(PROJECT_ROOT, 'report_template.html')
FINAL_HTML_REPORT_PATH = os.path.join(PROJECT_ROOT, 'reports', 'green_ghost_report.html')


def generate_report_html(template_path: str, output_path: str, metrics: dict, index_df: pd.DataFrame,
                         impact_metrics: dict):  # <-- UPDATED SIGNATURE
    """
    Reads the HTML template, injects model metrics and table data, and writes the final report.
    """
    # 1. Read the template file
    try:
        with open(template_path, 'r') as f:
            html_content = f.read()
    except FileNotFoundError:
        print(f"\nFATAL ERROR: HTML template not found at {template_path}. Cannot generate report.")
        return

    # 2. Prepare Data and Generate Dynamic HTML Sections

    # --- Model Performance ---
    roc_auc_score = f"{metrics['roc_auc']:.4f}"

    # Feature Importance List HTML
    feature_html = ""
    top_features = sorted(metrics['feature_importance'].items(),
                          key=lambda item: item[1], reverse=True)[:5]
    for feature, importance in top_features:
        display_name = feature.replace('_', ' ').title()
        feature_html += (
            f'<li class="flex justify-between"><span>- {display_name}:</span> '
            f'<span class="font-bold text-green-700">{importance:.4f}</span></li>\n'
        )

    # --- Risky Projects Table ---
    table_rows_html = ""
    for _, row in index_df.head(5).iterrows():
        risk_score = row['ghost_risk_score']
        score_text = f"{risk_score:.3f} ({'High' if risk_score > 0.8 else 'Medium' if risk_score > 0.6 else 'Low'})"
        score_class = 'text-red-600' if risk_score > 0.8 else 'text-orange-600' if risk_score > 0.6 else 'text-green-600'

        sat_status = row['audit_status']
        sat_class = 'text-red-600' if 'No Construction' in sat_status else 'text-yellow-600' if 'Inactivity' in sat_status else 'text-green-600'
        project_id = row['project_id'] if pd.notna(row['project_id']) else 'N/A'

        table_rows_html += f"""
        <tr>
            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{project_id}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{row['project_name']}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{row['country']}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm {score_class} font-bold">{score_text}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm {sat_class}">{sat_status}</td>
        </tr>
        """
    if len(index_df) > 5:
        table_rows_html += """
        <tr class="text-sm italic text-gray-400">
            <td colspan="5" class="px-6 py-4 text-center">... showing top 5 out of a total of {total_count} projects ...</td>
        </tr>
        """.format(total_count=len(index_df))

    # --- (NEW) Prepare Impact Metrics ---
    # These will be injected into new placeholders in the HTML template
    total_loan = f"${impact_metrics['total_portfolio_loan_usd']:,.0f}"
    at_risk_loan = f"${impact_metrics['predicted_at_risk_loan_usd']:,.0f}"
    total_mw = f"{impact_metrics['total_portfolio_capacity_mw']:,.1f} MW"
    at_risk_mw = f"{impact_metrics['predicted_at_risk_capacity_mw']:,.1f} MW"
    at_risk_loan_pct = f"{impact_metrics['pct_loans_at_risk']:.1%}"
    at_risk_mw_pct = f"{impact_metrics['pct_capacity_at_risk']:.1%}"

    # 3. Replace Placeholders
    final_html = html_content.replace('{{ GENERATION_DATE }}', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Model Performance Placeholders
    final_html = final_html.replace('{{ ROC_AUC_SCORE }}', roc_auc_score)
    final_html = final_html.replace('{{ FEATURE_IMPORTANCE_LIST_HTML }}', feature_html)

    # Risky Table Placeholder
    final_html = final_html.replace('{{ RISKY_PROJECTS_ROWS_HTML }}', table_rows_html)

    # (NEW) Impact Metric Placeholders
    # (Assumes your HTML template has these placeholders, e.g., in summary cards)
    final_html = final_html.replace('{{ TOTAL_PORTFOLIO_LOAN_USD }}', total_loan)
    final_html = final_html.replace('{{ PREDICTED_AT_RISK_LOAN_USD }}', at_risk_loan)
    final_html = final_html.replace('{{ TOTAL_PORTFOLIO_CAPACITY_MW }}', total_mw)
    final_html = final_html.replace('{{ PREDICTED_AT_RISK_CAPACITY_MW }}', at_risk_mw)
    final_html = final_html.replace('{{ PERCENT_LOANS_AT_RISK }}', at_risk_loan_pct)
    final_html = final_html.replace('{{ PERCENT_CAPACITY_AT_RISK }}', at_risk_mw_pct)

    # 4. Write the final HTML file
    try:
        with open(output_path, 'w') as f:
            f.write(final_html)
        print(f"\n✨ Report generated and opened in browser: {output_path}")

        # Open the generated report in the default web browser
        webbrowser.open_new_tab('file://' + os.path.abspath(output_path))
    except Exception as e:
        print(f"\nWarning: Could not write or launch web browser. Access report manually at: {output_path}. Error: {e}")


def main():
    # --- Step 0: Initial Setup (THE CRYSTAL BALL!) ---
    # ... (Branding print omitted for brevity) ...
    print("\n" + "\u2554" + "\u2550" * 68 + "\u2557")
    print("\u2551" + " " * 68 + "\u2551")
    print(
        "\u2551" + "         \u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF  " + "  \u2551")
    print("\u2551" + "        \u25CF  THE GREEN GHOST INDEX 🔮  \u25CF  " + " \u2551")
    print(
        "\u2551" + "         \u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF\u25CF  " + "  \u2551")
    print("\u2551" + " " * 68 + "\u2551")
    print("\u255a" + "\u2550" * 68 + "\u255d\n")
    print("**Activating predictive model to uncover fraudulent 'ghost' energy projects...**\n")

    # --- Pre-flight Check: Ensure the required folders exist ---
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # --- Pre-flight Check: Verifying Raw Data Files ---
    # ... (File check omitted for brevity) ...
    required_files = ['adb_projects_raw.csv', 'gcf_dashboard_raw.csv',
                      'ti_cpi_2024.csv', 'gem_trackers_raw.csv']
    missing_files = []
    print("\n--- Pre-flight Check: Verifying Raw Data Files ---")
    for filename in required_files:
        full_path = os.path.join(RAW_DATA_DIR, filename)
        if not os.path.exists(full_path):
            missing_files.append(full_path)
    if missing_files:
        print("\nFATAL ERROR: The following raw data files are missing or misnamed:")
        for path in missing_files:
            print(f" - MISSING: {path}")
        print("\nACTION REQUIRED: Please copy these files into the 'data/raw' folder and run again.")
        return

    # Initialize GEE (The warning is expected, as we agreed to simulate)
    try:
        ee.Initialize(project='your-gee-project-id')
        print("Google Earth Engine initialized successfully.")
    except Exception as e:
        print(f"Warning: GEE initialization failed. Satellite audit will be simulated instead: {e}")

    # --- Step 1: Data Acquisition & Merge ---
    master_df = create_master_data(RAW_DATA_DIR, PROCESSED_DATA_PATH)

    # --- Step 2: Satellite Audit & Labeling ---
    audited_df = run_satellite_audit(master_df.copy())

    # --- Step 3: Train the Risk Model ---
    model_metrics, df_with_score = train_ghost_risk_model(audited_df, MODEL_PATH)

    print("\n--- Model Performance Summary ---")
    print(f"ROC AUC Score: {model_metrics['roc_auc']:.4f}")
    print("Top Feature Importances:")
    for feature, importance in sorted(model_metrics['feature_importance'].items(),
                                      key=lambda item: item[1], reverse=True):
        print(f"- {feature}: {importance:.4f}")

    # --- STEP 4: Measure the Impact (NEWLY IMPLEMENTED) ---
    # This step calculates the "real-world damage" in MW and USD
    impact_metrics = measure_impact(df_with_score, risk_threshold=0.8)

    # --- Step 5: Create the Final Index (Formerly Step 4) ---
    final_index_df = create_final_index(df_with_score, FINAL_INDEX_PATH)

    print("\n--- PIPELINE COMPLETED SUCCESSFULLY ---\n")
    print(f"Top 5 Riskiest Projects (Predicted by the Crystal Ball):\n{final_index_df.head()}")

    # --- Step 6: Generate and Launch HTML Report (Formerly Step 5) ---
    generate_report_html(
        HTML_TEMPLATE_PATH,
        FINAL_HTML_REPORT_PATH,
        model_metrics,
        final_index_df,
        impact_metrics  # <-- Pass new metrics to the report
    )


if __name__ == "__main__":
    main()