import pandas as pd
import numpy as np
import os  # Added for path handling

# Helper function (Updated for robust CSV reading)
def load_data(file_path: str) -> pd.DataFrame:
    """Load a raw data file (CSV or XLSX), handling known ADB parsing issues."""
    if file_path.endswith('.csv'):
        # Added robust settings for the ADB file
        if 'adb_projects_raw.csv' in file_path:
            df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
        else:
            df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type for {file_path}")

    # Standardize columns (crucial for merging!)
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()
    return df


def create_master_data(raw_data_dir: str, output_path: str):
    """Loads, cleans, and merges all raw data into a master dataset (Step 1)."""
    print("\n--- STEP 1: Data Acquisition & Merge ---")

    # 1. Load data
    adb_df = load_data(os.path.join(raw_data_dir, 'adb_projects_raw.csv'))
    # NOTE: gcf_dashboard_raw is a CSV, not an XLSX. Updated file extension logic is in the helper.
    gcf_df = load_data(os.path.join(raw_data_dir, 'gcf_dashboard_raw.csv'))
    ti_df = load_data(os.path.join(raw_data_dir, 'ti_cpi_2024.csv'))
    gem_df = load_data(os.path.join(raw_data_dir, 'gem_trackers_raw.csv'))  # THE BASE PROJECT LIST

    # 2. Clean and Prepare GEM data (The Base Project List)
    gem_df.rename(columns={
        'country/area': 'country',
        'capacity_(mw)': 'funded_capacity_mw',
        'technology': 'project_type',
        'status': 'gem_status',  # Renamed to avoid collision with ADB 'status'
        'start_year': 'start_year'
    }, inplace=True)
    master_df = gem_df[['project_name', 'country', 'funded_capacity_mw', 'project_type',
                        'start_year', 'latitude', 'longitude', 'gem_status']].copy()

    # 3. Integrate Corruption Index (TI CPI)
    ti_df.rename(columns={'cpi_score_2024': 'cpi_score'}, inplace=True)
    master_df = pd.merge(master_df, ti_df[['country', 'cpi_score']], on='country', how='left')

    # 4. Integrate Financial Data (ADB/GCF) as a Country-Level Proxy
    # ADB: Use Country-Average Loan since project-level merging is impossible.
    adb_df.rename(columns={'loan_amount_usd_m': 'total_loan_usd'}, inplace=True)
    adb_loans = adb_df.groupby('country')['total_loan_usd'].mean().reset_index()
    adb_loans.rename(columns={'total_loan_usd': 'total_loan_usd'}, inplace=True)

    # Merge the average loan amount onto the Master list
    master_df = pd.merge(master_df, adb_loans, on='country', how='left')

    # 5. Add Placeholder/Target Columns
    master_df['project_id'] = np.nan  # Can't be populated without a linker file
    master_df['rule_of_law_score'] = np.nan  # Requires external WJP data
    master_df['is_ghost'] = np.nan  # The TARGET variable
    master_df['audit_status'] = np.nan

    # Reorder columns to match the desired structure
    master_df = master_df[['project_id', 'project_name', 'country', 'latitude', 'longitude',
                           'total_loan_usd', 'start_year', 'funded_capacity_mw',
                           'project_type', 'cpi_score', 'rule_of_law_score',
                           'is_ghost', 'audit_status', 'gem_status']]

    # Final save
    master_df.to_csv(output_path, index=False)
    print(f"Master dataset created with {len(master_df)} projects and saved to: {output_path}")

    return master_df