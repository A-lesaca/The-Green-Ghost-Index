import ee  # Google Earth Engine
import pandas as pd
import numpy as np


# Initialize GEE (Assuming 'ee.Initialize()' is run beforehand)

def get_ndvi_change(row: pd.Series, year_start: int, year_end: int) -> float:
    # GEE Initialization/Setup code remains the same...
    """
    Calculates the change in average NDVI (Normalized Difference Vegetation Index)
    between two time periods at a specific project coordinate.
    """
    # 1. Define the point of interest
    point = ee.Geometry.Point(row['longitude'], row['latitude'])

    # 2. Load Sentinel-2 imagery (a common choice for land-use change)
    S2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(point) \
        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 10)

    # 3. Calculate NDVI function
    def calculate_ndvi(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)

    # 4. Get mean NDVI for the start period
    ndvi_start = S2.filterDate(f'{year_start}-01-01', f'{year_start}-12-31') \
        .map(calculate_ndvi).select('NDVI').mean()

    # 5. Get mean NDVI for the end period
    ndvi_end = S2.filterDate(f'{year_end}-01-01', f'{year_end}-12-31') \
        .map(calculate_ndvi).select('NDVI').mean()

    try:
        # Extract mean NDVI value at the point for both periods
        mean_start = ndvi_start.reduceRegion(ee.Reducer.mean(), point, 30).get('NDVI').getInfo()
        mean_end = ndvi_end.reduceRegion(ee.Reducer.mean(), point, 30).get('NDVI').getInfo()

        if mean_start and mean_end:
            # Ghost_Metric: (Start NDVI - End NDVI).
            # A high positive number means a drop in vegetation (e.g., construction).
            return float(mean_start) - float(mean_end)
        else:
            return 999.0  # Sentinel value for no data
    except Exception as e:
        # Fixed: using project_name for error reporting
        print(f"GEE error for project {row['project_name']}: {e}")
        return 999.0


def run_satellite_audit(master_df: pd.DataFrame) -> pd.DataFrame:
    """Applies the GEE check and labels the project (Step 2)."""
    print("\n--- STEP 2: Running Satellite Audit (GEE Simulation) ---")

    # IMPORTANT: Simulating GEE results since the actual GEE code cannot run here.
    # In a real pipeline, the line below would be uncommented.

    # audit_results = master_df.apply(lambda row: get_ndvi_change(row, year_start=2020, year_end=2024), axis=1)

    # --- SIMULATION START ---
    # Dummy data generation for demonstration
    # Assume 10% of projects are ghosts, and low NDVI change is the marker.
    np.random.seed(42)
    master_df['ndvi_change_metric'] = np.random.uniform(low=0.001, high=0.2, size=len(master_df))
    # Add some projects that are 'cancelled' or 'announced' with no change
    mask_no_change = master_df['gem_status'].isin(['cancelled', 'announced'])
    master_df.loc[mask_no_change, 'ndvi_change_metric'] = np.random.uniform(low=0.001, high=0.01,
                                                                            size=mask_no_change.sum())
    # --- SIMULATION END ---

    # Add the key feature to the dataset
    # master_df.loc[audit_results.index, 'ndvi_change_metric'] = audit_results

    # The 'Ghost' label: Low NDVI change AND the project was funded/active
    # A low NDVI change (< 0.05) suggests no site preparation occurred.
    # A status of 'operating' or 'construction' makes it a potential ghost.

    # Fixed: Using the correct 'gem_status' column for labeling
    active_mask = master_df['gem_status'].isin(['operating', 'construction', 'pre-construction', 'retired'])
    low_change_mask = (master_df['ndvi_change_metric'] < 0.05)

    master_df['is_ghost'] = low_change_mask & active_mask
    master_df['is_ghost'] = master_df['is_ghost'].astype(int)  # Convert boolean to 0/1

    # Fixed: Adding a simple audit status
    master_df['audit_status'] = np.where(master_df['is_ghost'] == 1, 'Ghost Flagged', 'Activity Visible/Inactive')

    print(f"Audit completed. Found {master_df['is_ghost'].sum()} potential Green Ghosts (is_ghost=1).")
    return master_df