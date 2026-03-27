import os
import glob
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import rasterio
from rasterio.features import geometry_mask
from tqdm import tqdm

# Folder containing ET tiffs
et_folder = "/Users/sidchaudhary/Downloads/OpenET_Monthly_Full_Extent"

# Shapefile path
shp_path = "/Users/sidchaudhary/Downloads/Field_Boundaries/cafe_field_perisitance_stats.shp"

# Output CSV
output_csv = "/Users/sidchaudhary/Downloads/pixel_level_ET_timeseries.csv"

# Load shapefile
print("Loading shapefile...")
fields = gpd.read_file(shp_path)
print(f"CRS: {fields.crs}")
print(f"Number of fields: {len(fields)}")
print(fields.head())

# Only months 04,05,06,10
valid_months = ["04", "05", "06", "10"]

all_tiffs = sorted(glob.glob(os.path.join(et_folder, "ET_Monthly_*.tif")))

filtered_tiffs = [
    f for f in all_tiffs 
    if re.search(r'_(\d{4})_(\d{2})\.tif$', f) and re.search(r'_(\d{4})_(\d{2})\.tif$', f).group(2) in valid_months
]

print(f"Total files found: {len(filtered_tiffs)}")

if len(filtered_tiffs) == 0:
    print("No matching files found. Please check the folder path and file naming pattern.")
    exit(1)

records = []
skipped_fields = 0
processed_fields = 0

for tif_path in tqdm(filtered_tiffs, desc="Processing rasters"):
    # Extract year and month from filename
    match = re.search(r'ET_Monthly_(\d{4})_(\d{2})\.tif$', os.path.basename(tif_path))
    if not match:
        print(f"Skipping file with unexpected name: {os.path.basename(tif_path)}")
        continue
    
    year, month = match.groups()
    
    try:
        # Open raster
        raster = rxr.open_rasterio(tif_path, masked=True).squeeze()
    except Exception as e:
        print(f"Error opening {os.path.basename(tif_path)}: {e}")
        continue
    
    # Reproject fields if CRS differs
    if fields.crs != raster.rio.crs:
        fields_proj = fields.to_crs(raster.rio.crs)
    else:
        fields_proj = fields
    
    # Loop over each field
    for idx, field in fields_proj.iterrows():
        field_geom = [field.geometry]
        
        try:
            # Clip raster to field with all_touched=True to include edge pixels
            clipped = raster.rio.clip(field_geom, fields_proj.crs, drop=True, all_touched=True)
        except Exception as e:
            # Skip if no overlap between field and raster
            skipped_fields += 1
            continue
        
        if clipped.size == 0:
            skipped_fields += 1
            continue
        
        et_values = clipped.values.flatten()
        et_values = et_values[~np.isnan(et_values)]
        
        if len(et_values) == 0:
            skipped_fields += 1
            continue
        
        processed_fields += 1
        field_mean = np.mean(et_values)
        
        # Get pixel coordinates
        xs, ys = np.meshgrid(clipped.x.values, clipped.y.values)
        xs = xs.flatten()
        ys = ys.flatten()
        
        valid_mask = ~np.isnan(clipped.values.flatten())
        
        xs = xs[valid_mask]
        ys = ys[valid_mask]
        
        relative_et = et_values / field_mean
        
        for x, y, et, rel in zip(xs, ys, et_values, relative_et):
            records.append({
                "field_id": idx,
                "year": int(year),
                "month": int(month),
                "x": x,
                "y": y,
                "ET": float(et),
                "relative_ET": float(rel)
            })

print(f"\nProcessing complete!")
print(f"Processed fields: {processed_fields}")
print(f"Skipped fields (no data): {skipped_fields}")

# Create DataFrame
df = pd.DataFrame(records)

print(f"Total rows: {len(df)}")
if len(df) > 0:
    print(df.head())
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")
else:
    print("\nWarning: No data was extracted. Please check:")
    print("1. The raster files contain valid data")
    print("2. The field boundaries overlap with the raster extent")
    print("3. The CRS of both datasets are compatible")
