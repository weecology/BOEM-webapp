import requests
from pathlib import Path
import zipfile
import io
import pandas as pd
import cv2

def get_newest_report_url():
    """Find the newest timestamped zip file from the web server"""
    base_url = "https://data.rc.ufl.edu/pub/ewhite/BOEM"
    response = requests.get(f"{base_url}/")
    
    if response.status_code != 200:
        raise Exception("Could not access web server")
    
    # Parse the directory listing for zip files and their timestamps
    # This is a simplified version - you may need to adjust based on actual server response
    files = [line for line in response.text.split('\n') if line.endswith('.zip')]
    if not files:
        raise Exception("No zip files found on server")
    
    # Get newest file (assuming filenames contain timestamps)
    newest_file = sorted(files)[-1]
    return f"{base_url}/{newest_file}"

def download_report_files():
    """Download and extract newest report zip file from web server"""
    # Create app data directory if it doesn't exist
    data_dir = Path('app/data')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get URL of newest zip file
        zip_url = get_newest_report_url()
        print(f'Downloading {zip_url}...')
        
        # Download zip file
        response = requests.get(zip_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download file: {response.status_code}")
            
        # Extract zip contents to data directory
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(data_dir)
            
        print('Download and extraction complete')
            
    except Exception as e:
        print(f'Error downloading report files: {str(e)}')
        raise

def crop_image(image_path, bbox, padding=30):
    """
    Crop image based on bbox coordinates with padding
    bbox: [xmin, ymin, xmax, ymax]
    padding: number of pixels to add around the bbox
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Add padding to bbox coordinates
    xmin = max(0, int(bbox[0] - padding))
    ymin = max(0, int(bbox[1] - padding))
    xmax = min(width, int(bbox[2] + padding))
    ymax = min(height, int(bbox[3] + padding))
    
    # Crop image
    cropped = img[ymin:ymax, xmin:xmax]
    return cropped
            
def create_heatmap():
    """Create heatmap rasters showing observation density for each flight line"""
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import box
    import rasterio
    from rasterio.transform import from_bounds
    import os
    from pathlib import Path

    # Read the shapefile
    gdf = gpd.read_file("app/data/predictions.shp")
    
    # Get unique flight lines
    flight_lines = gdf['flight_nam'].unique()

    # Create output directory if it doesn't exist
    output_dir = Path("app/data/heatmaps")
    output_dir.mkdir(exist_ok=True)

    for flight in flight_lines:
        # Filter data for this flight
        flight_data = gdf[gdf['flight_nam'] == flight]
        
        if len(flight_data) == 0:
            continue
            
        # Get bounds of flight line
        minx, miny, maxx, maxy = flight_data.total_bounds
        
        # Create grid
        cell_size = 0.001  # Adjust cell size as needed
        nx = int((maxx - minx) / cell_size)
        ny = int((maxy - miny) / cell_size)
        
        # Initialize empty array
        heatmap = np.zeros((ny, nx))
        
        # Create transform for raster
        transform = from_bounds(minx, miny, maxx, maxy, nx, ny)
        
        # Count points in each grid cell
        for idx, row in flight_data.iterrows():
            # Get point coordinates
            x, y = row.geometry.x, row.geometry.y
            
            # Convert to array indices
            col = int((x - minx) / cell_size)
            row = int((y - miny) / cell_size)
            
            # Increment count if within bounds
            if 0 <= row < ny and 0 <= col < nx:
                heatmap[row, col] += 1
        
        # Save as GeoTIFF
        output_path = output_dir / f"{flight}_heatmap.tif"
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=ny,
            width=nx,
            count=1,
            dtype=heatmap.dtype,
            crs=gdf.crs,
            transform=transform
        ) as dst:
            dst.write(heatmap, 1)
            
        print(f"Created heatmap for flight {flight}")

if __name__ == '__main__':
    download_report_files()
    create_heatmap()
    # Create vector data
    #optimize_vector("app/data/predictions.csv", "app/data/processed/predictions.shp")
