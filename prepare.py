import paramiko
import os
from pathlib import Path
import datetime
from app.utils.vector_utils import optimize_vector
import pandas as pd
import requests
import cv2
import numpy as np

def get_newest_report_dir(sftp, base_dir):
    """Find the newest timestamped directory in the reports folder"""
    dirs = []
    for entry in sftp.listdir_attr(base_dir):
        if entry.longname.startswith('d'):  # Check if it's a directory
            try:
                # Try to parse the directory name as a timestamp
                timestamp = datetime.datetime.strptime(entry.filename, '%Y%m%d_%H%M%S')
                dirs.append((timestamp, entry.filename))
            except ValueError:
                continue
    
    if not dirs:
        raise Exception("No timestamped directories found in /reports")
        
    # Sort by timestamp and get the newest
    newest = sorted(dirs, key=lambda x: x[0], reverse=True)[0][1]
    return newest

def download_report_files(base_dir):
    """Download files from newest report directory on server"""
    
    # SSH connection details
    hostname = os.environ.get('REPORT_SERVER_HOST') 
    username = os.environ.get('REPORT_SERVER_USER') 
    password = os.environ.get('REPORT_SERVER_PASS')
    port = 2222
    
    if not all([hostname, username]):
        raise Exception("Missing required environment variables for SSH connection")
    
    # Create app data directory if it doesn't exist
    data_dir = Path('app/data')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Connect to server
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(hostname, username=username, password=password, port=port)
        sftp = ssh.open_sftp()

        # Get newest report directory
        newest_dir = get_newest_report_dir(sftp, base_dir)
        report_path = f'{base_dir}/{newest_dir}'
        
        # Download all files from the directory
        for filename in sftp.listdir(report_path):
            remote_path = f'{report_path}/{filename}'
            local_path = data_dir / filename
            
            print(f'Downloading {filename}...')
            sftp.get(remote_path, str(local_path))
            
        print('Download complete')
            
    except Exception as e:
        print(f'Error downloading report files: {str(e)}')
        raise
        
    finally:
        if 'sftp' in locals():
            sftp.close()
        if 'ssh' in locals():
            ssh.close()

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

def download_and_crop_images():
    """
    Downloads one sample image for each unique label from predictions.csv,
    crops around the detection, and saves the result
    """
    # SSH connection details
    hostname = os.environ.get('REPORT_SERVER_HOST')  # Remove square brackets
    username = os.environ.get('REPORT_SERVER_USER') 
    password = os.environ.get('REPORT_SERVER_PASS')
    port = 2222
    
    if not all([hostname, username]):
        raise Exception("Missing required environment variables for SSH connection")
    
    # Get the app's data directory
    app_data_dir = Path(__file__).parent / "app" / "data"
    images_dir = app_data_dir / "images"
    
    # Create images directory if it doesn't exist
    images_dir.mkdir(exist_ok=True)
    
    # Load predictions.csv
    predictions_file = app_data_dir / "predictions.csv"
    if not predictions_file.exists():
        raise FileNotFoundError(f"Could not find predictions file: {predictions_file}")
        
    df = pd.read_csv(predictions_file)
    
    # Get one sample for each unique label with bbox coordinates
    samples = df.groupby('label').first()
    
    # Connect to server
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(hostname, username=username, password=password, port=22)
        sftp = ssh.open_sftp()
        
        # Process each sample
        for label, row in samples.iterrows():
            # Get original image basename and create new filename with crop_{label}
            original_basename = Path(row['image_path']).stem
            safe_label = "".join(c if c.isalnum() else "_" for c in label)
            image_path = images_dir / f"{original_basename}.jpg"
            
            # Skip if local image already exists
            if image_path.exists():
                print(f"Image for {label} already exists, skipping...")
                continue
            
            try:
                # Full path on HPC
                remote_path = f"/blue/ewhite/b.weinstein/BOEM/sample_flight/JPG_2024_Jan27/annotated/{row['image_path']}"
                
                # Check if remote file exists
                try:
                    sftp.stat(remote_path)
                except FileNotFoundError:
                    print(f"Remote image not found: {remote_path}")
                    continue
                
                print(f"Downloading and processing image for {label}...")
                # Download to final location
                sftp.get(remote_path, str(image_path))
                
                # Get bbox coordinates
                bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                
                # Crop image in place
                cropped = crop_image(image_path, bbox)
                cv2.imwrite(str(image_path), cropped)
                
                print(f"Processed image for {label}")
                
            except Exception as e:
                print(f"Error processing image for {label}: {str(e)}")
                if image_path.exists():
                    image_path.unlink()
                
    except Exception as e:
        print(f"Error connecting to server: {str(e)}")
        raise
        
    finally:
        if 'sftp' in locals():
            sftp.close()
        if 'ssh' in locals():
            ssh.close()

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
    report_dir = os.environ.get('REPORT_DIR')
    #download_report_files(report_dir)
    download_and_crop_images()
    create_heatmap()
    # Create vector data
    #optimize_vector("app/data/predictions.csv", "app/data/processed/predictions.shp")
