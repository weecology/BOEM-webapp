import os
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import re

def parse_aflight_file(file_path):
    """Parse an .aflight file and extract center coordinates and image info for each capture, including date."""
    # Get flight name from basename of file
    flight_name = Path(file_path).stem
    """Parse an .aflight file and extract center coordinates and image info for each capture, including date."""
    # Try to open with utf-8, fallback to binary and let ElementTree handle encoding
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception:
        with open(file_path, 'rb') as f:
            content = f.read()
        try:
            root = ET.fromstring(content)
        except Exception as e:
            print(f"Failed to parse XML: {e}")
            return pd.DataFrame([])

    # Extract date from <StartDate>
    date = root.findtext('StartDate')

    records = []
    # For each CameraEntity, get all CaptureEntity children
    for camera in root.findall('.//CameraEntity'):
        for capture in camera.findall('.//CaptureEntity'):
            camera_guid = capture.findtext('CameraGUID')
            filename = capture.findtext('Filename')
            lat = capture.findtext('BRLat')
            lon = capture.findtext('BRLon')
            if camera_guid and filename and lat and lon:
                records.append({
                    'unique_image': os.path.splitext(filename)[0],
                    'lat': float(lat),
                    'long': float(lon),
                    'flight_name': flight_name,
                    'date': date
                })
    return pd.DataFrame(records)

def extract_flight_coordinates(flight_name):
    # Find all .aflight files
    data_dir = Path("app/data/metadata")
    file_path = data_dir / f"{flight_name}.aflight"
    
    if not file_path.exists():
        print(f"No .aflight file found for {flight_name}!")
        return
    
    # Create output directory
    output_dir = Path("app/data/metadata")
    output_dir.mkdir(exist_ok=True)
    
    # Process the .aflight file
    print(f"Processing {file_path}...")
    
    # Parse the file and get coordinates
    df = parse_aflight_file(file_path)
    
    if not df.empty:
        # Create output filename
        output_file = output_dir / f"{flight_name}.csv"
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Saved coordinates for flight {flight_name} to {output_file}")
    else:
        print(f"No valid coordinates found in {file_path}")

# Generate metadata for all flights and save as metadata.csv
def generate_metadata():
    data_dir = Path("app/data/metadata")
    metadata_df = pd.DataFrame()
    for file in data_dir.glob("*.csv"):
        df = pd.read_csv(file)
        metadata_df = pd.concat([metadata_df, df])
    metadata_df.to_csv("app/data/metadata.csv", index=False)