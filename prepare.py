import paramiko
import os
from pathlib import Path
import datetime
from app.utils.vector_utils import optimize_vector
import pandas as pd
import requests

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
    port = 22
    
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

def download_images():
    """
    Downloads one sample image for each unique label from predictions.csv
    using SFTP instead of HTTP requests
    """
    # SSH connection details
    hostname = os.environ.get('REPORT_SERVER_HOST')
    username = os.environ.get('REPORT_SERVER_USER') 
    password = os.environ.get('REPORT_SERVER_PASS')
    port = 22
    
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
    
    # Get one sample image path for each unique label
    sample_images = df.groupby('label').first()['image_path']
    # make the image path the full path on HPC by combining 
    sample_images = [f"/mnt/research/rc/data/images/{image_path}" for image_path in sample_images]

    # Connect to server
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(hostname, username=username, password=password, port=port)
        sftp = ssh.open_sftp()
        
        # Download each sample image
        for label, remote_path in sample_images.items():
            # Create safe filename from label
            safe_label = "".join(c if c.isalnum() else "_" for c in label)
            local_path = images_dir / f"{safe_label}.jpg"
            
            # Skip if image already exists
            if local_path.exists():
                print(f"Image for {label} already exists, skipping...")
                continue
                
            try:
                print(f"Downloading image for {label}...")
                sftp.get(remote_path, str(local_path))
                print(f"Downloaded image for {label}")
                
            except Exception as e:
                print(f"Error downloading image for {label}: {str(e)}")
                
    except Exception as e:
        print(f"Error connecting to server: {str(e)}")
        raise
        
    finally:
        if 'sftp' in locals():
            sftp.close()
        if 'ssh' in locals():
            ssh.close()

if __name__ == '__main__':
    report_dir = os.environ.get('REPORT_DIR')
    download_report_files(report_dir)
    # Create vector data
    optimize_vector("app/data/predictions.csv", "app/data/processed/predictions.shp")
    download_images()
