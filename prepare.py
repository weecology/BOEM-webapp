import requests
from pathlib import Path
import zipfile
import io
from bs4 import BeautifulSoup

def list_server_files():
    """List all files and directories on the web server"""
    base_url = "https://data.rc.ufl.edu/pub/ewhite/BOEM"
    response = requests.get(f"{base_url}/")
    
    if response.status_code != 200:
        raise Exception("Could not access web server")
    
    # Use BeautifulSoup to parse the HTML directory listing
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all links in the page
    files = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and not href.startswith('?') and not href.startswith('/pub/ewhite/'):
            # Remove URL encoding and trailing slashes
            filename = href.replace('%5F', '_').rstrip('/')
            files.append(filename)
            
    return files

def get_newest_report_url():
    """Find the newest timestamped zip file from the web server"""
    base_url = "https://data.rc.ufl.edu/pub/ewhite/BOEM"
    files = list_server_files()
    
    # Filter for zip files only
    zip_files = [f for f in files if f.endswith('.zip')]
            
    if not zip_files:
        raise Exception("No zip files found on server")
    
    # Get newest file (assuming filenames contain timestamps)
    newest_file = sorted(zip_files)[-1]
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

if __name__ == '__main__':
    # List all files
    files = list_server_files()
    print("\nFiles on server:")
    for file in files:
        print(file)
    
    # Download newest report
    download_report_files()
    # Create vector data
    #optimize_vector("app/data/predictions.csv", "app/data/processed/predictions.shp")
