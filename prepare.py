import paramiko
import os
from pathlib import Path
import datetime

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
    
    if not all([hostname, username]):
        raise Exception("Missing required environment variables for SSH connection")
    
    # Create app data directory if it doesn't exist
    data_dir = Path('app/data')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Connect to server
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(hostname, username=username, key_filename=os.path.expanduser('~/.ssh/id_rsa'))
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

if __name__ == '__main__':
    report_dir = os.environ.get('REPORT_DIR')
    download_report_files(report_dir)
