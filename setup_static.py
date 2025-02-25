import shutil
from pathlib import Path

def setup_static_files():
    # Define paths
    node_modules_path = Path('node_modules/@uswds/uswds/dist')
    static_path = Path('app/static')
    
    # Create static directory structure
    directories = ['css', 'js', 'fonts', 'img']
    for dir_name in directories:
        (static_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Copy USWDS files
    try:
        # CSS files
        shutil.copytree(
            node_modules_path / 'css',
            static_path / 'css',
            dirs_exist_ok=True
        )
        
        # JavaScript files
        shutil.copytree(
            node_modules_path / 'js',
            static_path / 'js',
            dirs_exist_ok=True
        )
        
        # Font files
        shutil.copytree(
            node_modules_path / 'fonts',
            static_path / 'fonts',
            dirs_exist_ok=True
        )
        
        # Image files
        shutil.copytree(
            node_modules_path / 'img',
            static_path / 'img',
            dirs_exist_ok=True
        )
        
        # Copy species_select.js if it exists
        species_select_src = Path('app/static/species_select.js')
        if species_select_src.exists():
            shutil.copy2(
                species_select_src,
                static_path / 'js' / 'species_select.js'
            )
        
        print("Static files setup complete!")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find necessary files. Make sure you have run 'npm install @uswds/uswds' first.")
        print(f"Detailed error: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    setup_static_files() 