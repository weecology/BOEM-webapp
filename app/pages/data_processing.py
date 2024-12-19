import streamlit as st
from pathlib import Path
from utils.raster_utils import convert_to_cog
from utils.vector_utils import optimize_vector
import os
import shutil

def get_full_path(file):
    """Get the full path of an uploaded file"""
    # Try different attributes that might contain the full path
    if hasattr(file, 'name'):
        return str(Path(file.name).absolute())
    return str(Path(file.name).absolute())

def app():
    st.title("Data Processing")

    # Get the app's data directory for processed outputs
    app_data_dir = Path(__file__).parents[1] / "data"
    processed_dir = app_data_dir / "processed"
    temp_dir = processed_dir / "temp"

    # Ensure directories exist
    for dir_path in [processed_dir, temp_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Add option to select data source type
    data_source = st.radio(
        "Select Data Type",
        ["Raster File", "Vector File"]
    )

    if data_source == "Raster File":
        uploaded_file = st.file_uploader(
            "Upload a raster file",
            type=['tif', 'tiff']
        )
        
        if uploaded_file:
            try:
                full_path = get_full_path(uploaded_file)
                st.write("File Information:")
                st.write("Full Path:")
                st.code(full_path)  # Display path in a code block for better visibility
                st.write(f"Size: {len(uploaded_file.getvalue()) / (1024*1024):.2f} MB")
                
                if st.button("Process Raster"):
                    # Convert to COG and store in processed directory
                    output_file = processed_dir / "raster" / "cog" / f"processed_{Path(full_path).name}"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    with st.spinner("Processing raster file..."):
                        convert_to_cog(full_path, str(output_file))
                    
                    st.success(f"Raster file processed and saved to {output_file}")
                    
                    # Add download button for processed file
                    with open(output_file, 'rb') as f:
                        st.download_button(
                            label="Download processed file",
                            data=f,
                            file_name=output_file.name,
                            mime="image/tiff"
                        )
                    
                    # Cleanup temporary file if it was uploaded
                    if 'temp_input' in locals() and temp_input.exists():
                        temp_input.unlink(missing_ok=True)
                        
            except Exception as e:
                st.error(f"Error processing raster file: {str(e)}")
                if 'temp_input' in locals() and temp_input.exists():
                    temp_input.unlink(missing_ok=True)

    elif data_source == "Vector File":
        uploaded_files = st.file_uploader(
            "Upload vector file(s)",
            type=['shp', 'shx', 'dbf', 'prj', 'cpg', 'sbn', 'sbx', 'xml', 'geojson', 'kml', 'mbtiles'],
            accept_multiple_files=True,
            help="For shapefiles, please upload all related files (.shp, .shx, .dbf, etc.)"
        )
        
        if uploaded_files:
            try:
                # Group files by their base name (for shapefiles)
                file_groups = {}
                for uploaded_file in uploaded_files:
                    full_path = get_full_path(uploaded_file)
                    original_path = Path(full_path)
                    base_name = original_path.stem
                    if '.shp' in original_path.name:
                        base_name = base_name  # Keep the base name for .shp files
                    elif any(ext in original_path.name for ext in ['.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx', '.xml']):
                        base_name = base_name.rsplit('.', 1)[0]  # Remove the second extension
                    
                    if base_name not in file_groups:
                        file_groups[base_name] = {
                            'files': [], 
                            'original_dir': original_path.parent,
                            'paths': []
                        }
                    file_groups[base_name]['files'].append(uploaded_file)
                    file_groups[base_name]['paths'].append(full_path)

                # Process each group of files
                for base_name, group_info in file_groups.items():
                    files = group_info['files']
                    paths = group_info['paths']
                    original_dir = group_info['original_dir']
                    
                    # Identify main file and related files
                    main_file = None
                    main_path = None
                    related_files = []
                    related_paths = []
                    
                    for file, path in zip(files, paths):
                        if file.name.endswith('.shp'):
                            main_file = file
                            main_path = path
                        elif not main_file and file.name.endswith(('.geojson', '.kml', '.mbtiles')):
                            main_file = file
                            main_path = path
                        else:
                            related_files.append(file)
                            related_paths.append(path)
                    
                    if main_file:
                        st.write("\nFile Information:")
                        st.write("Main File Path:")
                        st.code(main_path)  # Display path in a code block for better visibility
                        st.write(f"Size: {len(main_file.getvalue()) / (1024*1024):.2f} MB")
                        
                        if related_files:
                            st.write("Related Files:")
                            for rel_path in related_paths:
                                st.code(rel_path)  # Display each related file path in a code block
                        
                        if st.button(f"Process {Path(main_path).name}", key=f"process_{base_name}"):
                            # Optimize vector and store in processed directory
                            output_file = processed_dir / "vector" / f"optimized_{Path(main_path).name}"
                            output_file.parent.mkdir(parents=True, exist_ok=True)
                            
                            with st.spinner("Processing vector file..."):
                                optimize_vector(main_path, str(output_file))
                            
                            st.success(f"Vector file processed and saved to {output_file}")
                            
                            # Add download button for processed file
                            with open(output_file, 'rb') as f:
                                st.download_button(
                                    label="Download processed file",
                                    data=f,
                                    file_name=output_file.name,
                                    mime="application/json"
                                )
                            
                            # Cleanup temporary file if it was uploaded
                            if 'temp_process_dir' in locals() and temp_process_dir.exists():
                                shutil.rmtree(temp_process_dir)

            except Exception as e:
                st.error(f"Error processing vector file: {str(e)}")
                if 'temp_process_dir' in locals() and temp_process_dir.exists():
                    shutil.rmtree(temp_process_dir)

if __name__ == "__main__":
    app()