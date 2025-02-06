import geopandas as gpd
import os
import subprocess
import shutil
from shapely.geometry import shape, mapping, Polygon
import numpy as np
from pathlib import Path
import tempfile
import gc
import pandas as pd

MAX_FILE_SIZE_MB = 200
BATCH_SIZE = 1000  # Process rows in batches

def check_tippecanoe():
    """Check if tippecanoe is installed"""
    return shutil.which('tippecanoe') is not None

def convert_to_shapely(geometry_dict):
    """Convert dictionary geometry to Shapely object"""
    try:
        if isinstance(geometry_dict['coordinates'], np.ndarray):
            # Convert numpy array to list
            coords = geometry_dict['coordinates'].tolist()
            geometry_dict['coordinates'] = coords
        return shape(geometry_dict)
    except Exception as e:
        print(f"Error converting geometry: {str(e)}")
        return None

def simplify_geometries(gdf, tolerance=0.0001):
    """Simplify geometries while preserving topology"""
    try:
        simplified = gdf.copy()
        
        # Convert geometries if they're in dictionary format
        for idx, geom in simplified.geometry.items():
            if isinstance(geom, dict):
                simplified.at[idx, 'geometry'] = convert_to_shapely(geom)
        
        # Now simplify
        simplified.geometry = simplified.geometry.simplify(tolerance=tolerance, preserve_topology=True)
        return simplified
    except Exception as e:
        print(f"Error in simplify_geometries: {str(e)}")
        return gdf

def get_file_size_mb(file_path):
    """Get file size in MB"""
    return Path(file_path).stat().st_size / (1024 * 1024)

def estimate_size(gdf_chunk):
    """Estimate size of GeoDataFrame chunk without writing to disk"""
    return gdf_chunk.memory_usage(deep=True).sum() / (1024 * 1024)

def chunk_dataframe(gdf, max_size_mb=MAX_FILE_SIZE_MB):
    """Split GeoDataFrame into chunks based on estimated size"""
    chunks = []
    total_rows = len(gdf)
    
    # Process in batches
    for start_idx in range(0, total_rows, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, total_rows)
        batch = gdf.iloc[start_idx:end_idx].copy()
        
        # Estimate batch size
        batch_size = estimate_size(batch)
        
        if batch_size > max_size_mb:
            # If batch is too large, split it further
            mid_idx = (end_idx - start_idx) // 2
            chunks.extend([
                batch.iloc[:mid_idx],
                batch.iloc[mid_idx:]
            ])
        else:
            chunks.append(batch)
        
        # Clear memory
        del batch
        gc.collect()
    
    return chunks

def create_vector_tiles(gdf, output_path, min_zoom=0, max_zoom=14):
    """Convert GeoDataFrame to MBTiles format with optimization"""
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_geojson = os.path.join(temp_dir, "temp.geojson")
            gdf.to_file(temp_geojson, driver='GeoJSON')
            
            # Enhanced Tippecanoe options with more aggressive optimization
            cmd = [
                'tippecanoe',
                '-o', output_path,
                '--drop-densest-as-needed',
                '--extend-zooms-if-still-dropping',
                f'--minimum-zoom={min_zoom}',
                f'--maximum-zoom={max_zoom}',
                '--force',
                '--simplification=15',  # Increased from 10
                '--preserve-input-order',
                '--read-parallel',
                '--drop-smallest-as-needed',
                '--maximum-tile-bytes=2000000',  # Reduced to 2MB tile size limit
                '--hilbert',
                '--drop-rate=10',  # More aggressive feature dropping
                '--minimum-detail=12',  # Increased simplification
                '--cluster-distance=10',  # Cluster nearby features
                '--detect-shared-borders',  # Optimize shared borders
                '--grid-low-zooms',  # Grid-based simplification at low zooms
                '--drop-fraction-as-needed',  # Drop features to maintain size
                '--coalesce',  # Combine similar features
                '--coalesce-smallest-as-needed',
                '--limit-tile-feature-count=50000',  # Limit features per tile
                temp_geojson
            ]
            
            print("Running Tippecanoe with enhanced optimization flags...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Tippecanoe error: {result.stderr}")
            
            if os.path.exists(output_path):
                output_size = get_file_size_mb(output_path)
                print(f"MBTiles file created successfully. Size: {output_size:.2f}MB")
                
                # If still over limit, try one more time with even more aggressive settings
                if output_size > MAX_FILE_SIZE_MB:
                    print("File still over size limit. Attempting final optimization...")
                    os.remove(output_path)
                    cmd.extend([
                        '--drop-rate=15',
                        '--simplification=20',
                        '--maximum-tile-bytes=1000000',
                        '--minimum-detail=15',
                        '--cluster-distance=20'
                    ])
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        raise Exception(f"Tippecanoe error in final optimization: {result.stderr}")
                    
                    output_size = get_file_size_mb(output_path)
                    print(f"Final MBTiles file size: {output_size:.2f}MB")
            
            return output_path
        
    except Exception as e:
        print(f"Error in create_vector_tiles: {str(e)}")
        raise e

def optimize_vector(input_path, output_path, optimization_type='mbtiles'):
    """Main function to optimize vector data"""
    try:
        # Read the input file
        gdf = gpd.read_file(input_path)
        
        # Ensure the CRS is set to WGS 84
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        if gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        
        # Initial simplification
        gdf = simplify_geometries(gdf)
        
        file_size = estimate_size(gdf)
        if file_size > MAX_FILE_SIZE_MB:
            print(f"File size ({file_size:.2f}MB) exceeds {MAX_FILE_SIZE_MB}MB limit. Splitting into chunks.")
            chunks = chunk_dataframe(gdf)
            output_files = []
            
            for i, chunk in enumerate(chunks, 1):
                print(f"Processing chunk {i} of {len(chunks)}")
                chunk_output = output_path.replace(
                    '.' + output_path.split('.')[-1],
                    f'_part{i}.{output_path.split(".")[-1]}'
                )
                
                os.makedirs(os.path.dirname(chunk_output), exist_ok=True)
                
                if optimization_type == 'mbtiles':
                    if not check_tippecanoe():
                        raise Exception("Tippecanoe is not installed.")
                    chunk_output = create_vector_tiles(chunk, chunk_output)
                else:
                    chunk.to_file(chunk_output, driver='GeoJSON' if optimization_type == 'geojson' else 'ESRI Shapefile')
                
                output_files.append(chunk_output)
                del chunk
                gc.collect()
            
            return output_files
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if optimization_type == 'mbtiles':
                if not check_tippecanoe():
                    raise Exception("Tippecanoe is not installed.")
                return create_vector_tiles(gdf, output_path)
            else:
                gdf.to_file(output_path, driver='GeoJSON' if optimization_type == 'geojson' else 'ESRI Shapefile')
                return output_path
            
    except Exception as e:
        print(f"Error in optimize_vector: {str(e)}")
        raise e
    finally:
        if 'gdf' in locals():
            del gdf
        gc.collect()