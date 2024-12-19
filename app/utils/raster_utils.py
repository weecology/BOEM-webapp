from rio_cogeo.cogeo import cog_translate
import rasterio
import os
import numpy as np
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import math

def optimize_raster(input_path, output_path, max_size_mb=1000):
    """Optimize large raster for web viewing"""
    try:
        with rasterio.open(input_path) as src:
            # Calculate current size in MB
            current_size = os.path.getsize(input_path) / (1024 * 1024)
            
            # Calculate reduction factor needed
            reduction_factor = math.sqrt(current_size / max_size_mb)
            
            # Calculate new dimensions
            new_width = max(int(src.width / reduction_factor), 256)
            new_height = max(int(src.height / reduction_factor), 256)
            
            # Setup output profile based on input
            output_profile = src.profile.copy()
            
            # Force RGB if YCBCR
            if output_profile.get('photometric', '').upper() == 'YCBCR':
                output_profile['photometric'] = 'RGB'
            
            # Update profile with optimization settings
            output_profile.update({
                'driver': 'GTiff',
                'tiled': True,
                'blockxsize': 512,
                'blockysize': 512,
                'compress': 'DEFLATE',
                'predictor': 2,
                'width': new_width,
                'height': new_height,
                'interleave': 'pixel'
            })
            
            # Create VRT with resampling
            with WarpedVRT(src,
                          width=new_width,
                          height=new_height,
                          resampling=Resampling.average) as vrt:
                
                # Write optimized raster in chunks
                with rasterio.open(output_path, 'w', **output_profile) as dst:
                    # Process data in chunks
                    window_size = 1024
                    for i in range(1, src.count + 1):
                        for j in range(0, new_height, window_size):
                            window = rasterio.windows.Window(
                                col_off=0,
                                row_off=j,
                                width=new_width,
                                height=min(window_size, new_height - j)
                            )
                            data = vrt.read(i, window=window)
                            dst.write(data, window=window, indexes=i)
                    
                    # Build overviews with power of 2
                    overviews = [2**j for j in range(1, 6)]
                    dst.build_overviews(overviews, Resampling.average)
                    
                    # Copy metadata
                    dst.update_tags(**src.tags())
                    
        return output_path
    
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise Exception(f"Error optimizing raster: {str(e)}")

def convert_to_cog(input_path, output_path, max_size_mb=1000):
    """Convert a large raster to optimized Cloud Optimized GeoTIFF"""
    try:
        # Create output directories if they don't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create temp directory path
        temp_dir = os.path.join(os.path.dirname(output_path), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # First optimize the raster
        temp_optimized = os.path.join(temp_dir, os.path.basename(output_path).replace('.tif', '_temp.tif'))
        optimize_raster(input_path, temp_optimized, max_size_mb)
        
        # Setup COG profile
        cog_profile = {
            "driver": "GTiff",
            "interleave": "pixel",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "DEFLATE",
            "predictor": 2,
            "zlevel": 6
        }
        
        # Convert to COG
        cog_translate(
            temp_optimized,
            output_path,
            cog_profile,
            in_memory=False,
            quiet=False,
            allow_intermediate_compression=True
        )
        
        # Clean up temporary files
        if os.path.exists(temp_optimized):
            os.remove(temp_optimized)
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except:
                pass
            
        return output_path
        
    except Exception as e:
        # Clean up any temporary files
        if os.path.exists(temp_optimized):
            os.remove(temp_optimized)
        if os.path.exists(output_path):
            os.remove(output_path)
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except:
                pass
        raise Exception(f"Error converting to COG: {str(e)}")

def create_pyramid_tiles(input_path, output_dir):
    """Create pyramid tiles for web viewing"""
    try:
        with rasterio.open(input_path) as src:
            # Calculate zoom levels
            max_zoom = math.ceil(math.log2(max(src.width, src.height) / 256))
            
            for zoom in range(max_zoom + 1):
                # Calculate tile dimensions for this zoom level
                tile_size = 256
                scale = 2 ** (max_zoom - zoom)
                width = math.ceil(src.width / scale)
                height = math.ceil(src.height / scale)
                
                # Create zoom level directory
                zoom_dir = os.path.join(output_dir, str(zoom))
                os.makedirs(zoom_dir, exist_ok=True)
                
                # Create tiles
                for x in range(0, width, tile_size):
                    for y in range(0, height, tile_size):
                        window = rasterio.windows.Window(x, y, 
                                                       min(tile_size, width - x),
                                                       min(tile_size, height - y))
                        
                        with WarpedVRT(src,
                                      width=width,
                                      height=height,
                                      resampling=Resampling.average) as vrt:
                            data = vrt.read(window=window)
                            
                            if data.size > 0:  # Only save non-empty tiles
                                tile_path = os.path.join(zoom_dir, f"{x}_{y}.tif")
                                profile = src.profile.copy()
                                profile.update({
                                    'driver': 'GTiff',
                                    'height': data.shape[1],
                                    'width': data.shape[2],
                                    'transform': rasterio.windows.transform(window, src.transform)
                                })
                                
                                with rasterio.open(tile_path, 'w', **profile) as dst:
                                    dst.write(data)
                            
        return output_dir
        
    except Exception as e:
        raise Exception(f"Error creating pyramid tiles: {str(e)}")