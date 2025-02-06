import streamlit as st
import leafmap.foliumap as leafmap
import os
import sqlite3
from folium import TileLayer
import json

def get_mbtiles_info(mbtiles_path):
    """Extract metadata from MBTiles file"""
    try:
        conn = sqlite3.connect(mbtiles_path)
        cursor = conn.cursor()
        
        # Get metadata
        cursor.execute("SELECT name, value FROM metadata")
        metadata = dict(cursor.fetchall())
        
        # Get bounds
        bounds = metadata.get('bounds', '-180,-85,180,85').split(',')
        bounds = [float(x) for x in bounds]
        
        # Get center
        center = metadata.get('center', '0,0,2').split(',')
        center = [float(center[1]), float(center[0])]  # lat, lon
        zoom = int(center[2]) if len(center) > 2 else 2
        
        return {
            'bounds': bounds,
            'center': center,
            'zoom': zoom,
            'name': metadata.get('name', 'MBTiles Layer'),
            'attribution': metadata.get('attribution', '')
        }
    except Exception as e:
        st.error(f"Error reading MBTiles metadata: {str(e)}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def app():
    st.title("WebMap Viewer")

    # File uploader for MBTiles
    uploaded_file = st.file_uploader(
        "Upload an MBTiles file", type=['mbtiles']
    )

    if uploaded_file:
        try:
            # Create directories if they don't exist
            mbtiles_dir = os.path.join("data", "raw", "mbtiles")
            os.makedirs(mbtiles_dir, exist_ok=True)
            
            # Save uploaded file
            mbtiles_path = os.path.join(mbtiles_dir, uploaded_file.name)
            with open(mbtiles_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Get MBTiles information
            mbtiles_info = get_mbtiles_info(mbtiles_path)
            
            if mbtiles_info:
                # Create map centered on the data
                m = leafmap.Map(
                    center=mbtiles_info['center'],
                    zoom=mbtiles_info['zoom']
                )
                
                # Add MBTiles as a TileLayer
                tile_url = f"mbtiles://{os.path.abspath(mbtiles_path)}"
                
                # Add base layer
                m.add_basemap("OpenStreetMap")
                
                # Add MBTiles layer using TileLayer
                tile_layer = TileLayer(
                    tiles=tile_url,
                    name=mbtiles_info['name'],
                    attr=mbtiles_info['attribution'],
                    overlay=True,
                    control=True
                )
                m.add_child(tile_layer)
                
                # Fit bounds if available
                if 'bounds' in mbtiles_info:
                    m.fit_bounds([
                        [mbtiles_info['bounds'][1], mbtiles_info['bounds'][0]],
                        [mbtiles_info['bounds'][3], mbtiles_info['bounds'][2]]
                    ])

                # Display map
                m.to_streamlit(height=600)
                
                # Add download button
                with open(mbtiles_path, 'rb') as f:
                    st.download_button(
                        label="Download processed MBTiles",
                        data=f,
                        file_name=uploaded_file.name,
                        mime="application/x-sqlite3"
                    )
                
            else:
                st.error("Could not read MBTiles metadata")
                
        except Exception as e:
            st.error(f"Error loading MBTiles: {str(e)}")
            
        finally:
            # Cleanup
            if os.path.exists(mbtiles_path):
                os.remove(mbtiles_path)

if __name__ == "__main__":
    app()
