from setuptools import setup, find_namespace_packages

setup(
    name="boem-webapp",
    version="0.1",
    packages=find_namespace_packages(include=["app", "app.*"]),
    install_requires=[
        "streamlit",
        "leafmap",
        "geopandas",
        "rasterio",
        "rio-cogeo"
    ]
) 