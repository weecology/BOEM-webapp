"""
Main implementation of the MediaFinder class for searching eBird media.
"""

import os
from typing import List, Tuple, Optional, Dict, Any
import requests
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import urlencode

class MediaFinder:
    """A class to find and retrieve media records from the Macaulay Library via eBird."""
    
    EBIRD_API_BASE = "https://api.ebird.org/v2"
    MEDIA_SEARCH_BASE = "https://media.ebird.org/catalog"
    TAXONOMY_URL = "https://www.birds.cornell.edu/clementschecklist/wp-content/uploads/2024/02/eBird_Clements_v2024_01Jan2024.csv"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the MediaFinder.
        
        Args:
            api_key: eBird API key. If not provided, will try to get from EBIRD_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("EBIRD_API_KEY")
        if not self.api_key:
            raise ValueError("eBird API key must be provided or set in EBIRD_API_KEY environment variable")
        
        # Load taxonomy data
        self._taxonomy_df = None
    
    @property
    def taxonomy_df(self) -> pd.DataFrame:
        """Lazy load the taxonomy data when needed."""
        if self._taxonomy_df is None:
            self._taxonomy_df = pd.read_csv(self.TAXONOMY_URL)
        return self._taxonomy_df
    
    def get_species_code(self, species_name: str) -> str:
        """
        Get the eBird species code for a given species name.
        
        Args:
            species_name: Common name or scientific name of the species
            
        Returns:
            eBird species code
            
        Raises:
            ValueError: If species not found in taxonomy
        """
        species_match = self.taxonomy_df[
            (self.taxonomy_df["COMMON_NAME"].str.lower() == species_name.lower()) |
            (self.taxonomy_df["SCI_NAME"].str.lower() == species_name.lower())
        ]
        
        if len(species_match) == 0:
            raise ValueError(f"Species '{species_name}' not found in taxonomy")
            
        return species_match.iloc[0]["SPECIES_CODE"]
    
    def search_species(
        self,
        species_list: List[str],
        region: str,
        month_range: Optional[Tuple[int, int]] = None,
        media_type: str = "audio",
        tag: Optional[str] = "song"
    ) -> pd.DataFrame:
        """
        Search for media records for given species.
        
        Args:
            species_list: List of species names (common or scientific)
            region: eBird region code (e.g., "US-NY")
            month_range: Optional tuple of (start_month, end_month)
            media_type: Type of media to search for ("audio" or "video")
            tag: Optional media tag to filter by
            
        Returns:
            DataFrame with media records
        """
        results = []
        
        for species in species_list:
            try:
                species_code = self.get_species_code(species)
                
                params = {
                    "view": "list",
                    "all": "true",
                    "taxonCode": species_code,
                    "mediaType": media_type,
                    "regionCode": region
                }
                
                if tag:
                    params["tag"] = tag
                    
                if month_range:
                    params["beginMonth"] = month_range[0]
                    params["endMonth"] = month_range[1]
                
                url = f"{self.MEDIA_SEARCH_BASE}?{urlencode(params)}"
                response = requests.get(url)
                response.raise_for_status()
                
                # Parse the response and extract catalog IDs
                # Note: This is a placeholder as we need to implement proper HTML parsing
                # or use the actual API endpoint when available
                catalog_ids = []  # TODO: Implement extraction of catalog IDs
                
                for catalog_id in catalog_ids:
                    results.append({
                        "species": species,
                        "species_code": species_code,
                        "region": region,
                        "media_type": media_type,
                        "catalog_id": catalog_id,
                        "url": f"https://macaulaylibrary.org/asset/{catalog_id}"
                    })
                    
            except Exception as e:
                print(f"Error processing {species}: {str(e)}")
                
        return pd.DataFrame(results)
    
    def search_recent_observations(
        self,
        region: str,
        days_back: int = 7,
        media_type: str = "audio"
    ) -> pd.DataFrame:
        """
        Search for recent observations with media in a region.
        
        Args:
            region: eBird region code
            days_back: Number of days to look back
            media_type: Type of media to search for
            
        Returns:
            DataFrame with media records
        """
        url = f"{self.EBIRD_API_BASE}/data/obs/{region}/recent"
        headers = {"X-eBirdApiToken": self.api_key}
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        observations = response.json()
        species_list = list(set(obs["comName"] for obs in observations))
        
        return self.search_species(
            species_list=species_list,
            region=region,
            media_type=media_type
        )