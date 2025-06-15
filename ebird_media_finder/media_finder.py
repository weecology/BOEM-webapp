"""
Main implementation of the MediaFinder class for searching eBird media.
"""

import os
from typing import List, Tuple, Optional, Dict, Any, Set
import requests
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import urlencode
from bs4 import BeautifulSoup
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MediaFinder:
    """A class to find and retrieve media records from the Macaulay Library via eBird."""
    
    EBIRD_API_BASE = "https://api.ebird.org/v2"
    MEDIA_SEARCH_BASE = "https://media.ebird.org/catalog"
    TAXONOMY_URL = "https://www.birds.cornell.edu/clementschecklist/wp-content/uploads/2024/02/eBird_Clements_v2024_01Jan2024.csv"
    ML_ASSET_BASE = "https://macaulaylibrary.org/asset"
    
    def __init__(self, api_key: Optional[str] = None, max_workers: int = 4):
        """
        Initialize the MediaFinder.
        
        Args:
            api_key: eBird API key. If not provided, will try to get from EBIRD_API_KEY environment variable.
            max_workers: Maximum number of concurrent threads for parallel processing.
        """
        self.api_key = api_key or os.getenv("EBIRD_API_KEY")
        if not self.api_key:
            raise ValueError("eBird API key must be provided or set in EBIRD_API_KEY environment variable")
        
        self.max_workers = max_workers
        self._taxonomy_df = None
        self._session = requests.Session()
    
    @property
    def taxonomy_df(self) -> pd.DataFrame:
        """Lazy load the taxonomy data when needed."""
        if self._taxonomy_df is None:
            logger.info("Loading taxonomy data...")
            self._taxonomy_df = pd.read_csv(self.TAXONOMY_URL)
            # Clean up taxonomy data
            self._taxonomy_df["COMMON_NAME"] = self._taxonomy_df["COMMON_NAME"].str.strip()
            self._taxonomy_df["SCI_NAME"] = self._taxonomy_df["SCI_NAME"].str.strip()
            self._taxonomy_df["SPECIES_CODE"] = self._taxonomy_df["SPECIES_CODE"].str.strip()
            logger.info("Taxonomy data loaded successfully")
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
        species_name = species_name.strip()
        species_match = self.taxonomy_df[
            (self.taxonomy_df["COMMON_NAME"].str.lower() == species_name.lower()) |
            (self.taxonomy_df["SCI_NAME"].str.lower() == species_name.lower())
        ]
        
        if len(species_match) == 0:
            # Try fuzzy matching
            species_match = self.taxonomy_df[
                (self.taxonomy_df["COMMON_NAME"].str.lower().str.contains(species_name.lower())) |
                (self.taxonomy_df["SCI_NAME"].str.lower().str.contains(species_name.lower()))
            ]
            
            if len(species_match) == 1:
                logger.warning(f"Using fuzzy match for '{species_name}': {species_match.iloc[0]['COMMON_NAME']}")
                return species_match.iloc[0]["SPECIES_CODE"]
            elif len(species_match) > 1:
                matches = species_match["COMMON_NAME"].tolist()
                raise ValueError(f"Multiple matches found for '{species_name}': {matches}")
            else:
                raise ValueError(f"Species '{species_name}' not found in taxonomy")
            
        return species_match.iloc[0]["SPECIES_CODE"]
    
    def _extract_catalog_ids(self, html_content: str) -> Set[str]:
        """
        Extract catalog IDs from the Macaulay Library search results page.
        
        Args:
            html_content: HTML content of the search results page
            
        Returns:
            Set of catalog IDs
        """
        soup = BeautifulSoup(html_content, 'lxml')
        catalog_ids = set()
        
        # Look for catalog IDs in various HTML elements
        # Media cards
        for card in soup.find_all(class_=lambda x: x and 'MediaCard' in x):
            if card.get('data-asset-id'):
                catalog_ids.add(card['data-asset-id'])
        
        # Links to assets
        for link in soup.find_all('a', href=True):
            href = link['href']
            if '/asset/' in href:
                catalog_id = href.split('/asset/')[-1].split('?')[0].split('#')[0]
                if catalog_id.isdigit():
                    catalog_ids.add(catalog_id)
        
        return catalog_ids
    
    def _get_media_metadata(self, catalog_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific media asset.
        
        Args:
            catalog_id: Macaulay Library catalog ID
            
        Returns:
            Dictionary containing media metadata
        """
        url = f"{self.ML_ASSET_BASE}/{catalog_id}"
        try:
            response = self._session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            
            metadata = {
                "catalog_id": catalog_id,
                "url": url,
                "title": None,
                "recordist": None,
                "date": None,
                "time": None,
                "location": None,
                "quality_rating": None,
                "file_type": None,
                "duration": None
            }
            
            # Extract metadata from the page
            title_elem = soup.find(class_=lambda x: x and 'AssetHeader' in x)
            if title_elem:
                metadata["title"] = title_elem.get_text(strip=True)
            
            # Extract other metadata fields
            metadata_section = soup.find(class_=lambda x: x and 'AssetMetadata' in x)
            if metadata_section:
                for row in metadata_section.find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        key = cells[0].get_text(strip=True).lower().replace(' ', '_')
                        value = cells[1].get_text(strip=True)
                        metadata[key] = value
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error fetching metadata for catalog ID {catalog_id}: {str(e)}")
            return {"catalog_id": catalog_id, "url": url, "error": str(e)}
    
    def search_species(
        self,
        species_list: List[str],
        region: str,
        month_range: Optional[Tuple[int, int]] = None,
        media_type: str = "audio",
        tag: Optional[str] = "song",
        fetch_metadata: bool = True
    ) -> pd.DataFrame:
        """
        Search for media records for given species.
        
        Args:
            species_list: List of species names (common or scientific)
            region: eBird region code (e.g., "US-NY")
            month_range: Optional tuple of (start_month, end_month)
            media_type: Type of media to search for ("audio" or "video")
            tag: Optional media tag to filter by
            fetch_metadata: Whether to fetch detailed metadata for each media asset
            
        Returns:
            DataFrame with media records
        """
        results = []
        
        for species in species_list:
            try:
                logger.info(f"Processing species: {species}")
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
                response = self._session.get(url)
                response.raise_for_status()
                
                catalog_ids = self._extract_catalog_ids(response.text)
                logger.info(f"Found {len(catalog_ids)} media records for {species}")
                
                if fetch_metadata:
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        future_to_id = {
                            executor.submit(self._get_media_metadata, cid): cid 
                            for cid in catalog_ids
                        }
                        
                        for future in as_completed(future_to_id):
                            metadata = future.result()
                            metadata.update({
                                "species": species,
                                "species_code": species_code,
                                "region": region,
                                "media_type": media_type,
                                "search_tag": tag,
                                "search_url": url
                            })
                            results.append(metadata)
                else:
                    for catalog_id in catalog_ids:
                        results.append({
                            "species": species,
                            "species_code": species_code,
                            "region": region,
                            "media_type": media_type,
                            "catalog_id": catalog_id,
                            "url": f"{self.ML_ASSET_BASE}/{catalog_id}",
                            "search_tag": tag,
                            "search_url": url
                        })
                    
            except Exception as e:
                logger.error(f"Error processing {species}: {str(e)}")
                results.append({
                    "species": species,
                    "error": str(e)
                })
                
        return pd.DataFrame(results)
    
    def search_recent_observations(
        self,
        region: str,
        days_back: int = 7,
        media_type: str = "audio",
        fetch_metadata: bool = True
    ) -> pd.DataFrame:
        """
        Search for recent observations with media in a region.
        
        Args:
            region: eBird region code
            days_back: Number of days to look back
            media_type: Type of media to search for
            fetch_metadata: Whether to fetch detailed metadata for each media asset
            
        Returns:
            DataFrame with media records
        """
        url = f"{self.EBIRD_API_BASE}/data/obs/{region}/recent"
        headers = {"X-eBirdApiToken": self.api_key}
        
        response = self._session.get(url, headers=headers)
        response.raise_for_status()
        
        observations = response.json()
        species_list = list(set(obs["comName"] for obs in observations))
        logger.info(f"Found {len(species_list)} unique species in recent observations")
        
        return self.search_species(
            species_list=species_list,
            region=region,
            media_type=media_type,
            fetch_metadata=fetch_metadata
        )