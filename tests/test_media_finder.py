"""
Tests for the MediaFinder class.

This module contains comprehensive tests for the MediaFinder class, including:
- Initialization and configuration
- Species code lookup and matching
- Media search functionality
- Metadata extraction
- Error handling
"""

import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from ebird_media_finder import MediaFinder

@pytest.fixture
def mock_taxonomy_data() -> pd.DataFrame:
    """
    Create a mock taxonomy DataFrame for testing.
    
    Returns:
        DataFrame containing mock taxonomy data with common names,
        scientific names, and species codes for testing.
    """
    return pd.DataFrame({
        "COMMON_NAME": ["American Robin", "Blue Jay", "Northern Cardinal"],
        "SCI_NAME": ["Turdus migratorius", "Cyanocitta cristata", "Cardinalis cardinalis"],
        "SPECIES_CODE": ["amerob", "blujay", "norcar"]
    })

@pytest.fixture
def mock_html_content() -> str:
    """
    Create mock HTML content simulating a Macaulay Library search results page.
    
    Returns:
        String containing HTML with media cards and asset links for testing
        the catalog ID extraction functionality.
    """
    return """
    <div class="MediaCard" data-asset-id="123456">
        <a href="/asset/123456">Media 1</a>
    </div>
    <div class="MediaCard" data-asset-id="789012">
        <a href="/asset/789012">Media 2</a>
    </div>
    <a href="/asset/345678">Media 3</a>
    """

@pytest.fixture
def mock_asset_page() -> str:
    """
    Create mock HTML content simulating a Macaulay Library asset page.
    
    Returns:
        String containing HTML with metadata for a specific media asset,
        including title, recordist, date, time, location, and quality rating.
    """
    return """
    <div class="AssetHeader">American Robin Song</div>
    <div class="AssetMetadata">
        <table>
            <tr>
                <td>Recordist</td>
                <td>John Doe</td>
            </tr>
            <tr>
                <td>Date</td>
                <td>15 May 2024</td>
            </tr>
            <tr>
                <td>Time</td>
                <td>06:30</td>
            </tr>
            <tr>
                <td>Location</td>
                <td>Central Park, New York, NY</td>
            </tr>
            <tr>
                <td>Quality Rating</td>
                <td>4.5</td>
            </tr>
        </table>
    </div>
    """

@pytest.fixture
def media_finder() -> MediaFinder:
    """
    Create a MediaFinder instance with a test API key.
    
    Returns:
        Configured MediaFinder instance for testing.
    """
    with patch.dict(os.environ, {"EBIRD_API_KEY": "test_key"}):
        return MediaFinder()

def test_init_without_api_key() -> None:
    """
    Test that MediaFinder initialization fails without an API key.
    
    Verifies that attempting to create a MediaFinder instance without
    providing an API key or setting the EBIRD_API_KEY environment
    variable raises a ValueError with an appropriate message.
    """
    with patch.dict(os.environ, clear=True):
        with pytest.raises(ValueError, match="eBird API key must be provided"):
            MediaFinder()

def test_init_with_explicit_api_key() -> None:
    """
    Test MediaFinder initialization with an explicit API key.
    
    Verifies that the MediaFinder can be initialized with an API key
    provided directly to the constructor, and that the key is properly stored.
    """
    finder = MediaFinder(api_key="test_key")
    assert finder.api_key == "test_key"

def test_get_species_code(media_finder: MediaFinder, mock_taxonomy_data: pd.DataFrame) -> None:
    """
    Test species code lookup functionality.
    
    Tests various aspects of species code lookup:
    - Exact matches with common and scientific names
    - Case-insensitive matching
    - Fuzzy matching for partial names
    - Error handling for ambiguous matches
    - Error handling for invalid species names
    
    Args:
        media_finder: Configured MediaFinder instance
        mock_taxonomy_data: Mock taxonomy DataFrame
    """
    with patch.object(media_finder, "taxonomy_df", mock_taxonomy_data):
        # Test exact matches
        assert media_finder.get_species_code("American Robin") == "amerob"
        assert media_finder.get_species_code("Turdus migratorius") == "amerob"
        
        # Test case insensitivity
        assert media_finder.get_species_code("american robin") == "amerob"
        assert media_finder.get_species_code("BLUE JAY") == "blujay"
        
        # Test fuzzy matching
        assert media_finder.get_species_code("Robin") == "amerob"
        
        # Test multiple matches error
        with pytest.raises(ValueError, match="Multiple matches found"):
            media_finder.get_species_code("Cardinal")
        
        # Test no match error
        with pytest.raises(ValueError, match="not found in taxonomy"):
            media_finder.get_species_code("Invalid Bird")

def test_extract_catalog_ids(media_finder: MediaFinder, mock_html_content: str) -> None:
    """
    Test extraction of catalog IDs from HTML content.
    
    Verifies that the _extract_catalog_ids method correctly extracts
    catalog IDs from both MediaCard elements and asset links in the HTML.
    
    Args:
        media_finder: Configured MediaFinder instance
        mock_html_content: Mock HTML content with catalog IDs
    """
    catalog_ids = media_finder._extract_catalog_ids(mock_html_content)
    assert catalog_ids == {"123456", "789012", "345678"}

def test_get_media_metadata(media_finder: MediaFinder, mock_asset_page: str) -> None:
    """
    Test extraction of metadata from a media asset page.
    
    Verifies that the _get_media_metadata method correctly extracts
    all metadata fields from an asset page's HTML content.
    
    Args:
        media_finder: Configured MediaFinder instance
        mock_asset_page: Mock HTML content for an asset page
    """
    with patch.object(media_finder._session, "get") as mock_get:
        mock_response = MagicMock()
        mock_response.text = mock_asset_page
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        metadata = media_finder._get_media_metadata("123456")
        
        assert metadata["catalog_id"] == "123456"
        assert metadata["title"] == "American Robin Song"
        assert metadata["recordist"] == "John Doe"
        assert metadata["date"] == "15 May 2024"
        assert metadata["time"] == "06:30"
        assert metadata["location"] == "Central Park, New York, NY"
        assert metadata["quality_rating"] == "4.5"

def test_search_species(
    media_finder: MediaFinder,
    mock_taxonomy_data: pd.DataFrame,
    mock_html_content: str,
    mock_asset_page: str
) -> None:
    """
    Test the species search functionality.
    
    Verifies that the search_species method:
    - Correctly constructs search URLs
    - Handles the search response
    - Extracts catalog IDs
    - Fetches metadata for each asset
    - Returns properly formatted results
    
    Args:
        media_finder: Configured MediaFinder instance
        mock_taxonomy_data: Mock taxonomy data
        mock_html_content: Mock search results HTML
        mock_asset_page: Mock asset page HTML
    """
    with patch.object(media_finder, "taxonomy_df", mock_taxonomy_data):
        with patch.object(media_finder._session, "get") as mock_get:
            # Mock the search results page
            mock_search_response = MagicMock()
            mock_search_response.text = mock_html_content
            mock_search_response.raise_for_status.return_value = None
            
            # Mock the asset page
            mock_asset_response = MagicMock()
            mock_asset_response.text = mock_asset_page
            mock_asset_response.raise_for_status.return_value = None
            
            mock_get.side_effect = [mock_search_response] + [mock_asset_response] * 3
            
            result = media_finder.search_species(
                species_list=["American Robin"],
                region="US-NY",
                month_range=(5, 5)
            )
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3  # Three catalog IDs from mock_html_content
            assert all(result["species"] == "American Robin")
            assert all(result["species_code"] == "amerob")
            assert all(result["region"] == "US-NY")
            
            # Verify URL parameters in the search request
            search_call = mock_get.call_args_list[0]
            search_url = search_call[0][0]
            assert "taxonCode=amerob" in search_url
            assert "regionCode=US-NY" in search_url
            assert "beginMonth=5" in search_url
            assert "endMonth=5" in search_url

def test_search_recent_observations(
    media_finder: MediaFinder,
    mock_taxonomy_data: pd.DataFrame,
    mock_html_content: str
) -> None:
    """
    Test the recent observations search functionality.
    
    Verifies that the search_recent_observations method:
    - Correctly calls the eBird API
    - Processes the API response
    - Initiates species searches
    - Returns properly formatted results
    
    Args:
        media_finder: Configured MediaFinder instance
        mock_taxonomy_data: Mock taxonomy data
        mock_html_content: Mock search results HTML
    """
    with patch.object(media_finder, "taxonomy_df", mock_taxonomy_data):
        with patch.object(media_finder._session, "get") as mock_get:
            # Mock the eBird API response
            mock_api_response = MagicMock()
            mock_api_response.json.return_value = [
                {"comName": "American Robin"},
                {"comName": "Blue Jay"}
            ]
            mock_api_response.raise_for_status.return_value = None
            
            # Mock the search results page
            mock_search_response = MagicMock()
            mock_search_response.text = mock_html_content
            mock_search_response.raise_for_status.return_value = None
            
            mock_get.side_effect = [mock_api_response] + [mock_search_response] * 2
            
            result = media_finder.search_recent_observations(
                region="US-NY",
                fetch_metadata=False
            )
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 6  # 2 species * 3 catalog IDs from mock_html_content
            assert set(result["species"].unique()) == {"American Robin", "Blue Jay"}
            assert all(result["region"] == "US-NY")
            
            # Verify the eBird API call
            api_call = mock_get.call_args_list[0]
            assert api_call[0][0] == "https://api.ebird.org/v2/data/obs/US-NY/recent"
            assert api_call[1]["headers"]["X-eBirdApiToken"] == "test_key"