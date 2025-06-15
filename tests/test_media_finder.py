"""
Tests for the MediaFinder class.
"""

import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from ebird_media_finder import MediaFinder

@pytest.fixture
def mock_taxonomy_data():
    return pd.DataFrame({
        "COMMON_NAME": ["American Robin", "Blue Jay", "Northern Cardinal"],
        "SCI_NAME": ["Turdus migratorius", "Cyanocitta cristata", "Cardinalis cardinalis"],
        "SPECIES_CODE": ["amerob", "blujay", "norcar"]
    })

@pytest.fixture
def mock_html_content():
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
def mock_asset_page():
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
def media_finder():
    with patch.dict(os.environ, {"EBIRD_API_KEY": "test_key"}):
        return MediaFinder()

def test_init_without_api_key():
    with patch.dict(os.environ, clear=True):
        with pytest.raises(ValueError, match="eBird API key must be provided"):
            MediaFinder()

def test_init_with_explicit_api_key():
    finder = MediaFinder(api_key="test_key")
    assert finder.api_key == "test_key"

def test_get_species_code(media_finder, mock_taxonomy_data):
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

def test_extract_catalog_ids(media_finder, mock_html_content):
    catalog_ids = media_finder._extract_catalog_ids(mock_html_content)
    assert catalog_ids == {"123456", "789012", "345678"}

def test_get_media_metadata(media_finder, mock_asset_page):
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

def test_search_species(media_finder, mock_taxonomy_data, mock_html_content, mock_asset_page):
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

def test_search_recent_observations(media_finder, mock_taxonomy_data, mock_html_content):
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