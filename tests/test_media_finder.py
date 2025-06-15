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
        "COMMON_NAME": ["American Robin", "Blue Jay"],
        "SCI_NAME": ["Turdus migratorius", "Cyanocitta cristata"],
        "SPECIES_CODE": ["amerob", "blujay"]
    })

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
        assert media_finder.get_species_code("American Robin") == "amerob"
        assert media_finder.get_species_code("Turdus migratorius") == "amerob"
        
        with pytest.raises(ValueError, match="Species 'Invalid Bird' not found"):
            media_finder.get_species_code("Invalid Bird")

def test_search_species(media_finder, mock_taxonomy_data):
    with patch.object(media_finder, "taxonomy_df", mock_taxonomy_data):
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            result = media_finder.search_species(
                species_list=["American Robin"],
                region="US-NY",
                month_range=(5, 5)
            )
            
            assert isinstance(result, pd.DataFrame)
            assert mock_get.called
            
            # Verify the URL parameters
            call_args = mock_get.call_args[0][0]
            assert "taxonCode=amerob" in call_args
            assert "regionCode=US-NY" in call_args
            assert "beginMonth=5" in call_args
            assert "endMonth=5" in call_args

def test_search_recent_observations(media_finder):
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [
            {"comName": "American Robin"},
            {"comName": "Blue Jay"}
        ]
        mock_get.return_value = mock_response
        
        with patch.object(media_finder, "search_species") as mock_search:
            mock_search.return_value = pd.DataFrame()
            
            result = media_finder.search_recent_observations("US-NY")
            
            assert isinstance(result, pd.DataFrame)
            mock_search.assert_called_once_with(
                species_list=["American Robin", "Blue Jay"],
                region="US-NY",
                media_type="audio"
            )