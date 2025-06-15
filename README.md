# eBird Media Finder

A Python tool to automate finding media records from the Macaulay Library based on species, location, and time criteria.

## Features

- Search for bird media (audio/video) using the eBird API and Macaulay Library
- Support for species list input or direct eBird API queries
- Generate CSV reports with media catalog IDs and metadata
- Filter by location, date range, and media type
- Automated taxonomy lookup using the Clements Checklist
- Parallel processing for efficient metadata retrieval
- Fuzzy species name matching
- Comprehensive error handling and logging

## Installation

### Prerequisites

- Python 3.8 or higher
- eBird API key (get one from https://ebird.org/api/keygen)

### Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/ebird-media-finder.git
cd ebird-media-finder

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Set your eBird API key as an environment variable:
```bash
# On Linux/macOS
export EBIRD_API_KEY=your_key_here

# On Windows (Command Prompt)
set EBIRD_API_KEY=your_key_here

# On Windows (PowerShell)
$env:EBIRD_API_KEY = "your_key_here"
```

2. (Optional) Configure logging:
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Usage

### Basic Usage

```python
from ebird_media_finder import MediaFinder

# Initialize the finder
finder = MediaFinder()

# Search for specific species
results = finder.search_species(
    species_list=["American Robin", "Blue Jay"],
    region="US-NY",
    month_range=(5, 5)  # May only
)
results.to_csv("media_results.csv")

# Search recent observations
recent_results = finder.search_recent_observations(
    region="US-NY",
    days_back=7
)
recent_results.to_csv("recent_observations.csv")
```

### Advanced Usage

```python
# Search with more options
results = finder.search_species(
    species_list=["Turdus migratorius", "Cyanocitta cristata"],  # Scientific names work too
    region="US-NY",
    month_range=(5, 5),
    media_type="audio",
    tag="song",
    fetch_metadata=True  # Get detailed metadata for each recording
)

# Customize parallel processing
finder = MediaFinder(max_workers=8)  # Increase concurrent requests
```

## Output Format

The CSV output includes the following columns:

```csv
species,species_code,region,media_type,catalog_id,url,title,recordist,date,time,location,quality_rating,file_type,duration,search_tag,search_url
American Robin,amerob,US-NY,audio,123456,https://macaulaylibrary.org/asset/123456,American Robin Song,John Doe,15 May 2024,06:30,Central Park NY,4.5,MP3,02:30,song,https://media.ebird.org/catalog?...
```

Key columns:
- `species`: Common name of the species
- `species_code`: eBird species code
- `region`: eBird region code
- `media_type`: Type of media (audio/video)
- `catalog_id`: Unique Macaulay Library identifier
- `url`: Direct link to the media asset
- `title`: Title of the recording
- `recordist`: Name of the person who made the recording
- `date`: Recording date
- `time`: Recording time
- `location`: Recording location
- `quality_rating`: Quality rating (if available)
- `file_type`: Media file format
- `duration`: Length of the recording
- `search_tag`: Tag used in the search (e.g., "song")
- `search_url`: URL used to find the media

## Error Handling

The package implements comprehensive error handling:

1. **API Key Errors**:
   - Missing API key raises `ValueError`
   - Invalid API key results in HTTP 401 error

2. **Species Lookup Errors**:
   - Invalid species name raises `ValueError`
   - Multiple matches found raises `ValueError` with matching species
   - Fuzzy matching is attempted before failing

3. **Network Errors**:
   - Connection errors are caught and logged
   - Rate limiting is handled with exponential backoff
   - Timeout errors include the URL being accessed

4. **Data Parsing Errors**:
   - HTML parsing errors are caught and logged
   - Missing metadata fields are set to None
   - Invalid data formats are handled gracefully

Example error handling:

```python
try:
    results = finder.search_species(["Invalid Bird"], "US-NY")
except ValueError as e:
    print(f"Species error: {e}")
except requests.exceptions.RequestException as e:
    print(f"Network error: {e}")
```

## Logging

The package uses Python's built-in logging module. Default logging level is INFO.

### Configuring Logging

```python
import logging

# Basic configuration
logging.basicConfig(level=logging.INFO)

# Advanced configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ebird_media_finder.log'),
        logging.StreamHandler()
    ]
)
```

### Log Levels Used

- DEBUG: Detailed information for debugging
- INFO: General information about progress
- WARNING: Unexpected behavior that doesn't affect operation
- ERROR: Errors that prevent normal operation
- CRITICAL: Critical errors that require immediate attention

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the Repository**
   - Create a personal fork of the project
   - Clone your fork locally

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

4. **Run Tests**
   ```bash
   # Install development dependencies
   pip install -r requirements.txt
   
   # Run tests
   pytest tests/
   
   # Run linting
   black .
   flake8 .
   mypy ebird_media_finder
   ```

5. **Submit a Pull Request**
   - Push changes to your fork
   - Create a Pull Request with a clear description
   - Reference any related issues

### Code Style Guidelines

- Follow PEP 8
- Use type hints
- Write docstrings in Google format
- Keep functions focused and small
- Add tests for new features

### Reporting Issues

When reporting issues, please include:
- Python version
- Package version
- Complete error traceback
- Minimal code example to reproduce the issue
- Expected vs actual behavior

## License

This project is licensed under the MIT License - see the LICENSE file for details.

