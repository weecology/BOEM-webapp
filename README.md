# eBird Media Finder

A Python tool to automate finding media records from the Macaulay Library based on species, location, and time criteria.

## Features

- Search for bird media (audio/video) using the eBird API and Macaulay Library
- Support for species list input or direct eBird API queries
- Generate CSV reports with media catalog IDs and metadata
- Filter by location, date range, and media type
- Automated taxonomy lookup using the Clements Checklist

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

1. Get an eBird API key from https://ebird.org/api/keygen
2. Set your API key as an environment variable:
```bash
export EBIRD_API_KEY=your_key_here
```

## Usage

### Using a species list:
```python
from ebird_media_finder import MediaFinder

finder = MediaFinder()
results = finder.search_species(
    species_list=["American Robin", "Blue Jay"],
    region="US-NY",
    month_range=(5, 5)  # May only
)
results.to_csv("media_results.csv")
```

### Using eBird API for recent observations:
```python
results = finder.search_recent_observations(
    region="US-NY",
    days_back=7
)
results.to_csv("recent_observations.csv")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

