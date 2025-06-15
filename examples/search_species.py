"""
Example script demonstrating how to use the eBird Media Finder package.
"""

import os
import sys
from datetime import datetime
from ebird_media_finder import MediaFinder

def main():
    # Get API key from environment variable
    api_key = os.getenv("EBIRD_API_KEY")
    if not api_key:
        print("Error: EBIRD_API_KEY environment variable not set")
        sys.exit(1)
    
    # Initialize the MediaFinder
    finder = MediaFinder(api_key=api_key)
    
    # Example 1: Search for specific species in a region during May
    species_list = [
        "American Robin",
        "Blue Jay",
        "Northern Cardinal"
    ]
    
    print(f"\nSearching for {len(species_list)} species in New York during May...")
    results = finder.search_species(
        species_list=species_list,
        region="US-NY",
        month_range=(5, 5),
        media_type="audio",
        tag="song"
    )
    
    output_file = "species_search_results.csv"
    results.to_csv(output_file, index=False)
    print(f"Found {len(results)} media records. Results saved to {output_file}")
    
    # Example 2: Search recent observations in California
    print("\nSearching recent observations in California...")
    recent_results = finder.search_recent_observations(
        region="US-CA",
        days_back=7,
        media_type="audio"
    )
    
    output_file = f"recent_observations_{datetime.now().strftime('%Y%m%d')}.csv"
    recent_results.to_csv(output_file, index=False)
    print(f"Found {len(recent_results)} media records. Results saved to {output_file}")

if __name__ == "__main__":
    main()