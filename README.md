# Google Maps Scraper

This Python script allows you to search for establishments on Google Maps and extract their information, including names and links.

## Project Structure

```
.
├── config.py              # Configuration settings
├── google_maps_scraper.py # Core scraper implementation
├── main.py               # Main script to run the scraper
├── requirements.txt      # Project dependencies
└── results/             # Directory for output files
```

## Prerequisites

- Python 3.7 or higher
- Chrome browser installed
- Internet connection

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to customize the scraper settings:

```python
# Search settings
SEARCH_TERM = "restaurants in New York"  # The term to search for
MAX_RESULTS = 10  # Maximum number of results to collect

# Browser settings
HEADLESS_MODE = True  # Whether to run Chrome in headless mode
WINDOW_SIZE = "1920,1080"  # Browser window size
WAIT_TIME = 10  # Maximum wait time for elements in seconds

# Scraping settings
INITIAL_LOAD_DELAY = 3  # Initial delay after page load in seconds
SCROLL_DELAY = 2  # Delay between scrolls in seconds

# Output settings
OUTPUT_DIR = "results"  # Directory to save results
```

## Usage

Run the main script:
```bash
python main.py
```

The script will use the settings from `config.py` to:
1. Search for the specified term on Google Maps
2. Collect the specified number of results
3. Save the results to a JSON file in the `results` directory

## Features

- Configuration-based settings
- Headless browser operation (configurable)
- Automatic scrolling to load more results
- Duplicate detection to avoid repeated entries
- Error handling and logging
- Results saved in JSON format
- Organized output directory structure

## Output Format

The script generates a JSON file with the following structure:
```json
[
  {
    "name": "Establishment Name",
    "link": "https://www.google.com/maps/place/..."
  },
  ...
]
```

## Notes

- The script includes appropriate delays to avoid overwhelming Google's servers
- It's recommended to use reasonable search terms and result limits
- The script runs in headless mode by default (configurable in config.py)
- Make sure you have a stable internet connection while running the script
- Results are saved in the `results` directory (configurable in config.py) 