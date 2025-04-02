# Google Maps Scraper Configuration

# Search settings
SEARCH_TERM = "istanbul saç ekimi"  # The term to search for
MAX_RESULTS = 10  # Maximum number of results to collect

# Browser settings
HEADLESS_MODE = False  # Whether to run Chrome in headless mode (set to False for debugging)
WINDOW_SIZE = "1920,1080"  # Browser window size
WAIT_TIME = 20  # Maximum wait time for elements in seconds

# Scraping settings
INITIAL_LOAD_DELAY = 5  # Initial delay after page load in seconds
SCROLL_DELAY = 3  # Delay between scrolls in seconds
MAX_SCROLL_ATTEMPTS = 5  # Maximum number of scroll attempts

# Output settings
OUTPUT_DIR = "results"  # Directory to save results
EXPORT_TO_EXCEL = True  # Whether to export results to Excel
EXCEL_SHEET_NAME = "Places"  # Name of the Excel sheet

# Debug settings
DEBUG_MODE = True  # Enable debug logging
SAVE_SCREENSHOTS = False  # Save screenshots on error (for debugging) 