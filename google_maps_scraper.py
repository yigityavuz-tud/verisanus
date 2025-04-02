from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import json
from typing import List, Dict
import logging
import os
from config import (
    HEADLESS_MODE,
    WINDOW_SIZE,
    WAIT_TIME,
    INITIAL_LOAD_DELAY,
    SCROLL_DELAY
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GoogleMapsScraper:
    def __init__(self):
        """Initialize the Google Maps scraper with Chrome WebDriver."""
        try:
            chrome_options = Options()
            if HEADLESS_MODE:
                chrome_options.add_argument("--headless=new")  # Updated headless mode syntax
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument(f"--window-size={WINDOW_SIZE}")
            chrome_options.add_argument("--disable-notifications")
            chrome_options.add_argument("--disable-popup-blocking")
            
            # Get the ChromeDriver path and ensure it's the executable
            driver_path = ChromeDriverManager().install()
            # Extract the directory path from the full path
            driver_dir = os.path.dirname(driver_path)
            # Construct the path to the chromedriver executable
            driver_executable = os.path.join(driver_dir, "chromedriver.exe")
            
            logger.info(f"ChromeDriver executable path: {driver_executable}")
            
            if not os.path.exists(driver_executable):
                raise FileNotFoundError(f"ChromeDriver executable not found at: {driver_executable}")
            
            # Create service object with the correct executable path
            service = Service(executable_path=driver_executable)
            
            # Initialize the driver
            self.driver = webdriver.Chrome(
                service=service,
                options=chrome_options
            )
            self.wait = WebDriverWait(self.driver, WAIT_TIME)
            logger.info("Chrome WebDriver initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Chrome WebDriver: {str(e)}")
            raise

    def handle_cookie_consent(self):
        """Handle the cookie consent popup if it appears."""
        try:
            # Wait for the cookie consent button to appear (if it does)
            cookie_button = self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//button[contains(., 'Reject all') or contains(., 'Decline all') or contains(., 'No, thanks')]"))
            )
            cookie_button.click()
            logger.info("Cookie consent popup handled successfully")
            time.sleep(2)  # Wait for the popup to disappear
        except Exception as e:
            logger.info("No cookie consent popup found or already handled")
            pass

    def scroll_results(self):
        """Scroll the results panel to load more places."""
        try:
            # Find the scrollable container
            scrollable = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='feed']"))
            )
            
            # Get the current scroll height
            last_height = self.driver.execute_script("return arguments[0].scrollHeight", scrollable)
            
            # Scroll to bottom
            self.driver.execute_script("arguments[0].scrollTo(0, arguments[0].scrollHeight);", scrollable)
            
            # Wait for new results to load
            time.sleep(SCROLL_DELAY)
            
            # Get new scroll height
            new_height = self.driver.execute_script("return arguments[0].scrollHeight", scrollable)
            
            return new_height != last_height
            
        except Exception as e:
            logger.error(f"Error while scrolling: {str(e)}")
            return False

    def extract_place_info(self, element) -> Dict:
        """Extract place information from an element."""
        try:
            # Get the link
            href = element.get_attribute('href')
            if not href or '/maps/place/' not in href:
                return None

            # Get the name (try different selectors)
            name = None
            try:
                # Try to find the name in the main element
                name = element.text.strip()
            except:
                pass

            if not name:
                try:
                    # Try to find the name in child elements
                    name_element = element.find_element(By.CSS_SELECTOR, "div[role='heading']")
                    name = name_element.text.strip()
                except:
                    pass

            if not name:
                try:
                    # Try to find the name in the aria-label
                    name = element.get_attribute('aria-label')
                except:
                    pass

            if not name:
                return None

            # Get rating and review count using the exact same approach as the working notebook
            rating = None
            review_count = None
            
            try:
                # First try to find the rating container using the exact XPath from the notebook
                rating_container = element.find_element(By.XPATH, ".//div[contains(@class, 'fontDisplayLarge')]")
                rating_text = rating_container.text
                if rating_text:
                    rating = float(rating_text.split()[0])
            except:
                pass

            try:
                # Try to find the review count using the exact XPath from the notebook
                review_container = element.find_element(By.XPATH, ".//div[contains(@class, 'fontBodyMedium')]//span[contains(text(), 'reviews')]")
                review_text = review_container.text
                if review_text:
                    # Extract number from text like "(1,234 reviews)"
                    review_count = int(''.join(filter(str.isdigit, review_text)))
            except:
                pass

            # If we still don't have rating/reviews, try alternative selectors
            if not rating or not review_count:
                try:
                    # Try to find rating in the parent container
                    rating_container = element.find_element(By.CSS_SELECTOR, "div[role='img']")
                    rating_text = rating_container.get_attribute('aria-label')
                    if rating_text and 'stars' in rating_text:
                        rating = float(rating_text.split()[0])
                except:
                    pass

                try:
                    # Try to find review count in the parent container
                    review_container = element.find_element(By.CSS_SELECTOR, "div[role='img'] + div")
                    review_text = review_container.text
                    if review_text and 'reviews' in review_text:
                        review_count = int(''.join(filter(str.isdigit, review_text)))
                except:
                    pass

            return {
                'name': name,
                'link': href,
                'rating': rating,
                'review_count': review_count
            }
        except Exception as e:
            logger.error(f"Error extracting place info: {str(e)}")
            return None

    def search_places(self, search_term: str, max_results: int = 10) -> List[Dict]:
        """
        Search for places on Google Maps and extract their information.
        
        Args:
            search_term (str): The term to search for
            max_results (int): Maximum number of results to collect
            
        Returns:
            List[Dict]: List of dictionaries containing place information
        """
        try:
            # Construct the Google Maps search URL
            search_url = f"https://www.google.com/maps/search/{search_term.replace(' ', '+')}"
            logger.info(f"Navigating to: {search_url}")
            self.driver.get(search_url)
            
            # Handle cookie consent popup if it appears
            self.handle_cookie_consent()
            
            # Wait for the results to load
            time.sleep(INITIAL_LOAD_DELAY)
            
            places = []
            scroll_attempts = 0
            max_scroll_attempts = 20  # Prevent infinite scrolling
            
            while len(places) < max_results and scroll_attempts < max_scroll_attempts:
                # Wait for place elements to be present
                self.wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/maps/place']"))
                )
                
                # Get all place elements
                place_elements = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/maps/place']")
                
                # Process each place element
                for element in place_elements:
                    try:
                        place_info = self.extract_place_info(element)
                        if place_info and place_info['link'] not in [p['link'] for p in places]:
                            places.append(place_info)
                            logger.info(f"Found place: {place_info['name']}")
                            
                            if len(places) >= max_results:
                                break
                    except Exception as e:
                        logger.error(f"Error processing place element: {str(e)}")
                        continue
                
                if len(places) >= max_results:
                    break
                
                # Try to scroll and load more results
                if not self.scroll_results():
                    scroll_attempts += 1
                    logger.info(f"No new results loaded after scroll. Attempt {scroll_attempts}/{max_scroll_attempts}")
                else:
                    scroll_attempts = 0  # Reset counter if we successfully loaded new results
            
            logger.info(f"Finished scraping. Found {len(places)} places.")
            return places[:max_results]
            
        except Exception as e:
            logger.error(f"Error during scraping: {str(e)}")
            return []
        
    def close(self):
        """Close the browser."""
        self.driver.quit() 