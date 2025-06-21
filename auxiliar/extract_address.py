from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import csv

base_url = "https://www.codigo-postal.pt/torres-vedras/"
output = []

def setup_driver():
    chrome_options = Options()
    # Remove headless mode to see what's happening
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # Add additional options to help with Chrome startup issues
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-images")
    chrome_options.add_argument("--disable-javascript")
    chrome_options.add_argument("--remote-debugging-port=9222")
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    except Exception as e:
        print(f"Error setting up Chrome driver: {e}")
        # Fallback: try without service
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            return driver
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            raise

def extract_data_from_page(driver):
    # Each address block is inside a div with class 'cp_block'
    
    try:
        # Click on the accept cookies button
        accept_button = driver.find_element(By.XPATH, "/html/body/div[7]/div/div/div[2]/div[2]/div[1]/button[1]")
        accept_button.click()
        time.sleep(1)  # Wait for the modal to close
    except Exception as e:
        print(f"Could not find or click accept cookies button: {e}")
        try:
            # Click on the accept cookies button
            accept_button = driver.find_element(By.XPATH, "/html/body/div[7]/div[1]/div/div[2]/div[2]/div[1]/button[1]")
            accept_button.click()
            time.sleep(1)  # Wait for the modal to close
        except Exception as e:
            print(f"Could not find or click accept cookies button: {e}")
        
    blocks = driver.find_elements(By.XPATH, "/html/body/div[4]/div/div/div[2]/div/p")
    if len(blocks) == 0:
        blocks = driver.find_elements(By.XPATH, "/html/body/div[4]/div/div/div[2]/div/div[2]/div/p")
    for block in blocks:
        try:
            # Address
            address_elem = block.find_element(By.XPATH, ".//span[3]")
            address = address_elem.text.strip()
        except:
            address = ""
        try:
            # Lat/Lon
            latlon_elem = block.find_element(By.XPATH, ".//span[2]")
            latlon = latlon_elem.text.strip()
            lat, lon = latlon.split(",")
        except:
            lat, lon = "", ""
            lat = lat.replace('GPS: ','')
        output.append({
            "address": address,
            "latitude": lat.strip(),
            "longitude": lon.strip()
        })


def main():
    driver = setup_driver()
    try:
        driver.get(base_url)
        time.sleep(2)
        total_pages = 300
        print(f"Total pages: {total_pages}")

        for page in range(1, total_pages + 1):
            url = base_url if page == 1 else f"{base_url}{page}.html"
            print(f"Processing: {url}")
            driver.get(url)
            time.sleep(2)
            extract_data_from_page(driver)
            time.sleep(1)

        # Save to CSV
        with open("moradas_torres_vedras.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["address", "latitude", "longitude"])
            writer.writeheader()
            writer.writerows(output)
        print("✅ Ficheiro CSV guardado.")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
