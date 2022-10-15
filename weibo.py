import time

from selenium import webdriver
from selenium.webdriver.common.by import By

options = webdriver.ChromeOptions()
# options.add_argument("headless")
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
)
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_argument("--disable-blink-features=AutomationControlled")

if __name__ == "__main__":
    driver = webdriver.Chrome(options=options)
    start_url = "https://weibo.com/u/2687853450?tabtype=album"
    driver.get(start_url)
    time.sleep(10)
    urls = driver.find_elements(By.CLASS_NAME, "woo-picture-img")
    for url in urls:
        print(url.get_attribute("src"))
    driver.close()
    driver.quit()
