# web_scraping.py

import os
import json
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

class WebScraping:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.load_config()
        self.driver = self.create_driver()

    def load_config(self):
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        return config

    def create_driver(self):
        driver = webdriver.Chrome(ChromeDriverManager().install())
        return driver

    def scrape_page(self, url):
        self.driver.get(url)
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        return soup

    def wait_for_element(self, by, value, timeout=10):
        try:
            element = WebDriverWait(self.driver, timeout).until(EC.presence_of_element_located((by, value)))
            return element
        except Exception as e:
            print(f'Error: {e}')
            return None

    def close_driver(self):
        self.driver.close()

web_scraping = WebScraping('web_scraping_config.json')
soup = web_scraping.scrape_page('https://www.example.com')
print(soup.prettify())
web_scraping.close_driver()
