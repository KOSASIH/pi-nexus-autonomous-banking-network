# UI Integration Tests
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

@pytest.fixture
def browser():
    return webdriver.Chrome()

def test_login(browser):
    # Test login functionality
    browser.get("https://pi-nexus-autonomous-banking-network.com/login")
    username_input = browser.find_element_by_name("username")
    password_input = browser.find_element_by_name("password")
    username_input.send_keys("test_user")
    password_input.send_keys("test_password")
    browser.find_element_by_name("login").click()
    WebDriverWait(browser, 10).until(EC.title_contains("Dashboard"))

def test_account_list(browser):
    # Test account list page
    browser.get("https://pi-nexus-autonomous-banking-network.com/accounts")
    account_list = browser.find_elements_by_css_selector(".account-list li")
    assert len(account_list) > 0

def test_create_account(browser):
    # Test create account functionality
    browser.get("https://pi-nexus-autonomous-banking-network.com/accounts/new")
    account_name_input = browser.find_element_by_name("account_name")
    initial_balance_input = browser.find_element_by_name("initial_balance")
    account_name_input.send_keys("Test Account")
    initial_balance_input.send_keys("1000")
    browser.find_element_by_name("create_account").click()
    WebDriverWait(browser, 10).until(EC.title_contains("Account Created"))
