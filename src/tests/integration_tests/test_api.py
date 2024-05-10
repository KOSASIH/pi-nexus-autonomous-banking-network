# API Integration Tests
import pytest
import requests

@pytest.fixture
def api_url():
    return "https://pi-nexus-autonomous-banking-network.com/api"

@pytest.fixture
def auth_token():
    # Authenticate and retrieve token
    response = requests.post(f"{api_url}/auth", json={"username": "test_user", "password": "test_password"})
    return response.json()["token"]

def test_get_accounts(api_url, auth_token):
    # Test GET /accounts endpoint
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = requests.get(f"{api_url}/accounts", headers=headers)
    assert response.status_code == 200
    assert len(response.json()["accounts"]) > 0

def test_create_account(api_url, auth_token):
    # Test POST /accounts endpoint
    headers = {"Authorization": f"Bearer {auth_token}"}
    data = {"account_name": "Test Account", "initial_balance": 1000}
    response = requests.post(f"{api_url}/accounts", headers=headers, json=data)
    assert response.status_code == 201
    assert response.json()["account_id"] is not None

def test_transfer_funds(api_url, auth_token):
    # Test POST /transfers endpoint
    headers = {"Authorization": f"Bearer {auth_token}"}
    data = {"from_account_id": 1, "to_account_id": 2, "amount": 500}
    response = requests.post(f"{api_url}/transfers", headers=headers, json=data)
    assert response.status_code == 201
    assert response.json()["transfer_id"] is not None
