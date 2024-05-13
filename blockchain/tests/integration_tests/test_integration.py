import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch
from app.main import app

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
@patch("app.main.some_external_service.get_data", new_callable=AsyncMock)
async def test_get_data_endpoint(mock_get_data):
    mock_get_data.return_value = {"data": "mock_data"}
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/data")
        assert response.status_code == 200
        assert response.json() == {"data": "mock_data"}
        mock_get_data.assert_called_once()

@pytest.mark.asyncio
@patch("app.main.some_external_service.post_data", new_callable=AsyncMock)
async def test_post_data_endpoint(mock_post_data):
    mock_post_data.return_value = {"message": "data posted"}
    async with AsyncClient(app=app, base_url="http://test") as ac:
response = await ac.post("/data", json={"data": "test_data"})
        assert response.status_code == 200
        assert response.json() == {"message": "data posted"}
        mock_post_data.assert_called_once()

@pytest.mark.asyncio
@patch("app.main.some_external_service.post_data", new_callable=AsyncMock)
async def test_post_data_endpoint_invalid_json(mock_post_data):
    mock_post_data.return_value = {"message": "data posted"}
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/data", json={"data": "test_data", "invalid_field": "test"})
        assert response.status_code == 422
        assert response.json() == {"detail": [{"loc": ["body", "invalid_field"], "msg": "field required", "type": "value_error.missing"}]}
        mock_post_data.assert_not_called()

@pytest.mark.asyncio
@patch("app.main.some_external_service.post_data", new_callable=AsyncMock)
async def test_post_data_endpoint_server_error(mock_post_data):
    mock_post_data.side_effect = Exception("Server error")
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/data", json={"data": "test_data"})
        assert response.status_code == 500
        assert response.json() == {"detail": "Server error"}

@pytest.mark.asyncio
@patch("app.main.some_external_service.post_data", new_callable=AsyncMock)
async def test_post_data_endpoint_validation_error(mock_post_data):
    mock_post_data.return_value = {"message": "data posted"}
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/data", json={"data": ""})
        assert response.status_code == 422
        assert response.json() == {"detail": [{"loc": ["body", "data"], "msg": "ensure this value has at least 1 characters", "type": "value_error.length"}]}
        mock_post_data.assert_not_called()

@pytest.mark.asyncio
@patch("app.main.some_external_service.post_data", new_callable=AsyncMock)
async def test_post_data_endpoint_unprocessable_entity(mock_post_data):
    mock_post_data.return_value = {"message": "data posted"}
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/data", json={"data": "test_data", "invalid_field": "test"})
        assert response.status_code == 422
        assert response.json() == {"detail": [{"loc": ["body", "invalid_field"], "msg": "field required", "type": "value_error.missing"}]}
        mock_post_data.assert_not_called()

@pytest.mark.asyncio
@patch("app.main.some_external_service.post_data", new_callable=AsyncMock)
async def test_post_data_endpoint_unexpected_error(mock_post_data):
    mock_post_data.side_effect = Exception("Unexpected error")
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/data", json={"data": "test_data"})
        assert response.status_code == 500
        assert response.json() == {"detail": "Unexpected error"}
