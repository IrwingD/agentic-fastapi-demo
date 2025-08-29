import os
import pytest
from httpx import AsyncClient
from main import app

AGENT_API_KEY = os.environ["AGENT_API_KEY"]  # pulled from GitHub Actions secret

@pytest.mark.asyncio
async def test_agent_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/agent",
            headers={"x-api-key": AGENT_API_KEY},
            json={"user_input": "How many leaves do employees get each year"}
        )
    assert response.status_code == 200
