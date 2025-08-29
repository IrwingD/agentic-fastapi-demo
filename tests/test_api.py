import pytest
from httpx import AsyncClient, ASGITransport
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

AGENT_API_KEY = os.environ["AGENT_API_KEY"]

@pytest.mark.asyncio
async def test_agent_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/agent",
            headers={"x-api-key": AGENT_API_KEY},
            json={"user_input": "How many leaves do employees get each year"},
        )
    assert response.status_code == 200
    assert "leaves" in response.text.lower()
