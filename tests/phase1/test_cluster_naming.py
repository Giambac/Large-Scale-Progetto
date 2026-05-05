"""Tests for cluster_naming.py — FOUND-02: LLM cluster names and descriptions."""
import pytest
from unittest.mock import MagicMock, patch

from src.cluster_naming import name_cluster


def _make_mock_client(name: str, description: str):
    """Build a mock Anthropic client that returns a fixed JSON response."""
    import json
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps({"name": name, "description": description}))]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    return mock_client


def test_name_cluster_returns_name_and_description():
    """name_cluster returns a dict with 'name' and 'description' keys."""
    client = _make_mock_client("Positive Reviews", "Reviews expressing satisfaction.")
    result = name_cluster(client, ["Great product!", "Love this item."], cluster_id=0)
    assert "name" in result
    assert "description" in result


def test_name_cluster_name_is_nonempty_string():
    """name is a non-empty string."""
    client = _make_mock_client("Crafting Supplies", "Items used in arts and crafts.")
    result = name_cluster(client, ["Nice fabric.", "Good thread."], cluster_id=1)
    assert isinstance(result["name"], str)
    assert len(result["name"]) > 0


def test_name_cluster_description_is_nonempty_string():
    """description is a non-empty string."""
    client = _make_mock_client("Negative Reviews", "Customers who were disappointed.")
    result = name_cluster(client, ["Terrible quality.", "Fell apart."], cluster_id=2)
    assert isinstance(result["description"], str)
    assert len(result["description"]) > 0


def test_name_cluster_crashes_on_bad_llm_schema():
    """name_cluster raises AssertionError if LLM returns JSON missing 'name' or 'description'."""
    import json
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps({"unexpected_key": "value"}))]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    with pytest.raises(AssertionError, match="bad schema"):
        name_cluster(mock_client, ["Some text."], cluster_id=0)
