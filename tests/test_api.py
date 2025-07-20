import pytest
import asyncio
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.server import app

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "framework" in data
    assert "status" in data
    assert "version" in data

def test_docs_endpoint():
    """Test the documentation endpoint"""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_inference_text():
    """Test text inference endpoint"""
    payload = {
        "input_data": "This is a test text for toxicity detection",
        "input_type": "text",
        "confidence_threshold": 0.5
    }
    response = client.post("/inference", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "neural_output" in data
    assert "symbols" in data
    assert "reasoning_chain" in data
    assert "explanation" in data
    
    # Check neural output structure
    neural_output = data["neural_output"]
    assert "label" in neural_output
    assert "confidence" in neural_output
    assert "text" in neural_output

def test_inference_image():
    """Test image inference endpoint"""
    payload = {
        "input_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAAAAAAAD",  # Mock base64 image
        "input_type": "image",
        "confidence_threshold": 0.5
    }
    response = client.post("/inference", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "neural_output" in data
    assert "symbols" in data
    assert "reasoning_chain" in data
    assert "explanation" in data

def test_reasoning_endpoint():
    """Test reasoning endpoint"""
    payload = {
        "symbols": ["TOXIC_SUBSTANCE", "MEDICAL_CONTEXT"],
        "max_depth": 5
    }
    response = client.post("/api/reasoning/apply", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "success" in data
    assert "reasoning_chains" in data

def test_explanation_endpoint():
    """Test explanation endpoint"""
    payload = {
        "neural_output": {
            "label": "TOXIC_SUBSTANCE",
            "confidence": 0.8
        },
        "symbols": ["TOXIC_SUBSTANCE"],
        "reasoning_chains": [
            {
                "rule_id": "test_rule",
                "rule_description": "Test rule description",
                "premises": ["premise1"],
                "conclusion": "test conclusion",
                "confidence": 0.8,
                "reasoning_type": "deductive"
            }
        ],
        "format": "detailed"
    }
    response = client.post("/api/explanation/generate", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "explanation" in data
    assert "trace_id" in data
    assert "format_used" in data

if __name__ == "__main__":
    pytest.main([__file__])
