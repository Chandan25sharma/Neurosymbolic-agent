from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
import json
import logging

# Import our custom modules
from .routes import inference, reasoning, explanation
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_models.text_model_mock import TextClassifier
from neural_models.image_model_mock import ImageClassifier
from symbol_extractor.symbol_mapper import SymbolMapper
from reasoning_engine.inference_engine import InferenceEngine
from explanation_generator.explanation_builder import ExplanationBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Neurosymbolic AI Framework",
    description="A framework combining neural networks with symbolic reasoning for explainable AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class InferenceRequest(BaseModel):
    input_data: str
    input_type: str  # "text" or "image"
    confidence_threshold: float = 0.5

class InferenceResponse(BaseModel):
    neural_output: Dict[str, Any]
    symbols: List[str]
    reasoning_chain: List[Dict[str, Any]]
    explanation: Dict[str, Any]
    confidence_score: float

# Global instances (will be initialized on startup)
text_classifier = None
image_classifier = None
symbol_mapper = None
inference_engine = None
explanation_builder = None

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global text_classifier, image_classifier, symbol_mapper, inference_engine, explanation_builder

    logger.info("Initializing Neurosymbolic AI Framework...")

    try:
        # Initialize neural models
        text_classifier = TextClassifier()
        image_classifier = ImageClassifier()

        # Initialize symbol mapper
        symbol_mapper = SymbolMapper()

        # Initialize reasoning engine
        inference_engine = InferenceEngine()

        # Initialize explanation builder
        explanation_builder = ExplanationBuilder()

        logger.info("All components initialized successfully!")

    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise e

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Neurosymbolic AI Framework API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "components": {
            "text_classifier": text_classifier is not None,
            "image_classifier": image_classifier is not None,
            "symbol_mapper": symbol_mapper is not None,
            "inference_engine": inference_engine is not None,
            "explanation_builder": explanation_builder is not None
        }
    }

@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """Main inference endpoint that runs the full neurosymbolic pipeline"""
    try:
        # Step 1: Neural prediction
        if request.input_type == "text":
            neural_output = await text_classifier.predict(request.input_data)
        elif request.input_type == "image":
            neural_output = await image_classifier.predict(request.input_data)
        else:
            raise HTTPException(status_code=400, detail="Invalid input_type. Must be 'text' or 'image'")

        # Filter by confidence threshold
        if neural_output["confidence"] < request.confidence_threshold:
            raise HTTPException(status_code=400, detail="Prediction confidence below threshold")

        # Step 2: Symbol extraction
        symbols = symbol_mapper.map_to_symbols(neural_output)

        # Step 3: Symbolic reasoning
        reasoning_chain = inference_engine.infer(symbols)

        # Step 4: Generate explanation
        explanation = explanation_builder.build_explanation(
            neural_output, symbols, reasoning_chain
        )

        return InferenceResponse(
            neural_output=neural_output,
            symbols=symbols,
            reasoning_chain=reasoning_chain,
            explanation=explanation,
            confidence_score=neural_output["confidence"]
        )

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Include routers
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])
app.include_router(reasoning.router, prefix="/api/reasoning", tags=["reasoning"])
app.include_router(explanation.router, prefix="/api/explanation", tags=["explanation"])

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
