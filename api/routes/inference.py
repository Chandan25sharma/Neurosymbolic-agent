from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Request/Response models
class TextInferenceRequest(BaseModel):
    text: str
    confidence_threshold: float = 0.5
    include_entities: bool = True
    reasoning_depth: int = 5

class ImageInferenceRequest(BaseModel):
    image_data: str  # Base64 encoded image
    confidence_threshold: float = 0.5
    extract_visual_features: bool = True
    reasoning_depth: int = 5

class InferenceResponse(BaseModel):
    success: bool
    trace_id: str
    neural_output: Dict[str, Any]
    symbols: List[str]
    reasoning_chains: List[Dict[str, Any]]
    explanation: Dict[str, Any]
    processing_time: float
    metadata: Dict[str, Any]

class ModelInfoResponse(BaseModel):
    text_model: Dict[str, Any]
    image_model: Dict[str, Any]
    inference_engine: Dict[str, Any]
    system_status: str

@router.post("/text", response_model=InferenceResponse)
async def analyze_text(request: TextInferenceRequest):
    """
    Analyze text input using the neurosymbolic framework

    Args:
        request: Text inference request

    Returns:
        Complete inference response with reasoning chains and explanation
    """
    try:
        import time
        start_time = time.time()

        # Import components (these would be injected in a real app)
        from neural_models.text_model_mock import TextClassifier
        from symbol_extractor.symbol_mapper import SymbolMapper
        from reasoning_engine.inference_engine import InferenceEngine
        from explanation_generator.explanation_builder import ExplanationBuilder

        # Initialize components (in production, these would be singletons)
        text_classifier = TextClassifier()
        symbol_mapper = SymbolMapper()
        inference_engine = InferenceEngine()
        explanation_builder = ExplanationBuilder()

        # Step 1: Neural text analysis
        logger.info(f"Analyzing text: {request.text[:50]}...")
        neural_output = await text_classifier.predict(request.text)

        # Check confidence threshold
        if neural_output["confidence"] < request.confidence_threshold:
            raise HTTPException(
                status_code=400,
                detail=f"Neural prediction confidence ({neural_output['confidence']:.2f}) below threshold ({request.confidence_threshold})"
            )

        # Step 2: Symbol extraction
        symbols = symbol_mapper.map_to_symbols(neural_output)

        # Step 3: Symbolic reasoning
        inference_engine.max_inference_depth = request.reasoning_depth
        reasoning_chains = inference_engine.infer(symbols)

        # Step 4: Generate explanation
        explanation = explanation_builder.build_explanation(
            neural_output, symbols, reasoning_chains
        )

        processing_time = time.time() - start_time

        return InferenceResponse(
            success=True,
            trace_id=explanation.get("trace_id", "unknown"),
            neural_output=neural_output,
            symbols=symbols,
            reasoning_chains=reasoning_chains,
            explanation=explanation,
            processing_time=processing_time,
            metadata={
                "input_type": "text",
                "input_length": len(request.text),
                "symbols_extracted": len(symbols),
                "reasoning_chains": len(reasoning_chains),
                "confidence_threshold": request.confidence_threshold
            }
        )

    except Exception as e:
        logger.error(f"Text inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/image", response_model=InferenceResponse)
async def analyze_image(request: ImageInferenceRequest):
    """
    Analyze image input using the neurosymbolic framework

    Args:
        request: Image inference request

    Returns:
        Complete inference response with reasoning chains and explanation
    """
    try:
        import time
        start_time = time.time()

        # Import components
        from neural_models.image_model_mock import ImageClassifier
        from symbol_extractor.symbol_mapper import SymbolMapper
        from reasoning_engine.inference_engine import InferenceEngine
        from explanation_generator.explanation_builder import ExplanationBuilder

        # Initialize components
        image_classifier = ImageClassifier()
        symbol_mapper = SymbolMapper()
        inference_engine = InferenceEngine()
        explanation_builder = ExplanationBuilder()

        # Step 1: Neural image analysis
        logger.info("Analyzing image input...")
        neural_output = await image_classifier.predict(request.image_data)

        # Check confidence threshold
        if neural_output["confidence"] < request.confidence_threshold:
            raise HTTPException(
                status_code=400,
                detail=f"Neural prediction confidence ({neural_output['confidence']:.2f}) below threshold ({request.confidence_threshold})"
            )

        # Step 2: Symbol extraction
        symbols = symbol_mapper.map_to_symbols(neural_output)

        # Step 3: Symbolic reasoning
        inference_engine.max_inference_depth = request.reasoning_depth
        reasoning_chains = inference_engine.infer(symbols)

        # Step 4: Generate explanation
        explanation = explanation_builder.build_explanation(
            neural_output, symbols, reasoning_chains
        )

        processing_time = time.time() - start_time

        return InferenceResponse(
            success=True,
            trace_id=explanation.get("trace_id", "unknown"),
            neural_output=neural_output,
            symbols=symbols,
            reasoning_chains=reasoning_chains,
            explanation=explanation,
            processing_time=processing_time,
            metadata={
                "input_type": "image",
                "visual_features_extracted": request.extract_visual_features,
                "symbols_extracted": len(symbols),
                "reasoning_chains": len(reasoning_chains),
                "confidence_threshold": request.confidence_threshold
            }
        )

    except Exception as e:
        logger.error(f"Image inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about loaded models and system status

    Returns:
        Model information and system status
    """
    try:
        from neural_models.text_model_mock import TextClassifier
        from neural_models.image_model_mock import ImageClassifier
        from reasoning_engine.inference_engine import InferenceEngine

        # Get model information
        text_classifier = TextClassifier()
        image_classifier = ImageClassifier()
        inference_engine = InferenceEngine()

        return ModelInfoResponse(
            text_model=text_classifier.get_model_info(),
            image_model=image_classifier.get_model_info(),
            inference_engine=inference_engine.get_inference_statistics(),
            system_status="operational"
        )

    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def batch_inference(requests: List[Dict[str, Any]]):
    """
    Process multiple inference requests in batch

    Args:
        requests: List of inference requests

    Returns:
        List of inference responses
    """
    try:
        responses = []

        for i, req in enumerate(requests):
            try:
                if req.get("type") == "text":
                    text_req = TextInferenceRequest(**req)
                    response = await analyze_text(text_req)
                elif req.get("type") == "image":
                    image_req = ImageInferenceRequest(**req)
                    response = await analyze_image(image_req)
                else:
                    raise ValueError(f"Unknown request type: {req.get('type')}")

                responses.append(response.dict())

            except Exception as e:
                logger.error(f"Batch request {i} failed: {str(e)}")
                responses.append({
                    "success": False,
                    "error": str(e),
                    "request_index": i
                })

        return {"batch_results": responses}

    except Exception as e:
        logger.error(f"Batch inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint for the inference system"""
    try:
        # Quick health check of core components
        from neural_models.text_model_mock import TextClassifier
        from symbol_extractor.symbol_mapper import SymbolMapper
        from reasoning_engine.inference_engine import InferenceEngine

        # Basic initialization test
        text_classifier = TextClassifier()
        symbol_mapper = SymbolMapper()
        inference_engine = InferenceEngine()

        return {
            "status": "healthy",
            "components": {
                "text_classifier": "operational",
                "symbol_mapper": "operational",
                "inference_engine": "operational"
            },
            "timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z"
        }
