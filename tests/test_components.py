import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_models.text_model_mock import TextClassifier
from neural_models.image_model_mock import ImageClassifier
from reasoning_engine.inference_engine import InferenceEngine
from symbol_extractor.symbol_mapper import SymbolMapper
from explanation_generator.explanation_builder import ExplanationBuilder

class TestNeuralModels:
    """Test neural model components"""
    
    @pytest.mark.asyncio
    async def test_text_classifier_init(self):
        """Test text classifier initialization"""
        classifier = TextClassifier()
        assert classifier.model_name == "mock-distilbert-text-classifier"
        assert classifier.device == "cpu"
    
    @pytest.mark.asyncio
    async def test_text_classifier_prediction(self):
        """Test text classifier prediction"""
        classifier = TextClassifier()
        result = await classifier.predict("This is a test text")
        
        assert "label" in result
        assert "confidence" in result
        assert "raw_prediction" in result
        assert isinstance(result["confidence"], float)
        assert 0 <= result["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_image_classifier_init(self):
        """Test image classifier initialization"""
        classifier = ImageClassifier()
        assert classifier.model_name == "mock-vit-image-classifier"
        assert classifier.device == "cpu"
    
    @pytest.mark.asyncio
    async def test_image_classifier_prediction(self):
        """Test image classifier prediction"""
        classifier = ImageClassifier()
        result = await classifier.predict("data:image/jpeg;base64,test")
        
        assert "label" in result
        assert "confidence" in result
        assert "image_processed" in result
        assert isinstance(result["confidence"], float)
        assert 0 <= result["confidence"] <= 1

class TestSymbolicComponents:
    """Test symbolic reasoning components"""
    
    def test_symbol_mapper_init(self):
        """Test symbol mapper initialization"""
        mapper = SymbolMapper()
        assert hasattr(mapper, 'grounding_rules')
    
    def test_symbol_mapper_extraction(self):
        """Test symbol extraction"""
        mapper = SymbolMapper()
        prediction = {
            "label": "TOXIC_SUBSTANCE",
            "confidence": 0.8,
            "raw_prediction": {"label": "TOXIC_SUBSTANCE", "score": 0.8}
        }
        
        symbols = mapper.map_to_symbols(prediction)
        assert isinstance(symbols, list)
        assert len(symbols) > 0
    
    def test_inference_engine_init(self):
        """Test inference engine initialization"""
        engine = InferenceEngine()
        assert hasattr(engine, 'domain_rules')
        assert hasattr(engine, 'default_rules')
        assert hasattr(engine, 'working_memory')
    
    def test_inference_engine_reasoning(self):
        """Test basic reasoning"""
        engine = InferenceEngine()
        symbols = ["TOXIC_SUBSTANCE", "MEDICAL_CONTEXT"]
        
        reasoning_chain = engine.infer(symbols)
        assert isinstance(reasoning_chain, list)
        # The infer method returns a list of reasoning steps
    
    def test_explanation_builder_init(self):
        """Test explanation builder initialization"""
        builder = ExplanationBuilder()
        assert hasattr(builder, 'trace_logger')
    
    def test_explanation_generation(self):
        """Test explanation generation"""
        builder = ExplanationBuilder()
        
        # Create mock data for build_explanation
        neural_output = {
            "label": "TOXIC_SUBSTANCE",
            "confidence": 0.8
        }
        symbols = ["TOXIC_SUBSTANCE"]
        reasoning_chain = []
        
        explanation = builder.build_explanation(neural_output, symbols, reasoning_chain)
        assert "neural_analysis" in explanation
        assert "explanation_metadata" in explanation

if __name__ == "__main__":
    pytest.main([__file__])
