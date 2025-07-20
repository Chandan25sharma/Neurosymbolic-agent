# Neurosymbolic AI Framework

A complete neurosymbolic AI framework that combines neural networks with symbolic reasoning to provide explainable AI decisions.

## Features

- 🧠 **Neural Processing**: Text and image classification using mock neural models
- 🔗 **Symbol Mapping**: Converts neural outputs to symbolic representations
- ⚡ **Symbolic Reasoning**: Applies logical rules and inference chains
- 📖 **Explainable AI**: Generates human-readable explanations for AI decisions
- 🌐 **Web Interface**: Modern Next.js frontend with shadcn/ui components
- 🚀 **REST API**: FastAPI backend with comprehensive endpoints

## Architecture

```
Frontend (Next.js + React)  →  Backend (FastAPI + Python)
         ↓                              ↓
    User Interface              Neurosymbolic Pipeline:
    - Text/Image Input          1. Neural Models (Text/Image)
    - Results Display           2. Symbol Extraction
    - Explanations             3. Reasoning Engine
                               4. Explanation Builder
```

## Quick Start

### Prerequisites

- Python 3.13+ with pip
- Node.js 18+ 
- Bun package manager (automatically installed)

### Installation & Setup

1. **Clone and navigate to the project:**
   ```bash
   cd c:\PROJECTS\Neurosymbolic-agent
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies:**
   ```bash
   bun install
   ```

### Running the Application

1. **Start the FastAPI backend (Terminal 1):**
   ```bash
   python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Start the Next.js frontend (Terminal 2):**
   ```bash
   bun run dev
   ```

3. **Access the application:**
   - Frontend: http://localhost:3000
   - API Documentation: http://localhost:8000/docs
   - API Health Check: http://localhost:8000/

### Testing

Run the comprehensive test suite:
```bash
python -m pytest tests/ -v
```

## API Endpoints

### Core Endpoints

- `GET /` - Health check and API information
- `POST /inference` - Main neurosymbolic inference pipeline
- `GET /health` - Detailed component health status

### Extended API Routes

- `POST /api/reasoning/apply` - Apply symbolic reasoning to symbols
- `POST /api/explanation/generate` - Generate explanations from reasoning
- `GET /api/explanation/traces` - Retrieve reasoning traces

## Example Usage

### Text Analysis
```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "input_data": "This medical treatment is safe and effective",
    "input_type": "text",
    "confidence_threshold": 0.5
  }'
```

### Image Analysis
```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "input_data": "data:image/jpeg;base64,/9j/4AAQ...",
    "input_type": "image",
    "confidence_threshold": 0.5
  }'
```

## Response Format

The API returns comprehensive neurosymbolic analysis:

```json
{
  "neural_output": {
    "label": "SAFETY_POSITIVE",
    "confidence": 0.85,
    "text": "Input text here"
  },
  "symbols": ["MEDICAL_CONTEXT", "SAFETY_POSITIVE"],
  "reasoning_chain": [
    {
      "rule_id": "medical_safety_rule",
      "premises": ["MEDICAL_CONTEXT", "SAFETY_POSITIVE"],
      "conclusion": "SAFE_FOR_USE",
      "confidence": 0.9
    }
  ],
  "explanation": {
    "neural_analysis": "...",
    "symbolic_reasoning": "...",
    "final_decision": "..."
  },
  "confidence_score": 0.85
}
```

## Project Structure

```
├── api/                    # FastAPI backend
│   ├── server.py          # Main FastAPI application
│   └── routes/            # API route handlers
├── neural_models/         # Neural network components
├── symbol_extractor/      # Neural-to-symbolic mapping
├── reasoning_engine/      # Symbolic reasoning logic
├── explanation_generator/ # Explanation generation
├── src/                   # Next.js frontend
├── tests/                 # Test suite
└── requirements.txt       # Python dependencies
```

## Development

### Backend Development
- FastAPI with automatic reload: `uvicorn api.server:app --reload`
- Run tests: `pytest tests/ -v`
- Format code: `black .`
- Lint code: `flake8 .`

### Frontend Development
- Next.js with Turbopack: `bun run dev`
- Build: `bun run build`
- Format: `bun run format`
- Lint: `bun run lint`

## Technologies Used

### Backend
- **FastAPI**: Modern, fast web framework
- **Pydantic**: Data validation and serialization
- **PyTorch**: Neural network framework (mock implementations)
- **Z3**: Symbolic reasoning solver
- **NetworkX**: Graph-based reasoning

### Frontend
- **Next.js 15**: React framework with Turbopack
- **shadcn/ui**: Modern UI component library
- **Tailwind CSS**: Utility-first CSS framework
- **TypeScript**: Type-safe JavaScript

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For questions and support, please open an issue in the repository.
