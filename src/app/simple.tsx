"use client";

import { useState } from "react";

interface InferenceResult {
  neural_output: {
    label: string;
    confidence: number;
    text?: string;
  };
  symbols: string[];
  reasoning_chain: string[];
  explanation: string | object; // Use a more specific type for explanation
  confidence_score: number;
}

export default function SimplePage() {
  const [inputData, setInputData] = useState("");
  const [inputType, setInputType] = useState<"text" | "image">("text");
  const [confidence, setConfidence] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleInference = async () => {
    if (!inputData.trim()) {
      setError("Please enter some input data");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("http://localhost:8000/inference", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          input_data: inputData,
          input_type: inputType,
          confidence_threshold: confidence,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold mb-4 text-blue-600">
          🧠 Neurosymbolic AI Framework
        </h1>
        <p className="text-gray-600 text-lg">
          Combining Neural Networks with Symbolic Reasoning for Explainable AI
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Input Panel */}
        <div className="bg-white p-6 rounded-lg shadow-lg border">
          <h2 className="text-xl font-semibold mb-4">📝 Input & Configuration</h2>
          
          {/* Input Type Selection */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Input Type</label>
            <div className="flex gap-2">
              <button
                className={`px-4 py-2 rounded ${
                  inputType === "text" 
                    ? "bg-blue-500 text-white" 
                    : "bg-gray-200 text-gray-700"
                }`}
                onClick={() => setInputType("text")}
              >
                📄 Text
              </button>
              <button
                className={`px-4 py-2 rounded ${
                  inputType === "image" 
                    ? "bg-blue-500 text-white" 
                    : "bg-gray-200 text-gray-700"
                }`}
                onClick={() => setInputType("image")}
              >
                🖼️ Image
              </button>
            </div>
          </div>

          {/* Input Data */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">
              {inputType === "text" ? "Text Input" : "Image Data (Base64)"}
            </label>
            <textarea
              placeholder={
                inputType === "text"
                  ? "Enter your text for analysis..."
                  : "Enter base64 encoded image data..."
              }
              value={inputData}
              onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setInputData(e.target.value)}
              rows={4}
              className="w-full p-3 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Confidence Threshold */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">
              Confidence Threshold: {confidence.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={confidence}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setConfidence(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>

          {/* Submit Button */}
          <button
            onClick={handleInference}
            disabled={loading || !inputData.trim()}
            className={`w-full py-3 px-4 rounded font-medium ${
              loading || !inputData.trim()
                ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                : "bg-blue-500 hover:bg-blue-600 text-white"
            }`}
          >
            {loading ? "⏳ Processing..." : "⚡ Run Neurosymbolic Analysis"}
          </button>

          {/* Error Display */}
          {error && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded">
              <p className="text-red-700 text-sm">❌ {error}</p>
            </div>
          )}
        </div>

        {/* Results Panel */}
        <div className="bg-white p-6 rounded-lg shadow-lg border">
          <h2 className="text-xl font-semibold mb-4">🧠 Analysis Results</h2>
          
          {!result && !loading && (
            <div className="text-center py-8 text-gray-500">
              <p>🚀 Run an analysis to see results here</p>
            </div>
          )}

          {loading && (
            <div className="text-center py-8">
              <div className="animate-spin text-4xl mb-2">⏳</div>
              <p className="text-gray-600">Processing your request...</p>
            </div>
          )}

          {result && (
            <div className="space-y-4">
              {/* Neural Output */}
              <div className="p-3 bg-blue-50 rounded">
                <h3 className="font-semibold mb-2">🧠 Neural Classification</h3>
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-1 bg-blue-200 rounded text-sm">
                      {result.neural_output.label}
                    </span>
                    <span className="text-sm text-gray-600">
                      {(result.neural_output.confidence * 100).toFixed(1)}% confidence
                    </span>
                  </div>
                </div>
              </div>

              {/* Symbols */}
              <div className="p-3 bg-green-50 rounded">
                <h3 className="font-semibold mb-2">⚡ Extracted Symbols</h3>
                <div className="flex flex-wrap gap-1">
                  {result.symbols.map((symbol, index) => (
                    <span key={index} className="px-2 py-1 bg-green-200 rounded text-xs">
                      {symbol}
                    </span>
                  ))}
                </div>
              </div>

              {/* Reasoning Chain */}
              <div className="p-3 bg-yellow-50 rounded">
                <h3 className="font-semibold mb-2">🔗 Reasoning Chain</h3>
                <div className="text-sm space-y-1">
                  <p>📊 {result.reasoning_chain.length} reasoning steps applied</p>
                  <p>🎯 Overall confidence: {(result.confidence_score * 100).toFixed(1)}%</p>
                </div>
              </div>

              {/* Explanation Preview */}
              {result.explanation && (
                <div className="p-3 bg-purple-50 rounded">
                  <h3 className="font-semibold mb-2">📖 Explanation</h3>
                  <div className="text-xs bg-white p-2 rounded border max-h-32 overflow-y-auto">
                    <pre className="whitespace-pre-wrap">
                      {typeof result.explanation === "string"
                        ? result.explanation
                        : JSON.stringify(result.explanation, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="mt-8 text-center text-sm text-gray-500">
        <p>
          🔬 Neurosymbolic AI Framework - Combining the power of neural networks with symbolic reasoning
        </p>
        <p className="mt-1">
          Backend API: <a href="http://localhost:8000/docs" className="text-blue-600 hover:underline">http://localhost:8000/docs</a>
        </p>
      </div>
    </div>
  );
}
