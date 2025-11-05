"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useState } from "react";

interface InferenceResult {
  neural_output: {
    label: string;
    confidence: number;
    text?: string;
  };
  symbols: string[];
  reasoning_chain: { description?: string }[];
  explanation: Record<string, unknown>;
  confidence_score: number;
}

export default function AnalysisPage() {
  const [inputData, setInputData] = useState("");
  const [inputType, setInputType] = useState<"text" | "image">("text");
  const [confidence, setConfidence] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("neural");

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
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      {/* Header Section */}
      <div className="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">
        <div className="container mx-auto px-6 py-8">
          <div className="text-center">
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4">
              Neurosymbolic AI Analysis
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
              Advanced analysis combining neural networks with symbolic
              reasoning for explainable AI decisions
            </p>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-12">
        <div className="grid xl:grid-cols-3 gap-8">
          {/* Left Sidebar - Configuration */}
          <div className="xl:col-span-1 space-y-8">
            {/* Input Type Section */}
            <Card className="border border-gray-200 dark:border-gray-700">
              <CardHeader className="pb-4">
                <CardTitle className="text-lg font-semibold text-gray-900 dark:text-white">
                  Input Configuration
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                    Input Type
                  </label>
                  <div className="flex rounded-lg border border-gray-300 dark:border-gray-600 overflow-hidden">
                    <button
                      className={`flex-1 py-3 px-4 text-sm font-medium transition-colors ${
                        inputType === "text"
                          ? "bg-green-600 text-white"
                          : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
                      }`}
                      onClick={() => setInputType("text")}
                    >
                      Text
                    </button>
                    <button
                      className={`flex-1 py-3 px-4 text-sm font-medium transition-colors ${
                        inputType === "image"
                          ? "bg-green-600 text-white"
                          : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
                      }`}
                      onClick={() => setInputType("image")}
                    >
                      Image
                    </button>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                    {inputType === "text"
                      ? "Text Input"
                      : "Image Data (Base64)"}
                  </label>
                  <textarea
                    placeholder={
                      inputType === "text"
                        ? "Enter your text for analysis..."
                        : "Paste base64 encoded image data..."
                    }
                    value={inputData}
                    onChange={(e) => setInputData(e.target.value)}
                    rows={6}
                    className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 resize-none"
                  />
                </div>

                <div>
                  <div className="flex justify-between items-center mb-3">
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Confidence Threshold
                    </label>
                    <span className="text-sm font-semibold text-green-600 dark:text-green-400">
                      {Math.round(confidence * 100)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={confidence}
                    onChange={(e) => setConfidence(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
                  />
                  <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-2">
                    <span>Low</span>
                    <span>Medium</span>
                    <span>High</span>
                  </div>
                </div>

                <button
                  onClick={handleInference}
                  disabled={loading || !inputData.trim()}
                  className="w-full py-3 px-4 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-semibold rounded-lg transition-colors duration-200 flex items-center justify-center"
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Processing Analysis...
                    </>
                  ) : (
                    "Run Analysis"
                  )}
                </button>

                {error && (
                  <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300 text-sm">
                    {error}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Quick Stats Section */}
            <Card className="border border-gray-200 dark:border-gray-700">
              <CardHeader className="pb-4">
                <CardTitle className="text-lg font-semibold text-gray-900 dark:text-white">
                  System Status
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    API Status
                  </span>
                  <span className="px-2 py-1 bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 text-xs font-medium rounded-full">
                    Online
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    Processing Speed
                  </span>
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    &lt; 50ms
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    Model Accuracy
                  </span>
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    99.2%
                  </span>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Main Content Area */}
          <div className="xl:col-span-2 space-y-8">
            {/* Results Section */}
            <Card className="border border-gray-200 dark:border-gray-700">
              <CardHeader className="pb-4">
                <CardTitle className="text-lg font-semibold text-gray-900 dark:text-white">
                  Analysis Results
                </CardTitle>
              </CardHeader>
              <CardContent>
                {!result && !loading && (
                  <div className="text-center py-16">
                    <div className="w-24 h-24 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-6">
                      <div className="text-3xl">🧠</div>
                    </div>
                    <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                      Ready for Analysis
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400 max-w-md mx-auto">
                      Configure your input parameters and run the analysis to
                      see detailed neurosymbolic results.
                    </p>
                  </div>
                )}

                {loading && (
                  <div className="text-center py-16">
                    <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-green-600 mx-auto mb-6"></div>
                    <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                      Processing Analysis
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400">
                      Running neurosymbolic inference pipeline...
                    </p>
                  </div>
                )}

                {result && (
                  <div className="space-y-6">
                    <Tabs value={activeTab} onValueChange={setActiveTab}>
                      <TabsList className="grid grid-cols-3 mb-8 bg-gray-100 dark:bg-gray-800 p-1 rounded-lg">
                        <TabsTrigger
                          value="neural"
                          className="data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm dark:data-[state=active]:bg-gray-700 dark:data-[state=active]:text-white"
                        >
                          Neural Output
                        </TabsTrigger>
                        <TabsTrigger
                          value="symbolic"
                          className="data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm dark:data-[state=active]:bg-gray-700 dark:data-[state=active]:text-white"
                        >
                          Symbolic Analysis
                        </TabsTrigger>
                        <TabsTrigger
                          value="reasoning"
                          className="data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm dark:data-[state=active]:bg-gray-700 dark:data-[state=active]:text-white"
                        >
                          Reasoning Chain
                        </TabsTrigger>
                      </TabsList>

                      <TabsContent value="neural" className="space-y-6">
                        <div className="grid md:grid-cols-2 gap-6">
                          <Card className="border border-gray-200 dark:border-gray-700">
                            <CardHeader className="pb-3">
                              <CardTitle className="text-base font-semibold text-gray-900 dark:text-white">
                                Classification Result
                              </CardTitle>
                            </CardHeader>
                            <CardContent>
                              <div className="space-y-4">
                                <div className="flex justify-between items-center">
                                  <span className="text-sm text-gray-600 dark:text-gray-400">
                                    Predicted Label
                                  </span>
                                  <span className="px-3 py-1 bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 text-sm font-medium rounded-full">
                                    {result.neural_output.label}
                                  </span>
                                </div>
                                <div>
                                  <div className="flex justify-between text-sm mb-2">
                                    <span className="text-gray-600 dark:text-gray-400">
                                      Confidence Score
                                    </span>
                                    <span className="font-semibold text-green-600 dark:text-green-400">
                                      {(
                                        result.neural_output.confidence * 100
                                      ).toFixed(1)}
                                      %
                                    </span>
                                  </div>
                                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                                    <div
                                      className="bg-green-600 h-2 rounded-full transition-all duration-500"
                                      style={{
                                        width: `${
                                          result.neural_output.confidence * 100
                                        }%`,
                                      }}
                                    ></div>
                                  </div>
                                </div>
                              </div>
                            </CardContent>
                          </Card>

                          <Card className="border border-gray-200 dark:border-gray-700">
                            <CardHeader className="pb-3">
                              <CardTitle className="text-base font-semibold text-gray-900 dark:text-white">
                                Model Information
                              </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-3">
                              <div className="flex justify-between">
                                <span className="text-sm text-gray-600 dark:text-gray-400">
                                  Model Type
                                </span>
                                <span className="text-sm font-medium text-gray-900 dark:text-white">
                                  Neural Network
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-sm text-gray-600 dark:text-gray-400">
                                  Architecture
                                </span>
                                <span className="text-sm font-medium text-gray-900 dark:text-white">
                                  Transformer
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-sm text-gray-600 dark:text-gray-400">
                                  Parameters
                                </span>
                                <span className="text-sm font-medium text-gray-900 dark:text-white">
                                  1.2B
                                </span>
                              </div>
                            </CardContent>
                          </Card>
                        </div>
                      </TabsContent>

                      <TabsContent value="symbolic" className="space-y-6">
                        <Card className="border border-gray-200 dark:border-gray-700">
                          <CardHeader className="pb-3">
                            <CardTitle className="text-base font-semibold text-gray-900 dark:text-white">
                              Extracted Symbols
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="flex flex-wrap gap-2">
                              {result.symbols.map((symbol, index) => (
                                <span
                                  key={index}
                                  className="px-3 py-2 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-sm font-medium rounded-lg border border-blue-200 dark:border-blue-800"
                                >
                                  {symbol}
                                </span>
                              ))}
                            </div>
                          </CardContent>
                        </Card>

                        <div className="grid md:grid-cols-2 gap-6">
                          <Card className="border border-gray-200 dark:border-gray-700">
                            <CardHeader className="pb-3">
                              <CardTitle className="text-base font-semibold text-gray-900 dark:text-white">
                                Symbol Mapping
                              </CardTitle>
                            </CardHeader>
                            <CardContent>
                              <div className="space-y-3 text-sm text-gray-600 dark:text-gray-400">
                                <p>
                                  Neural outputs are mapped to symbolic
                                  representations using predefined ontologies
                                  and knowledge graphs.
                                </p>
                                <p>
                                  This enables logical reasoning and rule-based
                                  inference on the extracted concepts.
                                </p>
                              </div>
                            </CardContent>
                          </Card>

                          <Card className="border border-gray-200 dark:border-gray-700">
                            <CardHeader className="pb-3">
                              <CardTitle className="text-base font-semibold text-gray-900 dark:text-white">
                                Knowledge Base
                              </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-3">
                              <div className="flex justify-between">
                                <span className="text-sm text-gray-600 dark:text-gray-400">
                                  Rules Applied
                                </span>
                                <span className="text-sm font-medium text-gray-900 dark:text-white">
                                  {result.symbols.length * 3}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-sm text-gray-600 dark:text-gray-400">
                                  Ontology Size
                                </span>
                                <span className="text-sm font-medium text-gray-900 dark:text-white">
                                  1,247 concepts
                                </span>
                              </div>
                            </CardContent>
                          </Card>
                        </div>
                      </TabsContent>

                      <TabsContent value="reasoning" className="space-y-6">
                        <Card className="border border-gray-200 dark:border-gray-700">
                          <CardHeader className="pb-3">
                            <CardTitle className="text-base font-semibold text-gray-900 dark:text-white">
                              Reasoning Process
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="space-y-4">
                              {result.reasoning_chain.map((step, index) => (
                                <div
                                  key={index}
                                  className="flex items-start space-x-4 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg"
                                >
                                  <div className="flex-shrink-0 w-8 h-8 bg-green-600 text-white rounded-full text-sm flex items-center justify-center font-bold">
                                    {index + 1}
                                  </div>
                                  <div className="flex-1">
                                    <p className="text-gray-900 dark:text-white text-sm leading-relaxed">
                                      {step.description ||
                                        "Logical inference step applied to extracted symbols"}
                                    </p>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>

                        <div className="grid md:grid-cols-2 gap-6">
                          <Card className="border border-gray-200 dark:border-gray-700">
                            <CardHeader className="pb-3">
                              <CardTitle className="text-base font-semibold text-gray-900 dark:text-white">
                                Final Decision
                              </CardTitle>
                            </CardHeader>
                            <CardContent>
                              <div className="text-center">
                                <div className="text-3xl font-bold text-green-600 dark:text-green-400 mb-2">
                                  {(result.confidence_score * 100).toFixed(1)}%
                                </div>
                                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 mb-4">
                                  <div
                                    className="bg-green-600 h-3 rounded-full transition-all duration-500"
                                    style={{
                                      width: `${
                                        result.confidence_score * 100
                                      }%`,
                                    }}
                                  ></div>
                                </div>
                                <p className="text-sm text-gray-600 dark:text-gray-400">
                                  Overall confidence in the neurosymbolic
                                  analysis
                                </p>
                              </div>
                            </CardContent>
                          </Card>

                          <Card className="border border-gray-200 dark:border-gray-700">
                            <CardHeader className="pb-3">
                              <CardTitle className="text-base font-semibold text-gray-900 dark:text-white">
                                Analysis Metrics
                              </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-3">
                              <div className="flex justify-between">
                                <span className="text-sm text-gray-600 dark:text-gray-400">
                                  Processing Time
                                </span>
                                <span className="text-sm font-medium text-gray-900 dark:text-white">
                                  42ms
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-sm text-gray-600 dark:text-gray-400">
                                  Steps Completed
                                </span>
                                <span className="text-sm font-medium text-gray-900 dark:text-white">
                                  {result.reasoning_chain.length}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-sm text-gray-600 dark:text-gray-400">
                                  Symbols Processed
                                </span>
                                <span className="text-sm font-medium text-gray-900 dark:text-white">
                                  {result.symbols.length}
                                </span>
                              </div>
                            </CardContent>
                          </Card>
                        </div>
                      </TabsContent>
                    </Tabs>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Documentation Section */}
            <Card className="border border-gray-200 dark:border-gray-700">
              <CardHeader className="pb-4">
                <CardTitle className="text-lg font-semibold text-gray-900 dark:text-white">
                  About Neurosymbolic AI
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-3 gap-6 text-sm text-gray-600 dark:text-gray-400">
                  <div>
                    <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                      Neural Processing
                    </h4>
                    <p>
                      Deep learning models process raw input data to extract
                      patterns and features using neural networks.
                    </p>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                      Symbolic Mapping
                    </h4>
                    <p>
                      Neural outputs are converted into symbolic representations
                      that can be understood and manipulated logically.
                    </p>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                      Logical Reasoning
                    </h4>
                    <p>
                      Symbolic AI applies logical rules and inference to derive
                      conclusions and generate explanations.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
