export default function ApiPage() {
  const endpoints = [
    {
      method: "POST",
      path: "/inference",
      description: "Main neurosymbolic inference endpoint",
      request: {
        input_data: "string (text or base64 image)",
        input_type: "text | image",
        confidence_threshold: "number (0.0 - 1.0)",
      },
      response: {
        neural_output: {
          label: "string",
          confidence: "number",
          text: "string (optional)",
        },
        symbols: "string[]",
        reasoning_chain: "object[]",
        explanation: "object",
        confidence_score: "number",
      },
    },
    {
      method: "GET",
      path: "/health",
      description: "System health check",
      response: {
        status: "string",
        version: "string",
        components: "object",
      },
    },
    {
      method: "POST",
      path: "/api/reasoning/apply",
      description: "Apply symbolic reasoning to extracted symbols",
      request: {
        symbols: "string[]",
        ruleset: "string",
      },
      response: {
        reasoning_chain: "object[]",
        conclusions: "string[]",
      },
    },
    {
      method: "POST",
      path: "/api/explanation/generate",
      description: "Generate human-readable explanations",
      request: {
        reasoning_chain: "object[]",
        format: "simple | detailed",
      },
      response: {
        explanation: "string",
        steps: "string[]",
      },
    },
  ];

  const codeExamples = {
    python: `import requests

url = "http://localhost:8000/inference"
data = {
    "input_data": "This medical treatment is safe and effective",
    "input_type": "text", 
    "confidence_threshold": 0.7
}

response = requests.post(url, json=data)
result = response.json()
print(f"Label: {result['neural_output']['label']}")
print(f"Confidence: {result['confidence_score']:.2%}")`,

    javascript: `const response = await fetch('http://localhost:8000/inference', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        input_data: 'This medical treatment is safe and effective',
        input_type: 'text',
        confidence_threshold: 0.7
    })
});

const result = await response.json();
console.log(\`Label: \${result.neural_output.label}\`);
console.log(\`Confidence: \${(result.confidence_score * 100).toFixed(1)}%\`);`,

    curl: `curl -X POST http://localhost:8000/inference \\
  -H "Content-Type: application/json" \\
  -d '{
    "input_data": "This medical treatment is safe and effective",
    "input_type": "text",
    "confidence_threshold": 0.7
  }'`,
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-900 dark:to-gray-800 py-12">
      <div className="container mx-auto px-6 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4">
            API Documentation
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
            Complete reference for the Neurosymbolic AI Framework REST API
          </p>
        </div>

        {/* Quick Start */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            Quick Start
          </h2>
          <div className="space-y-4">
            <p className="text-gray-600 dark:text-gray-400">
              Get started with the Neurosymbolic AI API in minutes. The API
              provides endpoints for text and image analysis with comprehensive
              neurosymbolic reasoning.
            </p>
            <div className="bg-gray-900 rounded-lg p-4">
              <pre className="text-green-400 text-sm overflow-x-auto">
                {codeExamples.curl}
              </pre>
            </div>
          </div>
        </div>

        {/* Base Information */}
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <h3 className="font-bold text-gray-900 dark:text-white mb-2">
              Base URL
            </h3>
            <code className="bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded text-sm text-gray-900 dark:text-white">
              http://localhost:8000
            </code>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Production: https://api.neurosymbolic-ai.com
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <h3 className="font-bold text-gray-900 dark:text-white mb-2">
              Authentication
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              No authentication required for local development. Production
              requires API keys.
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <h3 className="font-bold text-gray-900 dark:text-white mb-2">
              Rate Limits
            </h3>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• 100 requests per minute</li>
              <li>• 10,000 requests per day</li>
              <li>• Batch processing available</li>
            </ul>
          </div>
        </div>

        {/* API Endpoints */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            API Endpoints
          </h2>
          <div className="space-y-6">
            {endpoints.map((endpoint, index) => (
              <div
                key={index}
                className="border border-gray-200 dark:border-gray-700 rounded-lg p-6"
              >
                <div className="flex items-center gap-4 mb-4">
                  <span
                    className={`px-3 py-1 rounded-full text-sm font-medium ${
                      endpoint.method === "POST"
                        ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                        : "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
                    }`}
                  >
                    {endpoint.method}
                  </span>
                  <code className="text-lg font-mono text-gray-900 dark:text-white bg-gray-100 dark:bg-gray-700 px-3 py-1 rounded">
                    {endpoint.path}
                  </code>
                </div>
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  {endpoint.description}
                </p>

                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                      Request
                    </h4>
                    <pre className="bg-gray-100 dark:bg-gray-700 p-3 rounded text-sm overflow-x-auto">
                      {JSON.stringify(endpoint.request, null, 2)}
                    </pre>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                      Response
                    </h4>
                    <pre className="bg-gray-100 dark:bg-gray-700 p-3 rounded text-sm overflow-x-auto">
                      {JSON.stringify(endpoint.response, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Code Examples */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            Code Examples
          </h2>
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
                Python
              </h3>
              <div className="bg-gray-900 rounded-lg p-4">
                <pre className="text-green-400 text-sm overflow-x-auto">
                  {codeExamples.python}
                </pre>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
                JavaScript
              </h3>
              <div className="bg-gray-900 rounded-lg p-4">
                <pre className="text-green-400 text-sm overflow-x-auto">
                  {codeExamples.javascript}
                </pre>
              </div>
            </div>
          </div>
        </div>

        {/* Live API Testing */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            Live API Testing
          </h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Test Request
              </h3>
              <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg">
                <pre className="text-sm text-gray-900 dark:text-white overflow-x-auto">
                  {`POST /inference
Content-Type: application/json

{
  "input_data": "Test input for analysis",
  "input_type": "text",
  "confidence_threshold": 0.5
}`}
                </pre>
              </div>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Expected Response
              </h3>
              <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg">
                <pre className="text-sm text-gray-900 dark:text-white overflow-x-auto">
                  {`{
  "neural_output": {
    "label": "TEST_RESULT",
    "confidence": 0.85
  },
  "symbols": ["TEST_SYMBOL"],
  "reasoning_chain": [...],
  "confidence_score": 0.82
}`}
                </pre>
              </div>
            </div>
          </div>
        </div>

        {/* Support */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            Support
          </h2>
          <div className="grid md:grid-cols-3 gap-6 text-sm">
            <div>
              <div className="font-semibold text-gray-900 dark:text-white mb-2">
                Documentation
              </div>
              <div className="text-gray-600 dark:text-gray-400">
                Full documentation available with examples
              </div>
            </div>
            <div>
              <div className="font-semibold text-gray-900 dark:text-white mb-2">
                Issues
              </div>
              <div className="text-gray-600 dark:text-gray-400">
                Report bugs and issues on GitHub repository
              </div>
            </div>
            <div>
              <div className="font-semibold text-gray-900 dark:text-white mb-2">
                Contact
              </div>
              <div className="text-gray-600 dark:text-gray-400">
                mrchandansharma25@gmail.com
              </div>
            </div>
          </div>
        </div>

        {/* Our Platforms CTA */}
        <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-xl shadow-lg p-8">
          <h2 className="text-2xl font-bold text-white mb-4 text-center">
            Explore Our Other Platforms
          </h2>
          <p className="text-green-100 text-center mb-6 max-w-2xl mx-auto">
            Discover more developer tools and platforms built with the same
            attention to detail and performance.
          </p>
          <div className="grid md:grid-cols-2 gap-6 max-w-2xl mx-auto">
            <a
              href="https://bug3.net"
              target="_blank"
              rel="noopener noreferrer"
              className="bg-white/20 hover:bg-white/30 transition-colors p-6 rounded-lg text-white text-center"
            >
              <div className="font-bold text-lg mb-2">Bug3.net</div>
              <div className="text-sm opacity-90">
                Advanced debugging and testing platform for modern applications
              </div>
            </a>

            <a
              href="https://coderspae.com"
              target="_blank"
              rel="noopener noreferrer"
              className="bg-white/20 hover:bg-white/30 transition-colors p-6 rounded-lg text-white text-center"
            >
              <div className="font-bold text-lg mb-2">Coderspae.com</div>
              <div className="text-sm opacity-90">
                Comprehensive developer tools and code optimization platform
              </div>
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
