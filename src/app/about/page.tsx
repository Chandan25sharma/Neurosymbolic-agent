export default function AboutPage() {
  const team = [
    {
      name: "Chandan Sharma",
      role: "Lead Developer & AI Researcher",
      description:
        "Full-stack developer and AI enthusiast with expertise in neurosymbolic systems, computer vision, and distributed systems.",
      contact: "mrchandansharma25@gmail.com",
    },
  ];

  const features = [
    {
      title: "Neural Processing",
      description:
        "Advanced deep learning models for text and image analysis using transformer architectures and convolutional neural networks.",
    },
    {
      title: "Symbolic Reasoning",
      description:
        "Logical inference engine that applies rule-based systems and knowledge graphs to neural outputs.",
    },
    {
      title: "Explainable AI",
      description:
        "Transparent decision-making with human-readable explanations for every inference result.",
    },
    {
      title: "Real-time Analysis",
      description:
        "High-performance processing with sub-50ms response times for both text and image inputs.",
    },
  ];

  const techStack = [
    {
      category: "Backend",
      items: ["FastAPI", "Python", "PyTorch", "Z3 Theorem Prover"],
    },
    {
      category: "Frontend",
      items: ["Next.js", "TypeScript", "Tailwind CSS", "Framer Motion"],
    },
    {
      category: "AI/ML",
      items: ["Transformers", "CNN", "Knowledge Graphs", "Rule-based Systems"],
    },
    {
      category: "Infrastructure",
      items: ["Docker", "Redis", "PostgreSQL", "WebSockets"],
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-900 dark:to-gray-800 py-12">
      <div className="container mx-auto px-6 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4">
            About Neurosymbolic AI Framework
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
            Bridging the gap between neural networks and symbolic AI to create
            transparent, trustworthy, and explainable artificial intelligence
            systems.
          </p>
        </div>

        {/* Mission Section */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-8 mb-12">
          <div className="grid md:grid-cols-2 gap-8 items-center">
            <div>
              <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
                Our Mission
              </h2>
              <p className="text-lg text-gray-600 dark:text-gray-400 mb-6">
                We believe that the future of AI lies in combining the pattern
                recognition capabilities of neural networks with the logical
                reasoning of symbolic AI. Our framework enables developers to
                build AI systems that are not only powerful but also transparent
                and explainable.
              </p>
              <div className="space-y-3">
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                  <span className="text-gray-700 dark:text-gray-300">
                    Transparent decision-making processes
                  </span>
                </div>
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                  <span className="text-gray-700 dark:text-gray-300">
                    Human-understandable explanations
                  </span>
                </div>
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                  <span className="text-gray-700 dark:text-gray-300">
                    Robust and reliable AI systems
                  </span>
                </div>
              </div>
            </div>
            <div className="bg-gradient-to-br from-green-50 to-blue-50 dark:from-gray-700 dark:to-gray-600 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                Why Neurosymbolic AI?
              </h3>
              <p className="text-gray-600 dark:text-gray-400">
                Traditional neural networks are powerful but often act as "black
                boxes." Symbolic AI provides transparency but lacks learning
                capabilities. By combining both approaches, we create AI systems
                that learn from data while maintaining explainability and
                logical consistency.
              </p>
            </div>
          </div>
        </div>

        {/* Features Section */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white text-center mb-8">
            Key Features
          </h2>
          <div className="grid md:grid-cols-2 gap-6">
            {features.map((feature, index) => (
              <div
                key={index}
                className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 hover:shadow-lg transition-shadow"
              >
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                  {feature.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Technology Stack */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-8 mb-12">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white text-center mb-8">
            Technology Stack
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {techStack.map((stack, index) => (
              <div key={index} className="text-center">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  {stack.category}
                </h3>
                <div className="space-y-2">
                  {stack.items.map((item, itemIndex) => (
                    <div
                      key={itemIndex}
                      className="bg-gray-100 dark:bg-gray-700 rounded-lg py-2 px-3 text-sm text-gray-700 dark:text-gray-300"
                    >
                      {item}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Architecture Overview */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white text-center mb-8">
            System Architecture
          </h2>
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <div className="grid md:grid-cols-4 gap-4 text-center">
              <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <div className="font-semibold text-green-700 dark:text-green-300 mb-2">
                  Input Layer
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Text & Image Processing
                </div>
              </div>
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <div className="font-semibold text-blue-700 dark:text-blue-300 mb-2">
                  Neural Processing
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Deep Learning Models
                </div>
              </div>
              <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                <div className="font-semibold text-purple-700 dark:text-purple-300 mb-2">
                  Symbolic Mapping
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Knowledge Graphs
                </div>
              </div>
              <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                <div className="font-semibold text-orange-700 dark:text-orange-300 mb-2">
                  Reasoning Engine
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Logical Inference
                </div>
              </div>
            </div>
            <div className="mt-6 text-center">
              <p className="text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
                Our pipeline processes input through neural networks, converts
                outputs to symbolic representations, applies logical reasoning,
                and generates comprehensive explanations for each decision.
              </p>
            </div>
          </div>
        </div>

        {/* Team Section */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-8 mb-12">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white text-center mb-8">
            Development Team
          </h2>
          <div className="max-w-2xl mx-auto">
            {team.map((member, index) => (
              <div key={index} className="text-center">
                <div className="w-24 h-24 bg-gradient-to-br from-green-400 to-blue-500 rounded-full mx-auto mb-4 flex items-center justify-center text-white text-2xl font-bold">
                  CS
                </div>
                <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-2">
                  {member.name}
                </h3>
                <p className="text-lg text-green-600 dark:text-green-400 mb-3">
                  {member.role}
                </p>
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  {member.description}
                </p>
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  Contact: {member.contact}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Our Platforms */}
        <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-xl shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-bold text-white text-center mb-4">
            Explore Our Other Platforms
          </h2>
          <p className="text-green-100 text-center mb-6 max-w-2xl mx-auto">
            Discover more innovative projects and developer tools from our
            ecosystem.
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
                with real-time error tracking.
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
                Comprehensive suite of developer tools for code optimization and
                performance analysis.
              </div>
            </a>
          </div>
        </div>

        {/* Contact & Links */}
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
          <div className="grid md:grid-cols-3 gap-6 text-center">
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                Documentation
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Complete API documentation and usage guides
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                GitHub
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Open source components and examples
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                Support
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                mrchandansharma25@gmail.com
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
