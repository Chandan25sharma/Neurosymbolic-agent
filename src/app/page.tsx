"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { motion } from "framer-motion";
import Link from "next/link";

export default function Dashboard() {
  const features = [
    {
      title: "Neural Processing",
      description:
        "Advanced text and image classification using neural networks",
      gradient: "from-green-600 to-green-800",
    },
    {
      title: "Symbolic Reasoning",
      description: "Logical inference and rule-based reasoning systems",
      gradient: "from-green-700 to-green-900",
    },
    {
      title: "Explainable AI",
      description:
        "Transparent decision-making with human-readable explanations",
      gradient: "from-green-500 to-green-700",
    },
    {
      title: "Real-time Analysis",
      description:
        "Instant processing with comprehensive results visualization",
      gradient: "from-green-800 to-green-950",
    },
  ];

  const stats = [
    { value: "99.2%", label: "Accuracy" },
    { value: "≤50ms", label: "Response Time" },
    { value: "24/7", label: "Availability" },
    { value: "100+", label: "Rules" },
  ];

  return (
    <div className="min-h-screen bg-white dark:bg-black transition-colors">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-grid-black/[0.02] dark:bg-grid-white/[0.02] [mask-image:linear-gradient(0deg,white,rgba(255,255,255,0.6))] dark:[mask-image:linear-gradient(0deg,black,rgba(0,0,0,0.6))]" />
        <div className="container relative mx-auto px-6 py-24">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-6">
              <span className="bg-gradient-to-r from-green-600 to-green-800 bg-clip-text text-transparent">
                Neurosymbolic AI
              </span>
            </h1>
            <p className="text-xl text-black dark:text-white max-w-3xl mx-auto mb-8 leading-relaxed">
              Advanced AI framework combining neural pattern recognition with
              symbolic logical reasoning for transparent, trustworthy, and
              explainable artificial intelligence.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button
                size="lg"
                className="bg-green-600 hover:bg-green-700 text-white border-0"
              >
                <Link href="/analysis">Start Analysis</Link>
              </Button>
              <Button
                variant="outline"
                size="lg"
                className="border-green-600 text-green-600 hover:bg-green-600 hover:text-white dark:border-green-400 dark:text-green-400 dark:hover:bg-green-400 dark:hover:text-black"
              >
                <Link href="/documentation">View Documentation</Link>
              </Button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-white dark:bg-black border-t border-b border-gray-200 dark:border-gray-800">
        <div className="container mx-auto px-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="text-center"
              >
                <div className="text-3xl md:text-4xl font-bold text-green-600 dark:text-green-400 mb-2">
                  {stat.value}
                </div>
                <div className="text-black dark:text-white font-medium">
                  {stat.label}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gray-50 dark:bg-gray-950">
        <div className="container mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-black dark:text-white mb-4">
              Advanced AI Capabilities
            </h2>
            <p className="text-lg text-black dark:text-white max-w-2xl mx-auto">
              Our framework integrates multiple AI paradigms to deliver
              comprehensive and explainable results
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card className="h-full border-0 shadow-lg hover:shadow-xl transition-all duration-300 bg-white dark:bg-black border border-gray-200 dark:border-gray-800">
                  <CardHeader>
                    <div
                      className={`w-12 h-12 rounded-lg bg-gradient-to-r ${feature.gradient} flex items-center justify-center text-white font-bold text-lg mb-4`}
                    >
                      {index + 1}
                    </div>
                    <CardTitle className="text-xl text-black dark:text-white">
                      {feature.title}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-black dark:text-white leading-relaxed">
                      {feature.description}
                    </p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-green-600 to-green-800 dark:from-green-700 dark:to-green-900">
        <div className="container mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-4xl font-bold text-white mb-4">
              Ready to Get Started?
            </h2>
            <p className="text-xl text-green-100 mb-8 max-w-2xl mx-auto">
              Experience the power of neurosymbolic AI with our comprehensive
              analysis tools and APIs
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button
                size="lg"
                variant="secondary"
                className="bg-white text-green-600 hover:bg-gray-100 border-0"
              >
                <Link href="/analysis">Try Live Analysis</Link>
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="bg-transparent border-white text-white hover:bg-white hover:text-green-600"
              >
                <Link href="/api">Explore API</Link>
              </Button>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
}
