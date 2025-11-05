export function Footer() {
  return (
    <footer className="border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950">
      <div className="container mx-auto px-6 py-8">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center mb-4 md:mb-0">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-r from-blue-600 to-purple-600 flex items-center justify-center text-white font-bold text-sm mr-3">
              NS
            </div>
            <span className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Neurosymbolic AI Framework
            </span>
          </div>

          <div className="flex space-x-6">
            <a
              href="https://github.com/Chandan25sharma"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 text-sm transition-colors"
            >
              GitHub
            </a>
            <a
              href="https://portfolio-chandan-sharma.vercel.app/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 text-sm transition-colors"
            >
              Portfolio
            </a>
            <a
              href="mailto:mrchandansharma25@gmail.com"
              className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 text-sm transition-colors"
            >
              Contact
            </a>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t border-gray-200 dark:border-gray-800 text-center text-sm text-gray-500 dark:text-gray-400">
          <p>
            © {new Date().getFullYear()} Chandan Sharma - Neurosymbolic AI
            Framework. Combining neural networks with symbolic reasoning.
          </p>
        </div>
      </div>
    </footer>
  );
}
