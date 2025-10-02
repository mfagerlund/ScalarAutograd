import { motion, AnimatePresence } from 'framer-motion';

export interface OptimizationMetrics {
  loss: number;
  iteration: number;
  converged: boolean;
  gradientNorm?: number;
}

interface MetricsDisplayProps {
  metrics: OptimizationMetrics;
}

export function MetricsDisplay({ metrics }: MetricsDisplayProps) {
  const statusColor = metrics.converged ? '#10b981' : '#f59e0b';

  return (
    <motion.div
      className="metrics-display"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className="metric">
        <span className="metric-label">Loss:</span>
        <motion.span
          className="metric-value"
          key={metrics.loss}
          initial={{ scale: 1.2, color: '#6366f1' }}
          animate={{ scale: 1, color: '#f1f5f9' }}
        >
          {metrics.loss.toExponential(3)}
        </motion.span>
      </div>

      <div className="metric">
        <span className="metric-label">Iteration:</span>
        <motion.span
          className="metric-value"
          key={metrics.iteration}
        >
          {metrics.iteration}
        </motion.span>
      </div>

      {metrics.gradientNorm !== undefined && (
        <div className="metric">
          <span className="metric-label">Gradient:</span>
          <motion.span className="metric-value">
            {metrics.gradientNorm.toExponential(3)}
          </motion.span>
        </div>
      )}

      <AnimatePresence>
        {metrics.converged && (
          <motion.div
            className="convergence-badge"
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            exit={{ scale: 0, rotate: 180 }}
            style={{ color: statusColor }}
          >
            ✓ Converged
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
