import type { OptimizationMetrics } from '../components/MetricsDisplay';

export interface DemoMetadata {
  id: string;
  title: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  tags: string[];
}

export interface DemoProps {
  width: number;
  height: number;
  onMetrics?: (metrics: OptimizationMetrics) => void;
}

export interface Demo {
  metadata: DemoMetadata;
  Component: React.FC<DemoProps>;
}
