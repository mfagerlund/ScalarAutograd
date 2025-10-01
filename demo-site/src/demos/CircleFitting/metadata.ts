import type { DemoMetadata } from '../types';

export const metadata: DemoMetadata = {
  id: 'circle-fitting',
  title: 'Circle Fitting with Levenberg-Marquardt',
  description: 'Fit a circle to noisy data points using the nonlinear least squares solver. Watch the LM algorithm converge 100-1000x faster than gradient descent.',
  difficulty: 'intermediate',
  tags: ['least-squares', 'optimization', 'curve-fitting'],
};
