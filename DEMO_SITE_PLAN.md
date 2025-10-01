# ScalarAutograd Interactive Demo Site

## Vision
A visually stunning, interactive showcase of ScalarAutograd solving various optimization problems in real-time. Each demo should be "juicy" with smooth animations, satisfying visual feedback, and clear demonstration of the optimization process.

## Tech Stack
- **React + TypeScript** - Component-based architecture for easy demo addition
- **Canvas API** - High-performance rendering for smooth animations
- **Framer Motion** - Smooth page transitions and UI animations
- **ScalarAutograd** - The star of the show (bundled for browser)

## Project Structure
```
demo-site/
├── src/
│   ├── demos/
│   │   ├── index.ts                    # Demo registry
│   │   ├── CircleFormation/
│   │   │   ├── CircleFormation.tsx     # Demo component
│   │   │   ├── CircleFormation.worker.ts # Heavy computation
│   │   │   ├── metadata.ts             # Title, description, tags
│   │   │   └── README.md               # Technical explanation
│   │   ├── SpringSystem/
│   │   ├── CurveApproximation/
│   │   ├── TravelingSalesman/
│   │   └── NeuralNetworkTraining/
│   ├── components/
│   │   ├── DemoCanvas.tsx              # Reusable canvas wrapper
│   │   ├── DemoControls.tsx            # Play/pause/reset/speed
│   │   ├── OptimizerSelector.tsx       # SGD/Adam/AdamW toggle
│   │   ├── MetricsDisplay.tsx          # Loss, iterations, etc.
│   │   └── Navigation/
│   ├── hooks/
│   │   ├── useAnimationFrame.ts        # Smooth 60fps loop
│   │   ├── useOptimizer.ts             # Shared optimization logic
│   │   └── useCanvas.ts                # Canvas setup/cleanup
│   ├── lib/
│   │   └── scalar-autograd.ts          # Browser build
│   ├── App.tsx
│   └── main.tsx
├── public/
└── package.json
```

## Demo Ideas (Ordered by Complexity)

### 2. **Spring Mass System**
- **Visual**: Connected masses with springs, gravity pulling down
- **Goal**: Find equilibrium positions that minimize energy
- **Juicy**: Springs stretch/compress with color gradient (red=tension, blue=compression)
- **Fun**: User can drag points and watch system re-optimize

### 3. **Curve Approximation**
- **Visual**: Scatter points and a smooth curve fitting through them
- **Goal**: Fit polynomial/bezier curve to noisy data
- **Juicy**: Curve morphs smoothly, residual lines shrink to zero
- **Interactive**: User can add/remove points by clicking

### 5. **Traveling Salesman (Simplified)**
- **Visual**: Cities as circles, path as lines
- **Goal**: Minimize total travel distance
- **Juicy**: Path animates improvements, cities pulse when visited
- **Note**: Use continuous relaxation for gradient-based approach

### 6. **Neural Network Training Visualization**
- **Visual**: 2D classification boundary morphing as network trains
- **Goal**: Classify spiral/circle/XOR dataset
- **Juicy**: Decision boundary animates, misclassified points highlighted
- **Live**: Loss graph updates in real-time

### 7. **Circle Packing**
- **Visual**: Circles of various sizes trying to fit in container
- **Goal**: Minimize overlap while maximizing space usage
- **Juicy**: Circles gently push each other, bounce when colliding

### 8. **Catenary Curve (Hanging Chain)**
- **Visual**: Chain hanging between two points under gravity
- **Goal**: Find natural curve that minimizes potential energy
- **Juicy**: Chain sways gently into final position

### 9. **Planetary Orbit Optimization**
- **Visual**: Planet trying to find stable orbit around sun
- **Goal**: Minimize energy while maintaining angular momentum
- **Juicy**: Orbit trail fades over time, sun glows

### 10. **Image Reconstruction**
- **Visual**: Pixelated/blurred image being optimized
- **Goal**: Reconstruct image from partial information
- **Juicy**: Image gradually comes into focus, pixel-by-pixel

## Core UI Components

### Demo Template Structure
```typescript
interface DemoMetadata {
  id: string;
  title: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  tags: string[];
  thumbnail: string;
}

interface DemoProps {
  width: number;
  height: number;
  onMetrics?: (metrics: OptimizationMetrics) => void;
}

// Each demo exports these
export const metadata: DemoMetadata;
export const Demo: React.FC<DemoProps>;
```

### Shared Controls
- **Play/Pause/Reset** - Large, satisfying buttons
- **Speed Slider** - 0.1x to 10x (with smooth easing)
- **Optimizer Toggle** - SGD | Adam | AdamW (with smooth slide animation)
- **Learning Rate** - Live adjustable
- **Iterations Counter** - Smooth number counting animation
- **Loss Graph** - Minimalist line chart with gradient fill

### Navigation
- **Grid View** - Card-based gallery of all demos
- **Category Filter** - Physics, ML, Geometry, etc.
- **Difficulty Badge** - Color-coded
- **Smooth Page Transitions** - Framer Motion shared layout animations

## Visual Design Principles

### Color Palette
```css
--primary: #6366f1;      /* Indigo */
--success: #10b981;      /* Emerald (converged) */
--warning: #f59e0b;      /* Amber (optimizing) */
--error: #ef4444;        /* Red (constraints violated) */
--background: #0f172a;   /* Dark blue-gray */
--surface: #1e293b;      /* Lighter gray */
--text: #f1f5f9;         /* Off-white */
```

### Animation Guidelines
- **Easing**: Use `ease-out` for optimization movements (natural deceleration)
- **Duration**: 16ms updates (60fps), 300ms UI transitions
- **Springs**: Use Framer Motion springs for organic movement
- **Feedback**: Subtle particle effects when converged

### Canvas Rendering
```typescript
// Every demo follows this pattern
class DemoRenderer {
  // Smooth interpolation between optimization steps
  private interpolate(t: number): State { }

  // Draw with anti-aliasing and shadows
  private draw(ctx: CanvasRenderingContext2D): void { }

  // Highlight interesting features (constraints, gradients)
  private drawDebugOverlay(ctx: CanvasRenderingContext2D): void { }
}
```

## Performance Optimizations

1. **Web Workers**: Run optimization in separate thread
2. **RequestAnimationFrame**: Smooth 60fps rendering
3. **Canvas Pooling**: Reuse canvas contexts
4. **Lazy Loading**: Load demos on demand
5. **Memoization**: Cache expensive calculations

## Adding a New Demo (Developer Experience)

```bash
# 1. Generate demo scaffold
npm run create-demo SpringMass

# Creates:
# - src/demos/SpringMass/SpringMass.tsx
# - src/demos/SpringMass/metadata.ts
# - src/demos/SpringMass/README.md
# - Automatically registers in src/demos/index.ts
```

```typescript
// 2. Implement demo (minimal boilerplate)
export const Demo: React.FC<DemoProps> = ({ width, height }) => {
  const { optimize, reset, metrics } = useOptimizer({
    initialize: () => createInitialState(),
    objective: (state) => computeLoss(state),
    onStep: (state) => setState(state),
  });

  return (
    <DemoCanvas
      width={width}
      height={height}
      onRender={(ctx) => render(ctx, state)}
    />
  );
};
```

## Deployment
- **Vercel/Netlify** - Zero-config deployment
- **GitHub Pages** - Alternative with custom domain
- **Bundle Size** - Target < 500KB total (tree-shaking, compression)

## Success Metrics
1. **Visual Appeal**: "Wow" factor when first loaded
2. **Educational**: Clear demonstration of optimization concepts
3. **Performance**: Consistent 60fps on mid-range devices
4. **Extensibility**: New demo added in < 2 hours
5. **Shareability**: Each demo has unique URL, shareable GIFs

## Phase 1 (MVP)
- [ ] Project setup (React + Vite + TypeScript)
- [ ] Bundle ScalarAutograd for browser
- [ ] Create demo template/scaffold
- [ ] Implement Circle Formation demo
- [ ] Basic UI (play/pause/reset)
- [ ] Deploy to Vercel

## Phase 2 (Polish)
- [ ] Add 3-4 more demos
- [ ] Optimizer comparison view
- [ ] Smooth animations and transitions
- [ ] Metrics dashboard
- [ ] Navigation/gallery view

## Phase 3 (Advanced)
- [ ] Remaining demos
- [ ] Interactive parameter tuning
- [ ] Code view/export
- [ ] Performance profiling overlay
- [ ] Social sharing (GIF generation)

## Inspiration
- **TensorFlow Playground** - Clean, educational
- **Desmos Calculator** - Smooth, responsive
- **Three.js Examples** - Gallery navigation
- **Apple Product Pages** - Buttery transitions
- **Lenia** - Mesmerizing visual feedback
