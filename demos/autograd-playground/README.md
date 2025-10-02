# ScalarAutograd Demo Site

Interactive demonstrations of ScalarAutograd optimization algorithms in action.

## Features

- **Circle Formation Demo**: Watch 25 points smoothly align into a perfect circle
- **Optimizer Comparison**: Switch between SGD, Adam, and AdamW in real-time
- **Smooth Animations**: 60fps Canvas rendering with Framer Motion transitions
- **Live Metrics**: Track loss, iterations, and convergence in real-time
- **Interactive Controls**: Play/pause, reset, and adjust optimization speed

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
demo-site/
├── src/
│   ├── demos/              # Self-contained demo modules
│   │   ├── CircleFormation/
│   │   │   ├── CircleFormation.tsx
│   │   │   ├── metadata.ts
│   │   │   └── index.ts
│   │   ├── types.ts        # Shared demo types
│   │   └── index.ts        # Demo registry
│   ├── components/         # Reusable UI components
│   │   ├── DemoCanvas.tsx
│   │   ├── DemoControls.tsx
│   │   └── MetricsDisplay.tsx
│   ├── hooks/             # Custom React hooks
│   │   ├── useAnimationFrame.ts
│   │   └── useCanvas.ts
│   ├── lib/               # ScalarAutograd library
│   │   └── scalar-autograd.ts
│   ├── App.tsx
│   ├── App.css
│   └── main.tsx
```

## Adding a New Demo

1. Create a new directory in `src/demos/YourDemo/`
2. Create three files:
   - `YourDemo.tsx` - The demo component
   - `metadata.ts` - Demo metadata (title, description, tags)
   - `index.ts` - Export the demo

3. Register your demo in `src/App.tsx`:

```typescript
import { yourDemo } from './demos/YourDemo';
registerDemo(yourDemo);
```

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Framer Motion** - Smooth animations
- **Canvas API** - High-performance rendering
- **ScalarAutograd** - Automatic differentiation engine

## Live Demo

Visit the site at http://localhost:5173 during development.

## Phase 1 (MVP) ✅

- [x] Project setup (React + Vite + TypeScript)
- [x] Bundle ScalarAutograd for browser
- [x] Create demo template/scaffold
- [x] Implement Circle Formation demo
- [x] Basic UI (play/pause/reset)
- [x] Optimizer comparison (SGD, Adam, AdamW)
- [x] Smooth animations and visual feedback

## Next Steps (Phase 2)

- [ ] Add 3-4 more demos (Spring System, Curve Fitting, etc.)
- [ ] Create demo gallery/navigation
- [ ] Add loss graph visualization
- [ ] Implement demo thumbnails
- [ ] Deploy to Vercel/Netlify
