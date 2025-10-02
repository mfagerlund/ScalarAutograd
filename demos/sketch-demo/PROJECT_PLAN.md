# Parametric Sketch Solver Demo - Project Plan

## Overview
A **juicy**, interactive parametric sketch editor inspired by Fusion 360's sketch mode, showcasing ScalarAutograd's nonlinear least squares solver. Users create geometric entities (points, lines, circles) and apply constraints, with the system continuously solving to maintain constraint satisfaction while allowing interactive manipulation.

## Core Entities

### Point
- Position: `(x, y)`
- Can be free or constrained
- Visual: Circle with radius based on zoom level

### Line
- Connects two points (start, end)
- Constraints:
  - **Horizontal**: `y1 = y2`
  - **Vertical**: `x1 = x2`
  - **Fixed Length**: `distance(p1, p2) = length`
  - **Free**: No intrinsic constraints

### Circle
- Center point + radius
- Constraints:
  - **Fixed Radius**: `radius = value`
  - **Free Radius**: Solver decides

## Inter-Entity Constraints

1. **Coincident** (implicit): Shared points between entities
2. **Collinear**: Two lines share the same infinite line
3. **Parallel**: Two lines have same direction
4. **Perpendicular**: Two lines at 90°
5. **Angle**: Two lines at specified angle (degrees)
6. **Point-on-Line**: Point lies anywhere on line ray
7. **Point-on-Circle**: Point lies on circle perimeter
8. **Tangent**: Line is tangent to circle
9. **Radial-Alignment**: Point, circle center, and another point are collinear (for tangent normal constraints)

## Solver Behavior

- **Continuous solving**: Runs on every change (entity creation, constraint addition, drag movement)
- **Drag interaction**: Mouse position acts as a temporary "attractor" constraint with high weight - system solves in real-time as mouse moves, restricting point to solvable positions
- **Free entities**: Maintain current position unless influenced by constraints
- **Over-constrained detection**: Show error state visually, halt solving, display message
- **Under-constrained**: Allow solver to find valid solution (may have multiple solutions)

## User Interactions

### Selection
- Click to select (highlight)
- Shift+click for multi-select
- Click empty space to deselect all
- Selected entities have glow/highlight effect

### Creation (Toolbar + Keyboard)
- **Point** (P): Click to place
- **Line** (L): Click point → click point (creates new point if clicking empty space)
- **Circle** (C): Click center point → drag for radius (or click second point for radius)

### Manipulation
- **Drag**: Click and drag points/entities (solver maintains constraints in real-time)
- **Delete** (Delete/Backspace): Remove selected entities (cascade delete connected entities)

### Constraints (Context-Sensitive UI)
- Select entities → constraint buttons appear based on selection:
  - 1 line: Horizontal, Vertical, Fix Length
  - 2 lines: Parallel, Perpendicular, Angle, Collinear
  - 1 point + 1 line: Point-on-Line
  - 1 point + 1 circle: Point-on-Circle
  - 1 line + 1 circle: Tangent
  - 2 points + 1 circle: Radial-Alignment

### Properties Panel
- Selected entity/constraint shows editable values:
  - Line length (if fixed)
  - Circle radius (if fixed)
  - Angle value
- Edit numerically → solver updates

## Visual Design (Make it JUICY)

### Animations & Feedback
- **Smooth solving**: Animate entity positions during solve (spring interpolation)
- **Constraint satisfaction pulse**: Briefly flash green when constraint becomes satisfied
- **Selection**: Smooth glow animation, scale pulse on select
- **Hover effects**: Entities brighten/scale slightly
- **Drag**: Entity follows cursor with slight lag (easing), shows "ghost" at rest position
- **Creation**: Rubber-band effect for lines/circles during creation
- **Delete**: Fade out + slight rotation

### Visual Style
- **Entities**:
  - Points: Filled circles, larger when selected
  - Lines: Clean strokes, thicker when selected
  - Circles: Dashed or solid stroke
- **Constraints**:
  - Constraint symbols near entities (⊥ for perpendicular, ∥ for parallel, etc.)
  - Dimension annotations for fixed lengths/angles
- **Colors**:
  - Default: Cool blue/gray
  - Selected: Bright cyan/electric blue
  - Constrained: Subtle green tint
  - Over-constrained: Red
  - Hover: Yellow/gold highlight
- **Grid**: Subtle dot grid, snaps for precise placement (toggleable)

### Particle Effects
- **Constraint added**: Small particle burst at constraint location
- **Solve success**: Subtle shimmer across satisfied constraints
- **Over-constrained**: Red warning particles

## Project Structure

```
demos/sketch-demo/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── index.html
├── src/
│   ├── main.tsx                    # React entry
│   ├── App.tsx                     # Main app component
│   ├── types/
│   │   ├── Entities.ts             # Point, Line, Circle types
│   │   ├── Constraints.ts          # Constraint type definitions
│   │   └── SketchState.ts          # Overall state type
│   ├── solver/
│   │   ├── SketchSolver.ts         # Main solver interface
│   │   ├── ConstraintBuilder.ts    # Convert constraints to residuals
│   │   └── SolverUtils.ts          # Helpers
│   ├── ui/
│   │   ├── Canvas.tsx              # Main canvas component
│   │   ├── Toolbar.tsx             # Entity creation tools
│   │   ├── ConstraintPanel.tsx     # Context-sensitive constraints
│   │   ├── PropertiesPanel.tsx     # Edit values
│   │   └── StatusBar.tsx           # Solver status, error messages
│   ├── rendering/
│   │   ├── Renderer.ts             # Canvas drawing logic
│   │   ├── EntityRenderer.ts       # Draw entities
│   │   ├── ConstraintRenderer.ts   # Draw constraint symbols
│   │   └── AnimationSystem.ts      # Juice/animations
│   ├── interaction/
│   │   ├── SelectionManager.ts     # Handle selection state
│   │   ├── DragHandler.ts          # Drag interaction
│   │   ├── CreationTool.ts         # Entity creation state machine
│   │   └── KeyboardShortcuts.ts    # Keyboard handling
│   └── utils/
│       ├── Geometry.ts             # Vector math helpers
│       └── ConstraintDetection.ts  # Smart constraint suggestions
└── public/
```

## Development Milestones

### Milestone 1: Project Setup & Data Model
- [x] Create Vite + React + TypeScript project in `demos/sketch-demo/`
- [ ] Define entity types (Point, Line, Circle)
- [ ] Define constraint types
- [ ] Define sketch state structure
- [ ] **Review checkpoint** - Get feedback on data structures

### Milestone 2: Core Solver Integration
- [ ] Implement `SketchSolver` class
- [ ] Implement `ConstraintBuilder` to convert constraints → residual functions
- [ ] Wire up ScalarAutograd's `V.nonlinearLeastSquares()`
- [ ] Test solver with hardcoded simple scenarios
- [ ] Handle over-constrained detection

### Milestone 3: Basic Rendering
- [ ] Setup Canvas component with pan/zoom
- [ ] Implement `EntityRenderer` (draw points, lines, circles)
- [ ] Implement basic selection highlighting
- [ ] Grid background with optional snap

### Milestone 4: Entity Creation
- [ ] Implement `CreationTool` state machine
- [ ] Point creation (click to place)
- [ ] Line creation (click-click, auto-create points)
- [ ] Circle creation (click center, drag radius)
- [ ] Toolbar UI

### Milestone 5: Selection & Deletion
- [ ] Click selection (single + multi-select)
- [ ] Selection visual feedback
- [ ] Delete key handling
- [ ] Cascade deletion (delete point → delete connected entities)

### Milestone 6: Drag Interaction
- [ ] Implement `DragHandler`
- [ ] Real-time solver during drag (mouse position as constraint)
- [ ] Smooth position updates
- [ ] Prevent invalid drags (show constraint conflict)

### Milestone 7: Constraint UI
- [ ] Context-sensitive constraint panel
- [ ] Add horizontal/vertical/fixed-length to lines
- [ ] Add parallel/perpendicular/angle between lines
- [ ] Add point-on-line, point-on-circle
- [ ] Add tangent, radial-alignment

### Milestone 8: Properties Panel
- [ ] Display selected entity/constraint properties
- [ ] Editable numeric inputs (lengths, angles, radii)
- [ ] Update solver when values change

### Milestone 9: Make it JUICY 🧃
- [ ] Animation system (spring interpolation for positions)
- [ ] Selection animations (glow, pulse, scale)
- [ ] Hover effects
- [ ] Constraint satisfaction visual feedback
- [ ] Particle effects (constraint add, solve success, errors)
- [ ] Rubber-band creation visuals
- [ ] Smooth delete animations
- [ ] Constraint symbols/annotations rendering

### Milestone 10: Polish & Extras
- [ ] Undo/redo system
- [ ] Save/load sketches (JSON export/import)
- [ ] Keyboard shortcuts overlay (press ? to show)
- [ ] Smart constraint suggestions (hover shows possible constraints)
- [ ] Performance optimization (only re-solve when needed)
- [ ] Demo sketches (pre-built examples to load)

---

## Technical Notes

### Solver Integration
- All entity positions are `V.W()` (trainable parameters)
- Fixed values (lengths, radii, angles) are `V.C()` (constants)
- Drag position is temporary `V.C()` that gets removed after drag ends
- Use high residual weight for drag constraint to prioritize user intent

### Performance
- Debounce solver during rapid changes (but not during drag - that needs real-time)
- Only rebuild constraint residuals when constraints change, not when positions change
- Consider spatial hashing for large sketches

### Error Handling
- Solver divergence → freeze last valid state, show error
- Numerical instability → add small epsilon to prevent division by zero
- Over-constrained → detect via solver failure + constraint count heuristic
