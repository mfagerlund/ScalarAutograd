# Sketch Demo - Project Tracker

## ✅ Milestone 1: Project Setup & Data Model
- [x] Create Vite + React + TypeScript project
- [x] Install ScalarAutograd dependency
- [x] Define entity types (Point, Line, Circle) - object references only
- [x] Define 8 constraint types (all using object references)
- [x] Define complete SketchState structure
- [ ] **REVIEW CHECKPOINT** - Get feedback on data structures

---

## 📋 Milestone 2: Core Solver Integration
- [ ] `src/solver/SketchSolver.ts` - Main solver class
  - [ ] Extract all Point positions as V.W() trainables
  - [ ] Extract circle radii (if not fixed) as V.W() trainables
  - [ ] Build parameter array for LM solver
  - [ ] Solve and update entity positions from results
  - [ ] Detect over-constrained (solver failure)

- [ ] `src/solver/ConstraintBuilder.ts` - Convert constraints → residuals
  - [ ] Line constraints (horizontal, vertical, fixed length)
  - [ ] Collinear constraint
  - [ ] Parallel constraint
  - [ ] Perpendicular constraint
  - [ ] Angle constraint
  - [ ] Point-on-line constraint
  - [ ] Point-on-circle constraint
  - [ ] Tangent constraint
  - [ ] Radial-alignment constraint

- [ ] `src/solver/SolverUtils.ts` - Geometry helpers
  - [ ] Line direction vector
  - [ ] Point-line distance
  - [ ] Point-circle distance
  - [ ] Angle between vectors

- [ ] Test solver with hardcoded scenarios

---

## 🎨 Milestone 3: Basic Rendering
- [ ] `src/rendering/Renderer.ts` - Main canvas drawing orchestrator
  - [ ] Canvas setup and coordinate transforms
  - [ ] Pan/zoom support
  - [ ] Clear and redraw loop

- [ ] `src/rendering/EntityRenderer.ts` - Draw entities
  - [ ] Draw point (circle)
  - [ ] Draw line (with direction indicators)
  - [ ] Draw circle
  - [ ] Selection highlighting (glow/thicker)
  - [ ] Hover highlighting

- [ ] `src/ui/Canvas.tsx` - React canvas component
  - [ ] Canvas ref and resize handling
  - [ ] Mouse position tracking
  - [ ] Render loop integration

- [ ] Grid background with optional snap

---

## 🔨 Milestone 4: Entity Creation
- [ ] `src/interaction/CreationTool.ts` - State machine for creation
  - [ ] Point creation (single click)
  - [ ] Line creation (click-click, auto-create points)
  - [ ] Circle creation (click center, drag/click radius)
  - [ ] Preview rendering during creation

- [ ] `src/ui/Toolbar.tsx` - Tool selection UI
  - [ ] Select tool button
  - [ ] Create point button (P)
  - [ ] Create line button (L)
  - [ ] Create circle button (C)
  - [ ] Active tool indicator

- [ ] Keyboard shortcuts for tool switching

---

## 👆 Milestone 5: Selection & Deletion
- [ ] `src/interaction/SelectionManager.ts` - Selection logic
  - [ ] Click to select entity (hit testing)
  - [ ] Shift+click for multi-select
  - [ ] Click empty space to deselect
  - [ ] Hover detection

- [ ] Delete key handling
  - [ ] Remove entities from state
  - [ ] Cascade delete (point → connected lines/circles)
  - [ ] Remove invalid constraints

- [ ] Visual feedback for selection

---

## 🖱️ Milestone 6: Drag Interaction
- [ ] `src/interaction/DragHandler.ts` - Drag logic
  - [ ] Detect drag start on point/entity
  - [ ] Add temporary "mouse attractor" constraint during drag
  - [ ] Trigger solver on mouse move (real-time solving)
  - [ ] Remove temporary constraint on drag end
  - [ ] Handle drag of lines/circles (drag their points)

- [ ] Smooth position updates
- [ ] Visual feedback during drag

---

## 🔗 Milestone 7: Constraint UI
- [ ] `src/ui/ConstraintPanel.tsx` - Context-sensitive constraint panel
  - [ ] Detect valid constraints based on selection
  - [ ] Show available constraint buttons
  - [ ] Add constraint on button click

- [ ] Constraint buttons:
  - [ ] 1 line: Horizontal, Vertical, Fix Length
  - [ ] 2 lines: Parallel, Perpendicular, Angle, Collinear
  - [ ] Point + Line: Point-on-Line
  - [ ] Point + Circle: Point-on-Circle
  - [ ] Line + Circle: Tangent
  - [ ] 2 Points + Circle: Radial-Alignment

- [ ] `src/rendering/ConstraintRenderer.ts` - Draw constraint symbols
  - [ ] Perpendicular symbol (⊥)
  - [ ] Parallel symbol (∥)
  - [ ] Dimension annotations (lengths, angles)
  - [ ] Tangent indicator

---

## 📝 Milestone 8: Properties Panel
- [ ] `src/ui/PropertiesPanel.tsx` - Edit constraint values
  - [ ] Display selected entity/constraint
  - [ ] Numeric input for line fixed length
  - [ ] Numeric input for circle radius
  - [ ] Numeric input for angle constraint
  - [ ] Update entity/constraint on change
  - [ ] Trigger solver on value change

- [ ] `src/ui/StatusBar.tsx` - Solver status
  - [ ] Show solver status (idle/solving/success/failed)
  - [ ] Show error messages
  - [ ] Show diagnostics (iterations, residual, time)

---

## 🧃 Milestone 9: Make it JUICY
- [ ] `src/rendering/AnimationSystem.ts` - Animation engine
  - [ ] Spring interpolation for smooth position updates
  - [ ] Easing functions
  - [ ] Animation state tracking

- [ ] Selection animations
  - [ ] Glow effect
  - [ ] Scale pulse on select
  - [ ] Smooth transitions

- [ ] Hover effects
  - [ ] Brightness increase
  - [ ] Subtle scale

- [ ] Constraint satisfaction feedback
  - [ ] Flash green when constraint satisfied
  - [ ] Shimmer across constraints

- [ ] Particle effects
  - [ ] Constraint add burst
  - [ ] Solve success sparkle
  - [ ] Over-constrained warning particles

- [ ] Creation effects
  - [ ] Rubber-band line preview
  - [ ] Expanding circle preview

- [ ] Delete animations
  - [ ] Fade out
  - [ ] Slight rotation

---

## ✨ Milestone 10: Polish & Extras
- [ ] `src/interaction/KeyboardShortcuts.ts` - Keyboard handling
  - [ ] Tool shortcuts (P, L, C, Esc)
  - [ ] Delete key
  - [ ] Undo/Redo (Ctrl+Z, Ctrl+Shift+Z)
  - [ ] Help overlay (?)

- [ ] Undo/redo system
  - [ ] State history stack
  - [ ] Undo/redo actions

- [ ] Save/load
  - [ ] Export sketch to JSON
  - [ ] Import sketch from JSON
  - [ ] Local storage auto-save

- [ ] `src/utils/ConstraintDetection.ts` - Smart suggestions
  - [ ] Detect near-parallel lines
  - [ ] Detect near-perpendicular lines
  - [ ] Detect near-coincident points
  - [ ] Suggest constraints on hover

- [ ] Demo sketches
  - [ ] Create pre-built example sketches
  - [ ] Load demo button

- [ ] Performance optimization
  - [ ] Only re-solve when needed (dirty flag)
  - [ ] Debounce non-drag solver calls
  - [ ] Optimize rendering (only redraw on change)

---

## 📁 File Structure

```
demos/sketch-demo/
├── PROJECT.md                      ← YOU ARE HERE
├── PROJECT_PLAN.md                 ← Overall design doc
├── package.json
├── tsconfig.json
├── vite.config.ts
├── index.html
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── types/
    │   └── index.ts                ✅ DONE
    ├── solver/
    │   ├── SketchSolver.ts         ⏳ Next
    │   ├── ConstraintBuilder.ts
    │   └── SolverUtils.ts
    ├── ui/
    │   ├── Canvas.tsx
    │   ├── Toolbar.tsx
    │   ├── ConstraintPanel.tsx
    │   ├── PropertiesPanel.tsx
    │   └── StatusBar.tsx
    ├── rendering/
    │   ├── Renderer.ts
    │   ├── EntityRenderer.ts
    │   ├── ConstraintRenderer.ts
    │   └── AnimationSystem.ts
    ├── interaction/
    │   ├── SelectionManager.ts
    │   ├── DragHandler.ts
    │   ├── CreationTool.ts
    │   └── KeyboardShortcuts.ts
    └── utils/
        ├── Geometry.ts
        └── ConstraintDetection.ts
```

---

## 🎯 Current Status

**Milestone 1 Complete!** Data structures defined with object references (no IDs).

**Next:** Waiting for review feedback before proceeding to Milestone 2 (Solver Integration).
