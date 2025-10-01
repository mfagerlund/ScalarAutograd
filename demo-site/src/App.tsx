import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { circleFormationDemo } from './demos/CircleFormation';
import { robotArmIKDemo } from './demos/RobotArmIK';
import { selfAvoidingRopeDemo } from './demos/SelfAvoidingRope';
import { registerDemo, getAllDemos } from './demos';
import type { Demo } from './demos/types';
import './App.css';

// Register demos
registerDemo(circleFormationDemo);
registerDemo(robotArmIKDemo);
registerDemo(selfAvoidingRopeDemo);

const allDemos = getAllDemos();

function App() {
  const [currentDemo, setCurrentDemo] = useState<Demo>(allDemos[0]);

  return (
    <div className="app">
      <header className="app-header">
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          ScalarAutograd
          <span className="subtitle">Interactive Demonstrations</span>
        </motion.h1>

        <div className="demo-selector">
          {allDemos.map((demo) => (
            <button
              key={demo.metadata.id}
              className={currentDemo.metadata.id === demo.metadata.id ? 'active' : ''}
              onClick={() => setCurrentDemo(demo)}
            >
              {demo.metadata.title}
            </button>
          ))}
        </div>
      </header>

      <main className="app-main">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentDemo.metadata.id}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.3 }}
            className="demo-wrapper"
          >
            <currentDemo.Component width={800} height={400} />
          </motion.div>
        </AnimatePresence>
      </main>

      <footer className="app-footer">
        <p>
          Powered by{' '}
          <a href="https://github.com/yourusername/ScalarAutograd" target="_blank" rel="noopener noreferrer">
            ScalarAutograd
          </a>
        </p>
      </footer>
    </div>
  );
}

export default App;
