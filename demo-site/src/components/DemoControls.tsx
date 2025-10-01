import { motion } from 'framer-motion';

interface DemoControlsProps {
  isPlaying: boolean;
  onPlayPause: () => void;
  onReset?: () => void;
  speed: number;
  onSpeedChange: (speed: number) => void;
}

export function DemoControls({
  isPlaying,
  onPlayPause,
  speed,
  onSpeedChange,
}: DemoControlsProps) {
  return (
    <motion.div
      className="demo-controls"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="control-buttons">
        <motion.button
          className="control-btn play"
          onClick={onPlayPause}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          ▶
        </motion.button>
      </div>

      <div className="speed-control">
        <label>Speed: {speed.toFixed(1)}x</label>
        <input
          type="range"
          min="0.1"
          max="10"
          step="0.1"
          value={speed}
          onChange={(e) => onSpeedChange(parseFloat(e.target.value))}
        />
      </div>
    </motion.div>
  );
}
