import { useEffect, useRef } from 'react';

/**
 * Hook for smooth 60fps animation loop
 */
export function useAnimationFrame(callback: (deltaTime: number) => void, isRunning: boolean = true) {
  const requestRef = useRef<number | undefined>(undefined);
  const previousTimeRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    if (!isRunning) {
      if (requestRef.current !== undefined) {
        cancelAnimationFrame(requestRef.current);
      }
      return;
    }

    const animate = (time: number) => {
      if (previousTimeRef.current !== undefined) {
        const deltaTime = time - previousTimeRef.current;
        callback(deltaTime);
      }
      previousTimeRef.current = time;
      requestRef.current = requestAnimationFrame(animate);
    };

    requestRef.current = requestAnimationFrame(animate);

    return () => {
      if (requestRef.current !== undefined) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [callback, isRunning]);
}
