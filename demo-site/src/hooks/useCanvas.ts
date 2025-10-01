import { useEffect, useRef } from 'react';

/**
 * Hook for canvas setup and cleanup
 */
export function useCanvas(
  onRender: (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => void,
  deps: React.DependencyList = []
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set up high DPI rendering
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    onRender(ctx, canvas);
  }, [onRender, ...deps]);

  return canvasRef;
}
