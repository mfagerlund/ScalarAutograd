import { useEffect, useRef } from 'react';

interface DemoCanvasProps {
  width: number;
  height: number;
  onRender: (ctx: CanvasRenderingContext2D) => void;
  className?: string;
}

export function DemoCanvas({ width, height, onRender, className = '' }: DemoCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // High DPI support
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    // Enable anti-aliasing
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    onRender(ctx);
  }, [width, height, onRender]);

  return (
    <canvas
      ref={canvasRef}
      className={`demo-canvas ${className}`}
      style={{ display: 'block' }}
    />
  );
}
