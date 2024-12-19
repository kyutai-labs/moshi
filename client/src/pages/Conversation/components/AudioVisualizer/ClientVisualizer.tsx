import { FC, RefObject, useCallback, useEffect, useRef, useState } from "react";
import { clamp } from "../../hooks/audioUtils";

type AudioVisualizerProps = {
  analyser: AnalyserNode | null;
  parent: RefObject<HTMLElement>;
  copyCanvasRef: RefObject<HTMLCanvasElement>;
};

const MAX_INTENSITY = 255;

const COLORS = [
  "#197556",
  "#299e77",
  "#32b89b",
  "#31d4b8",
  "#14d9d5",
  "#41eff2",
  "#7ff3f5",
  "#789bf5",
  "#eb94eb",
  "#e63280",
  "#c41862",
];

export const ClientVisualizer: FC<AudioVisualizerProps> = ({ analyser, parent, copyCanvasRef }) => {
  const [canvasWidth, setCanvasWidth] = useState(parent.current ? Math.min(parent.current.clientWidth, parent.current.clientHeight) : 0);
  const requestRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const drawBars = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      x: number,
      y: number,
      volume: number,
      height: number,
      width: number,
      gap: number,
    ) => {
      const barHeight = height / 10 - gap;
      for (let i = 1; i <= 10; i++) {
        const barY = y + height + gap + Math.min(1, width / 30) - (i * barHeight + i * gap);
        ctx.fillStyle = COLORS[i - 1];
        ctx.strokeStyle = "white";
        ctx.lineWidth = Math.min(1, height / 100);
        if (i <= volume) {
          ctx.fillRect(x, barY, width, barHeight);
        }
        ctx.strokeRect(x, barY, width, barHeight);
      }
    },
    [],
  );

  const draw = useCallback((ctx: CanvasRenderingContext2D, audioData: Uint8Array, x: number, y: number, width: number, height: number) => {
    const stereoGap = Math.floor(width / 30);
    const barGap = Math.floor(height / 30);
    const padding = Math.floor(width / 30);
    const maxBarHeight = Math.floor(height - padding * 2);
    const maxBarWidth = Math.floor(
      width / 2.5 - stereoGap - padding * 2,
    );

    const centerX = x + width / 2;
    const averageIntensity = Math.sqrt(
      audioData.reduce((acc, curr) => acc + curr * curr, 0) / audioData.length,
    );
    const intensity = clamp(
      averageIntensity * 1.4,
      averageIntensity,
      MAX_INTENSITY,
    );
    const volume = Math.floor((intensity * 10) / MAX_INTENSITY);
    ctx.fillStyle = "rgba(0, 0, 0, 0)";
    ctx.fillRect(x, y, width, height);
    drawBars(
      ctx,
      centerX - maxBarWidth - stereoGap / 2,
      y,
      volume,
      maxBarHeight,
      maxBarWidth,
      barGap,
    );
    drawBars(
      ctx,
      centerX + stereoGap / 2,
      y,
      volume,
      maxBarHeight,
      maxBarWidth,
      barGap,
    );
  }, [analyser, drawBars]);

  const visualizeData = useCallback(() => {
    const width = parent.current ? Math.min(parent.current.clientWidth, parent.current.clientHeight) : 0
    if (width !== canvasWidth) {
      console.log("Setting canvas width");
      setCanvasWidth(width);
    }
    requestRef.current = window.requestAnimationFrame(() => visualizeData());
    if (!canvasRef.current) {
      console.log("Canvas not found");
      return;
    }
    const audioData = new Uint8Array(140);
    analyser?.getByteFrequencyData(audioData);

    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) {
      console.log("Canvas context not found");
      return;
    }
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    draw(ctx, audioData, 0, 0, width, width);
    if (copyCanvasRef?.current) {
      const copyCtx = copyCanvasRef.current.getContext("2d");
      if (copyCtx) {
        copyCtx.clearRect(220, 40, 140, 180);
        draw(copyCtx, audioData, 220, 40, 140, 180);
      }
    }
  }, [analyser, canvasWidth, drawBars, parent, copyCanvasRef, draw]);

  useEffect(() => {
    visualizeData();
    return () => {
      if (requestRef.current) {
        console.log("Canceling animation frame");
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [visualizeData, analyser]);
  return (
    <canvas
      ref={canvasRef}
      className="max-h-full max-w-full"
      width={canvasWidth}
      height={canvasWidth}
    />
  );
};
