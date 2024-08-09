import { FC, useCallback, useEffect, useRef } from "react";

type AudioVisualizerProps = {
  analyser: AnalyserNode | null;
};

export const AudioVisualizer: FC<AudioVisualizerProps> = ({ analyser }) => {
  const requestRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const visualizeData = useCallback(() => {
    requestRef.current = window.requestAnimationFrame(() => visualizeData());
    if (!canvasRef.current) {
      console.log("Canvas not found");
      return;
    }
    const audioData = new Uint8Array(140);
    analyser?.getByteFrequencyData(audioData);
    const bar_width = 3;
    let start = 0;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) {
      console.log("Canvas context not found");
      return;
    }
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    for (let i = 0; i < audioData.length; i++) {
      start = i * 4;
      let gradient = ctx.createLinearGradient(
        0,
        0,
        canvasRef.current.width,
        canvasRef.current.height,
      );
      gradient.addColorStop(0.2, "#2392f5");
      gradient.addColorStop(0.5, "#fe0095");
      gradient.addColorStop(1.0, "purple");
      ctx.fillStyle = gradient;
      ctx.fillRect(
        start,
        canvasRef.current.height,
        bar_width,
        (-audioData[i] * 100) / 255,
      );
    }
  }, [analyser]);

  const resetCanvas = useCallback(() => {
    if (!canvasRef.current) {
      return;
    }
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  }, []);

  useEffect(() => {
    if (!analyser) {
      return;
    }
    visualizeData();
    return () => {
      if (requestRef.current) {
        console.log("Canceling animation frame");
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [visualizeData, analyser, resetCanvas]);

  return <canvas ref={canvasRef} width={250} height={100} />;
};
