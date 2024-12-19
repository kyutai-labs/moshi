import { FC, RefObject, useCallback, useEffect, useRef, useState } from "react";
import { clamp } from "../../hooks/audioUtils";
import { useSocketContext } from "../../SocketContext";

type AudioVisualizerProps = {
  analyser: AnalyserNode | null;
  parent: RefObject<HTMLElement>;
  imageUrl: string | undefined;
  copyCanvasRef?: RefObject<HTMLCanvasElement>;
};

const MAX_INTENSITY = 255;

export const ServerVisualizer: FC<AudioVisualizerProps> = ({ analyser, parent, imageUrl, copyCanvasRef }) => {
  const [canvasWidth, setCanvasWidth] = useState(parent.current ? Math.min(parent.current.clientWidth, parent.current.clientHeight) : 0);
  const requestRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const { isConnected } = useSocketContext();


  const draw = useCallback((width: number, centerX: number, centerY: number, audioData: Uint8Array, imageUrl: string | undefined, ctx: CanvasRenderingContext2D) => {
    const maxCircleWidth = Math.floor(width * 0.95);
    const averageIntensity = Math.sqrt(
      audioData.reduce((acc, curr) => acc + curr * curr, 0) / audioData.length,
    );
    const intensity = clamp(
      averageIntensity * 1.4,
      averageIntensity,
      MAX_INTENSITY,
    );
    const relIntensity = intensity / MAX_INTENSITY;
    const radius = ((isConnected ? 0.3 + 0.7 * relIntensity : relIntensity) * maxCircleWidth) / 2;
    // Draw a circle with radius based on intensity
    if (imageUrl == undefined) {
      ctx.clearRect(centerX - width / 2, centerY - width / 2, width, width);
      ctx.fillStyle = 'rgba(0, 0, 0, 0)';
      ctx.fillRect(centerX - width / 2, centerY - width / 2, width, width);
    } else {
      const img = new Image()
      img.src = imageUrl;
      img.onload = function () {
        ctx.drawImage(img, centerX - width / 2, centerY - width / 2, width, width);
      };
      console.log(img.src);
    }
    ctx.beginPath();
    //ctx.fillStyle = "#39e3a7";
    ctx.fillStyle = 'rgba(57, 227, 167, 0.5)';
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.closePath();

    // Draw an inner circle if we are connected.
    if (isConnected) {
      ctx.beginPath();
      ctx.arc(centerX, centerY, maxCircleWidth / 6, 0, 2 * Math.PI);
      // ctx.fillStyle = "#BCFCE5";
      ctx.fillStyle = 'rgba(188, 252, 229, 0.5)';
      ctx.fill();
      ctx.closePath();
    }

    //Draw a circle with max radius
    ctx.beginPath();
    ctx.arc(centerX, centerY, maxCircleWidth / 2, 0, 2 * Math.PI);
    ctx.strokeStyle = "white";
    ctx.lineWidth = width / 50;
    ctx.stroke();
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.fill()
    ctx.closePath();
  }, [isConnected]);

  const visualizeData = useCallback(() => {
    const width = parent.current ? Math.min(parent.current.clientWidth, parent.current.clientHeight) : 0;
    if (width !== canvasWidth) {
      console.log("Setting canvas width");
      setCanvasWidth(width);
    }
    requestRef.current = window.requestAnimationFrame(() => visualizeData());
    if (!canvasRef.current) {
      console.log("Canvas not found");
      return;
    }
    const ctx = canvasRef.current.getContext("2d");
    const audioData = new Uint8Array(140);
    analyser?.getByteFrequencyData(audioData);
    if (!ctx) {
      console.log("Canvas context not found");
      return;
    }
    const centerX = width / 2;
    const centerY = width / 2;
    // Hack: For the image, we display it using CSS  background-image
    // in the main image, but we display it via canvas so that 
    // it is in the video export
    draw(width, centerX, centerY, audioData, undefined, ctx);
    if (copyCanvasRef?.current) {
      const copyCtx = copyCanvasRef.current.getContext("2d");
      if (copyCtx) {
        draw(150, 125, 125, audioData, imageUrl, copyCtx);
      }
    }
  }, [analyser, isConnected, canvasWidth, parent, copyCanvasRef]);


  useEffect(() => {
    if (!analyser) {
      return;
    }
    analyser.smoothingTimeConstant = 0.95;
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
      className="max-h-full max-w-full"
      ref={canvasRef}
      width={canvasWidth}
      height={canvasWidth}
    />
  );
};
