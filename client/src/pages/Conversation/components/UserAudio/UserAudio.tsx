import { FC, useCallback, useEffect, useRef, useState } from "react";
import { useSocketContext } from "../../SocketContext";
import { useUserAudio } from "../../hooks/useUserAudio";
import { ClientVisualizer } from "../AudioVisualizer/ClientVisualizer";

type UserAudioProps = {
  copyCanvasRef: React.RefObject<HTMLCanvasElement>;
};
export const UserAudio: FC<UserAudioProps> = ({ copyCanvasRef }) => {
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const { sendMessage, isConnected } = useSocketContext();
  const containerRef = useRef<HTMLDivElement>(null);
  const onRecordingStart = useCallback(() => {
    console.log("Recording started");
  }, []);

  const onRecordingStop = useCallback(() => {
    console.log("Recording stopped");
  }, []);

  const onRecordingChunk = useCallback(
    (chunk: Uint8Array) => {
      if (!isConnected) {
        return;
      }
      sendMessage({
        type: "audio",
        data: chunk,
      });
    },
    [sendMessage, isConnected],
  );

  const { startRecordingUser, stopRecording } = useUserAudio({
    constraints: {
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
      video: false,
    },
    onDataChunk: onRecordingChunk,
    onRecordingStart,
    onRecordingStop,
  });

  useEffect(() => {
    let res: Awaited<ReturnType<typeof startRecordingUser>>;
    if (isConnected) {
      startRecordingUser().then(result => {
        if (result) {
          res = result;
          setAnalyser(result.analyser);
        }
      });
    }
    return () => {
      console.log("Stop recording called from somewhere else.");
      stopRecording();
      res?.source?.disconnect();
    };
  }, [startRecordingUser, stopRecording, isConnected]);

  return (
    <div className="user-audio h-5/6 aspect-square" ref={containerRef}>
      <ClientVisualizer analyser={analyser} parent={containerRef} copyCanvasRef={copyCanvasRef} />
    </div>
  );
};
