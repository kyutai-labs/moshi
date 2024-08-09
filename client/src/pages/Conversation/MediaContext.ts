import { MutableRefObject, createContext, useContext } from "react";
type MediaContextType = {
  startRecording: () => void;
  stopRecording: () => void;
  audioContext: MutableRefObject<AudioContext>;
  audioStreamDestination: MutableRefObject<MediaStreamAudioDestinationNode>;
  worklet: MutableRefObject<AudioWorkletNode>;
  micDuration: MutableRefObject<number>;
  actualAudioPlayed: MutableRefObject<number>;
};

export const MediaContext = createContext<MediaContextType | null>(null);

export const useMediaContext = () => {
  const context = useContext(MediaContext);
  if (!context) {
    throw new Error(
      "useMediaContext must be used within a MediaContextProvider",
    );
  }

  return context;
};