import moshiProcessorUrl from "../../audio-processor.ts?worker&url";
import { FC, useEffect, useState, useCallback, useRef, MutableRefObject } from "react";
import eruda from "eruda";
import { useSearchParams } from "react-router-dom";
import { Conversation } from "../Conversation/Conversation";
import { useModelParams } from "../Conversation/hooks/useModelParams";
import { ModelParams } from "../Conversation/components/ModelParams/ModelParams";
import { env } from "../../env";
import { Button } from "@/components/ui/button";
import { Settings, Power } from "lucide-react";

export const Queue: FC = () => {
  const [searchParams] = useSearchParams();
  const overrideWorkerAddr = searchParams.get("worker_addr");
  const [hasMicrophoneAccess, setHasMicrophoneAccess] = useState<boolean>(false);
  const [showMicrophoneAccessMessage, setShowMicrophoneAccessMessage] = useState<boolean>(false);
  const [shouldConnect, setShouldConnect] = useState<boolean>(false);
  const modelParams = useModelParams();
  const modalRef = useRef<HTMLDialogElement>(null);

  const audioContext = useRef<AudioContext | null>(null);
  const worklet = useRef<AudioWorkletNode | null>(null);

  useEffect(() => {
    if(env.VITE_ENV === "development") {
      eruda.init();
    }
    return () => {
      if(env.VITE_ENV === "development") {
        eruda.destroy();
      }
    };
  }, []);

  const getMicrophoneAccess = useCallback(async () => {
    try {
      await window.navigator.mediaDevices.getUserMedia({ audio: true });
      setHasMicrophoneAccess(true);
      return true;
    } catch(e) {
      console.error(e);
      setShowMicrophoneAccessMessage(true);
      setHasMicrophoneAccess(false);
    }
    return false;
  }, [setHasMicrophoneAccess, setShowMicrophoneAccessMessage]);

  const startProcessor = useCallback(async () => {
    if(!audioContext.current) {
      audioContext.current = new AudioContext();
    }
    if(worklet.current) {
      return;
    }
    let ctx = audioContext.current;
    ctx.resume();
    try {
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    } catch (err) {
      await ctx.audioWorklet.addModule(moshiProcessorUrl);
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    }
    worklet.current.connect(ctx.destination);
  }, [audioContext, worklet]);

  const onConnect = useCallback(async() => {
    await startProcessor();
    const hasAccess = await getMicrophoneAccess();
    if(hasAccess) {
      setShouldConnect(true);
    }
  }, [setShouldConnect, startProcessor, getMicrophoneAccess]);

  if(hasMicrophoneAccess && audioContext.current && worklet.current) {
    return (
      <Conversation
        workerAddr={overrideWorkerAddr ?? ""}
        audioContext={audioContext as MutableRefObject<AudioContext>}
        worklet={worklet as MutableRefObject<AudioWorkletNode>}
        {...modelParams}
      />
    );
  }

  return (
    <div className="bg-black text-white min-h-screen flex flex-col items-center justify-between p-8 relative">
      <Button
        variant="ghost"
        size="icon"
        className="absolute top-4 right-4 text-white hover:text-blue-400 transition-colors duration-200"
        onClick={() => modalRef.current?.showModal()}
      >
        <Settings className="h-6 w-6" />
      </Button>

      <div className="w-full max-w-md flex flex-col items-center justify-center flex-grow">
        <h1 className="text-5xl font-bold mb-8 text-center">
          <span className="text-white">Model</span>
          <span className="text-blue-500">Slab</span>
        </h1>
        
        <div className="space-y-6 w-full">
          {showMicrophoneAccessMessage && (
            <p className="text-center text-yellow-400">Please enable your microphone before proceeding</p>
          )}
          
          <Button
            onClick={onConnect}
            className="w-full bg-white text-black hover:bg-gray-200 font-bold py-3 px-4 rounded-md transition duration-200 flex items-center justify-center space-x-2"
          >
            <Power className="h-5 w-5" />
            <span>Connect</span>
          </Button>
        </div>
      </div>

      <dialog ref={modalRef} className="bg-black border border-blue-500 rounded-lg p-6 w-full max-w-md">
        <ModelParams {...modelParams} isConnected={shouldConnect} modal={modalRef} />
        <form method="dialog" className="mt-4">
          <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition duration-200">
            Close
          </Button>
        </form>
      </dialog>

      <div className="text-center space-y-2 text-xs text-white">
        <a href="https://kyutai.org/moshi-terms.pdf" target="_blank" rel="noopener noreferrer" className="hover:text-blue-300 transition-colors duration-200">
          Terms of Use
        </a>
        <span className="mx-2">|</span>
        <a href="https://kyutai.org/moshi-privacy.pdf" target="_blank" rel="noopener noreferrer" className="hover:text-blue-300 transition-colors duration-200">
          Privacy Policy
        </a>
      </div>
    </div>
  );
};
