import moshiProcessorUrl from "../../audio-processor.ts?worker&url";
import { FC, useEffect, useState, useCallback, useRef, MutableRefObject } from "react";
import eruda from "eruda";
import { Conversation } from "../Conversation/Conversation";
import { Button } from "../../components/Button/Button";
import { env } from "../../env";

export const Queue: FC = () => {
  const audioContext = useRef<AudioContext | null>(null);
  const worklet = useRef<AudioWorkletNode | null>(null);
  const [shouldConnect, setShouldConnect] = useState<boolean>(false);
  // enable eruda in development
  useEffect(() => {
    if (env.VITE_ENV === "development") {
      eruda.init();
    }
    () => {
      if (env.VITE_ENV === "development") {
        eruda.destroy();
      }
    };
  }, []);

  const startProcessor = useCallback(async () => {
    if (!audioContext.current) {
      audioContext.current = new AudioContext();
    }
    if (worklet.current) {
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

  const onConnect = useCallback(async () => {
    await startProcessor();
    setShouldConnect(true)
  }, [startProcessor, setShouldConnect]);

  if (shouldConnect && audioContext.current && worklet.current) {
    return (
      <Conversation
        audioContext={audioContext as MutableRefObject<AudioContext>}
        worklet={worklet as MutableRefObject<AudioWorkletNode>}
      />
    );
  }

  return (
    <div className="text-white text-center h-screen w-screen p-4 flex flex-col items-center ">
      <div>
        <h1 className="text-4xl">Hibiki Live Feed</h1>
        <div className="pt-8 text-sm flex justify-center items-center flex-col ">
          <div className="presentation text-left">
            <p><span className='cute-words'><a href="https://github.com/kyutai-labs/hibiki">Hibiki</a></span> is an experimental speech-to-speech translation AI. </p>
            <p>Click on `Join` below to join the <span className='cute-words'>live</span> translated feed.</p>
             <p>Baked with &lt;3 @<a href="https://kyutai.org/" className='cute-words underline'>Kyutai</a>.</p>
          </div>
        </div>
      </div>
      <div className="flex flex-grow justify-center items-center flex-col presentation">
          <Button onClick={async () => await onConnect()}>Join</Button>
      </div>
    </div >
  )
};
