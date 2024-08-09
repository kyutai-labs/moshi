import { FC, useMemo, useCallback, useState } from "react";
import { env } from "../../../../env";
import { getAPIClient } from "../../../Queue/api/client";
import { ThumbDown } from "./components/ThumbDown/ThumbDown";
import { ThumbUp } from "./components/ThumbUp/ThumbUp";
import { useMediaContext } from "../../MediaContext";

type FeedbackProps = {
  workerAuthId?: string;
  sessionId?: number;
  sessionAuthId?: string;
  email?: string;
};

export const Feedback:FC<FeedbackProps> =  ({
  workerAuthId,
  sessionId,
  sessionAuthId,
  email
}) => {
  const [feedback, setFeedback] = useState<0|1|null>(null);
  const {
    actualAudioPlayed,
  } = useMediaContext();

  const client = useMemo(() => {
    return getAPIClient(env.VITE_QUEUE_API_PATH)
  }, [env.VITE_QUEUE_API_PATH]);

  const sendFeedBack = useCallback((value: 0|1) => {
    setFeedback(value);
    if(!sessionId) {
      console.error("No session id given for feedback");
      return;
    }
    if(!workerAuthId) {
      console.error("No worker auth id given for feedback");
      return;
    }
    if(!sessionAuthId) {
      console.error("No session auth id given for feedback");
      return;
    }
    if(!email){
      console.error("No email given for feedback");
      return;
    }
    client.addFeedback({
      workerAuthId,
      sessionId,
      sessionAuthId,
      feedback:value,
      timestamp:actualAudioPlayed.current,
      email
    });
  }, [setFeedback,client, sessionId, sessionAuthId]);

  return (
    <div className="flex justify-center gap-2">
      <ThumbUp className="cursor-pointer" isSelected={feedback===1} onClick={() => sendFeedBack(1)} />
      <ThumbDown className="cursor-pointer" isSelected={feedback===0} onClick={() => sendFeedBack(0)} />
    </div>
  )
};
