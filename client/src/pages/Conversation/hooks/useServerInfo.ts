import { useCallback, useEffect, useState } from "react";
import { useSocketContext } from "../SocketContext";
import { decodeMessage } from "../../../protocol/encoder";
import { z } from "zod";

const ServersInfoSchema = z.object({
  text_temperature: z.number(),
  text_topk: z.number(),
  audio_temperature: z.number(),
  audio_topk: z.number(),
  pad_mult: z.number(),
  repetition_penalty_context: z.number(),
  repetition_penalty: z.number(),
  lm_model_file: z.string(),
  instance_name: z.string(),
  build_info: z.object({
    build_timestamp: z.string(),
    build_date: z.string(),
    git_branch: z.string(),
    git_timestamp: z.string(),
    git_date: z.string(),
    git_hash: z.string(),
    git_describe: z.string(),
    rustc_host_triple: z.string(),
    rustc_version: z.string(),
    cargo_target_triple: z.string(),
  }),
});

const parseInfo = (infos: any) => {
  const serverInfo =  ServersInfoSchema.safeParse(infos);
  if (!serverInfo.success) {
    console.error(serverInfo.error);
    return null;
  }
  return serverInfo.data;
};

type ServerInfo = {
  text_temperature: number;
  text_topk: number;
  audio_temperature: number;
  audio_topk: number;
  pad_mult: number;
  repetition_penalty_context: number;
  repetition_penalty: number;
  lm_model_file: string;
  instance_name: string;
  build_info: {
      build_timestamp: string;
      build_date: string;
      git_branch: string;
      git_timestamp: string;
      git_date: string;
      git_hash: string;
      git_describe: string;
      rustc_host_triple: string;
      rustc_version: string;
      cargo_target_triple: string;
  };
}

export const useServerInfo = () => {
  const [serverInfo, setServerInfo] = useState<ServerInfo|null>(null);
  const { socket } = useSocketContext();

  const onSocketMessage = useCallback((e: MessageEvent) => {
    const dataArray = new Uint8Array(e.data);
    const message = decodeMessage(dataArray);
    if (message.type === "metadata") {
      const infos = parseInfo(message.data);
      if (infos) {
        setServerInfo(infos);
        console.log("received metadata", infos);
      }
    }
  }, [setServerInfo]);

  useEffect(() => {
    const currentSocket = socket;
    if (!currentSocket) {
      return;
    }
    setServerInfo(null);
    currentSocket.addEventListener("message", onSocketMessage);
    return () => {
      currentSocket.removeEventListener("message", onSocketMessage);
    };
  }, [socket]);

  return { serverInfo };
};
