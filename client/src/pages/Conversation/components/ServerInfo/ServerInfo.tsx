import { useServerInfo } from "../../hooks/useServerInfo";

export const ServerInfo = () => {
  const { serverInfo } = useServerInfo();
  if (!serverInfo) {
    return null;
  }
  return (
    <div className="p-2 pt-4 self-center flex flex-col text-white border-2 border-white break-words">
      Our server is running on the following configuration:
        <div>Text temperature: {serverInfo.text_temperature}</div>
        <div>Text topk: {serverInfo.text_topk}</div>
        <div>Audio temperature: {serverInfo.audio_temperature}</div>
        <div>Audio topk: {serverInfo.audio_topk}</div>
        <div>Pad mult: {serverInfo.pad_mult}</div>
        <div>Repeat penalty last N: {serverInfo.repetition_penalty_context}</div>
        <div>Repeat penalty: {serverInfo.repetition_penalty}</div>
        <div>LM model file: {serverInfo.lm_model_file}</div>
        <div>Instance name: {serverInfo.instance_name}</div>
    </div>
  );
};
