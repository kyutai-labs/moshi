import { WSMessage } from "./types";

export const handshakeMessage: WSMessage = {
  type: "handshake",
  version: 0,
  model: 0,
};

export const audioMessage: WSMessage = {
  type: "audio",
  data: new Uint8Array(10),
};

export const textMessage: WSMessage = {
  type: "text",
  data: "Hello",
};

export const controlBOSMessage: WSMessage = {
  type: "control",
  action: "start",
};

export const controlEOSMessage: WSMessage = {
  type: "control",
  action: "endTurn",
};

export const metadataMessage: WSMessage = {
  type: "metadata",
  data: { key: "value" },
};
