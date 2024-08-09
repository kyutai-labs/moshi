import {
  CONTROL_MESSAGE,
  CONTROL_MESSAGES_MAP,
  MODELS_MAP,
  WSMessage,
  VERSIONS_MAP,
} from "./types";

export const encodeMessage = (message: WSMessage): Uint8Array => {
  switch (message.type) {
    case "handshake":
      return new Uint8Array([
        0x00,
        VERSIONS_MAP[message.version],
        MODELS_MAP[message.model],
      ]);
    case "audio":
      return new Uint8Array([0x01, ...message.data]);
    case "text":
      return new Uint8Array([0x02, ...new TextEncoder().encode(message.data)]);
    case "control":
      return new Uint8Array([0x03, CONTROL_MESSAGES_MAP[message.action]]);
    case "metadata":
      return new Uint8Array([
        0x04,
        ...new TextEncoder().encode(JSON.stringify(message.data)),
      ]);
    case "error":
      return new Uint8Array([0x05, ...new TextEncoder().encode(message.data)]);
    case "ping":
      return new Uint8Array([0x06]);
  }
};

export const decodeMessage = (data: Uint8Array): WSMessage => {
  const type = data[0];
  const payload = data.slice(1);
  switch (type) {
    case 0x00: {
      return {
        type: "handshake",
        version: 0,
        model: 0,
      };
    }
    case 0x01:
      return {
        type: "audio",
        data: payload,
      };
    case 0x02:
      return {
        type: "text",
        data: new TextDecoder().decode(payload),
      };
    case 0x03: {
      const action = Object.keys(CONTROL_MESSAGES_MAP).find(
        key => CONTROL_MESSAGES_MAP[key as CONTROL_MESSAGE] === payload[0],
      ) as CONTROL_MESSAGE | undefined;

      //TODO: log this and don't throw
      if (!action) {
        throw new Error("Unknown control message");
      }
      return {
        type: "control",
        action,
      };
    }
    case 0x04:
      return {
        type: "metadata",
        data: JSON.parse(new TextDecoder().decode(payload)),
      }
    case 0x05:
      return {
        type: "error",
        data: new TextDecoder().decode(payload),
      }
    case 0x06:
      return {
        type: "ping",
      }
    default: {
      console.log(type);
      throw new Error("Unknown message type");
    }
  }
};
