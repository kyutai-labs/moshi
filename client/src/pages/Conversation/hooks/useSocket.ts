import { useState, useEffect, useCallback, useRef } from "react";
import { WSMessage } from "../../../protocol/types";
import { decodeMessage, encodeMessage } from "../../../protocol/encoder";

export const useSocket = ({
  onMessage,
  uri,
  onDisconnect: onDisconnectProp,
}: {
  onMessage?: (message: WSMessage) => void;
  uri: string;
  onDisconnect?: () => void;
}) => {
  const lastMessageTime = useRef<null|number>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [socket, setSocket] = useState<WebSocket | null>(null);

  const sendMessage = useCallback(
    (message: WSMessage) => {
      if (!socket || !isConnected) {
        console.log("socket not connected");
        return;
      }
      socket.send(encodeMessage(message));
    },
    [isConnected],
  );

  const onConnect = useCallback(() => {
    console.log("connected, now waiting for handshake.");
    // setIsConnected(true);
  }, [setIsConnected]);

  const onDisconnect = useCallback(() => {
    console.log("disconnected");
    if (onDisconnectProp) {
      onDisconnectProp();
    }
    setIsConnected(false);
  }, [onDisconnectProp]);

  const onMessageEvent = useCallback(
    (eventData: MessageEvent) => {
      lastMessageTime.current = Date.now();
      const dataArray = new Uint8Array(eventData.data);
      const message = decodeMessage(dataArray);
      if (message.type == "handshake") {
        console.log("Handshake received, let's rocknroll.");
        setIsConnected(true);
      }
      if (!onMessage) {
        return;
      }
      onMessage(message);
    },
    [onMessage, setIsConnected],
  );

  const start = useCallback(() => {
    const ws = new WebSocket(uri);
    ws.binaryType = "arraybuffer";
    ws.addEventListener("open", onConnect);
    ws.addEventListener("close", onDisconnect);
    ws.addEventListener("message", onMessageEvent);
    setSocket(ws);
    console.log("Socket created", ws);
    lastMessageTime.current = Date.now();
  }, [uri, onMessage, onDisconnectProp]);

  const stop = useCallback(() => {
      setIsConnected(false);
      if (onDisconnectProp) {
        onDisconnectProp();
      }
      socket?.close();
      setSocket(null);
  }, [socket]);

  useEffect(() => {
    if(!isConnected){
      return;
    }
    let intervalId = setInterval(() => {
      if (lastMessageTime.current && Date.now() - lastMessageTime.current > 10000) {
        console.log("closing socket due to inactivity", socket);
        socket?.close();
        onDisconnect();
        clearInterval(intervalId);
      }
    }, 500);

    return () => {
      lastMessageTime.current = null;
      clearInterval(intervalId);
    };
  }, [isConnected, socket]);

  return {
    isConnected,
    socket,
    sendMessage,
    start,
    stop,
  };
};
