import { createContext, useContext } from "react";
import { WSMessage } from "../../protocol/types";

type SocketContextType = {
  isConnected: boolean;
  socket: WebSocket | null;
  sendMessage: (message: WSMessage) => void;
};

export const SocketContext = createContext<SocketContextType>({
  isConnected: false,
  socket: null,
  sendMessage: () => {},
});

export const useSocketContext = () => {
  return useContext(SocketContext);
};
