import ReactDOM from "react-dom/client";
import {
  createBrowserRouter,
  RouterProvider,
} from "react-router-dom";
import "./index.css";
// @ts-expect-error - Worker is not recognized by the TS compiler
import { DecoderWorker } from "./decoder/decoderWorker";
import { Queue } from "./pages/Queue/Queue";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Queue />,
  },
]);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
    <RouterProvider router={router}/>
);
