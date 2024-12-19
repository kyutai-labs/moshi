import { FC, RefObject } from "react";
import { useModelParams } from "../../hooks/useModelParams";
import { Button } from "../../../../components/Button/Button";

type ModelParamsProps = {
  isConnected: boolean;
  isImageMode: boolean;
  modal?: RefObject<HTMLDialogElement>,
} & ReturnType<typeof useModelParams>;
export const ModelParams: FC<ModelParamsProps> = ({
  textTemperature,
  textTopk,
  audioTemperature,
  audioTopk,
  padMult,
  repetitionPenalty,
  repetitionPenaltyContext,
  imageResolution,
  setTextTemperature,
  setTextTopk,
  setAudioTemperature,
  setAudioTopk,
  setPadMult,
  setRepetitionPenalty,
  setRepetitionPenaltyContext,
  setImageResolution,
  resetParams,
  isConnected,
  isImageMode,
  modal,
}) => {
  return (
    <div className=" p-2 mt-6 self-center flex flex-col text-white items-center text-center">
      <table>
        <tbody>
          <tr>
            <td>Text temperature:</td>
            <td className="w-12 text-center">{textTemperature}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="text-temperature" name="text-temperature" step="0.01" min="0.2" max="1.2" value={textTemperature} onChange={e => setTextTemperature(parseFloat(e.target.value))} /></td>
          </tr>
          <tr>
            <td>Text topk:</td>
            <td className="w-12 text-center">{textTopk}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="text-topk" name="text-topk" step="1" min="10" max="500" value={textTopk} onChange={e => setTextTopk(parseInt(e.target.value))} /></td>
          </tr>
          <tr>
            <td>Audio temperature:</td>
            <td className="w-12 text-center">{audioTemperature}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="audio-temperature" name="audio-temperature" step="0.01" min="0.2" max="1.2" value={audioTemperature} onChange={e => setAudioTemperature(parseFloat(e.target.value))} /></td>
          </tr>
          <tr>
            <td>Audio topk:</td>
            <td className="w-12 text-center">{audioTopk}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="audio-topk" name="audio-topk" step="1" min="10" max="500" value={audioTopk} onChange={e => setAudioTopk(parseInt(e.target.value))} /></td>
          </tr>
          <tr>
            <td>Padding multiplier:</td>
            <td className="w-12 text-center">{padMult}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="audio-pad-mult" name="audio-pad-mult" step="0.05" min="-4" max="4" value={padMult} onChange={e => setPadMult(parseFloat(e.target.value))} /></td>
          </tr>
          <tr>
            <td>Repeat penalty:</td>
            <td className="w-12 text-center">{repetitionPenalty}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="repetition-penalty" name="repetition-penalty" step="0.01" min="1" max="2" value={repetitionPenalty} onChange={e => setRepetitionPenalty(parseFloat(e.target.value))} /></td>
          </tr>
          <tr>
            <td>Repeat penalty last N:</td>
            <td className="w-12 text-center">{repetitionPenaltyContext}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="repetition-penalty-context" name="repetition-penalty-context" step="1" min="0" max="200" value={repetitionPenaltyContext} onChange={e => setRepetitionPenaltyContext(parseFloat(e.target.value))} /></td>
          </tr>
          {isImageMode &&
            <tr>
              <td>Image max-side (px):</td>
              <td className="w-12 text-center">{imageResolution}</td>
              <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="image-resolution" name="image-resolution" step="16" min="64" max="512" value={imageResolution} onChange={e => setImageResolution(parseFloat(e.target.value))} /></td>
            </tr>
          }
        </tbody>
      </table>
      <div>
        {!isConnected && <Button onClick={resetParams} className="m-2">Reset</Button>}
        {!isConnected && <Button onClick={() => modal?.current?.close()} className="m-2">Validate</Button>}
      </div>
    </div >
  )
};
