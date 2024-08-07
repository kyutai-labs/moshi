import { useCallback, useState } from "react";

export const DEFAULT_TEXT_TEMPERATURE = 0.7;
export const DEFAULT_TEXT_TOPK = 25;
export const DEFAULT_AUDIO_TEMPERATURE = 0.8;
export const DEFAULT_AUDIO_TOPK = 250;
export const DEFAULT_PAD_MULT = 0;
export const DEFAULT_REPETITION_PENALTY_CONTEXT = 64;
export const DEFAULT_REPETITION_PENALTY = 1.0;

export type ModelParamsValues = {
  textTemperature: number;
  textTopk: number;
  audioTemperature: number;
  audioTopk: number;
  padMult: number;
  repetitionPenaltyContext: number,
  repetitionPenalty: number,
};

type useModelParamsArgs = Partial<ModelParamsValues>;

export const useModelParams = (params?:useModelParamsArgs) => {

  const [textTemperature, setTextTemperatureBase] = useState(params?.textTemperature || DEFAULT_TEXT_TEMPERATURE);
  const [textTopk, setTextTopkBase]= useState(params?.textTopk || DEFAULT_TEXT_TOPK);
  const [audioTemperature, setAudioTemperatureBase] = useState(params?.audioTemperature || DEFAULT_AUDIO_TEMPERATURE);
  const [audioTopk, setAudioTopkBase] = useState(params?.audioTopk || DEFAULT_AUDIO_TOPK);
  const [padMult, setPadMultBase] = useState(params?.padMult || DEFAULT_PAD_MULT);
  const [repetitionPenalty, setRepetitionPenaltyBase] = useState(params?.repetitionPenalty || DEFAULT_REPETITION_PENALTY);
  const [repetitionPenaltyContext, setRepetitionPenaltyContextBase] = useState(params?.repetitionPenaltyContext || DEFAULT_REPETITION_PENALTY_CONTEXT);

  const resetParams = useCallback(() => {
    setTextTemperatureBase(DEFAULT_TEXT_TEMPERATURE);
    setTextTopkBase(DEFAULT_TEXT_TOPK);
    setAudioTemperatureBase(DEFAULT_AUDIO_TEMPERATURE);
    setAudioTopkBase(DEFAULT_AUDIO_TOPK);
    setPadMultBase(DEFAULT_PAD_MULT);
    setRepetitionPenalty(DEFAULT_REPETITION_PENALTY);
    setRepetitionPenaltyContext(DEFAULT_REPETITION_PENALTY_CONTEXT);
  }, [
    setTextTemperatureBase,
    setTextTopkBase,
    setAudioTemperatureBase,
    setAudioTopkBase,
    setPadMultBase,
    setRepetitionPenaltyBase,
    setRepetitionPenaltyContextBase,
  ]);

  const setTextTemperature = useCallback((value: number) => {
    if(value <= 1.2 || value >= 0.2) {
      setTextTemperatureBase(value);
    }
  }, []);
  const setTextTopk = useCallback((value: number) => {
    if(value <= 500 || value >= 10) {
      setTextTopkBase(value);
    }
  }, []);
  const setAudioTemperature = useCallback((value: number) => {
    if(value <= 1.2 || value >= 0.2) {
      setAudioTemperatureBase(value);
    }
  }, []);
  const setAudioTopk = useCallback((value: number) => {
    if(value <= 500 || value >= 10) {
      setAudioTopkBase(value);
    }
  }, []);
  const setPadMult = useCallback((value: number) => {
    if(value <= 4 || value >= -4) {
      setPadMultBase(value);
    }
  }, []);
  const setRepetitionPenalty = useCallback((value: number) => {
    if(value <= 2.0 || value >= 1.0) {
      setRepetitionPenaltyBase(value);
    }
  }, []);
  const setRepetitionPenaltyContext = useCallback((value: number) => {
    if(value <= 200|| value >= 0) {
      setRepetitionPenaltyContextBase(value);
    }
  }, []);

  return {
    textTemperature,
    textTopk,
    audioTemperature,
    audioTopk,
    padMult,
    repetitionPenalty,
    repetitionPenaltyContext,
    setTextTemperature,
    setTextTopk,
    setAudioTemperature,
    setAudioTopk,
    setPadMult,
    setRepetitionPenalty,
    setRepetitionPenaltyContext,
    resetParams,
  }
}
