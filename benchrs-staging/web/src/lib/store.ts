import { atom } from "jotai";
import type { ModelResult } from "./types";

export const apiKeysAtom = atom({
  openai_key: "",
  gemini_key: "",
  cohere_key: "",
  voyage_key: "",
});

export const liveResultsAtom = atom<ModelResult[]>([]);
export const progressAtom = atom<string[]>([]);
export const isRunningAtom = atom(false);
