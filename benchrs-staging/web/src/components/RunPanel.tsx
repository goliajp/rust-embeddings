import { useAtom, useSetAtom } from "jotai";
import { Key, Play, Loader2, Terminal } from "lucide-react";
import { useCallback, useRef, useState } from "react";
import { runExperiment } from "../lib/api";
import {
  apiKeysAtom,
  isRunningAtom,
  liveResultsAtom,
  progressAtom,
} from "../lib/store";

const PROVIDERS = [
  { key: "openai_key" as const, label: "OpenAI", placeholder: "sk-..." },
  { key: "gemini_key" as const, label: "Gemini", placeholder: "AI..." },
  { key: "cohere_key" as const, label: "Cohere", placeholder: "..." },
  { key: "voyage_key" as const, label: "Voyage", placeholder: "pa-..." },
];

export default function RunPanel() {
  const [keys, setKeys] = useAtom(apiKeysAtom);
  const [isRunning, setIsRunning] = useAtom(isRunningAtom);
  const setLiveResults = useSetAtom(liveResultsAtom);
  const [progress, setProgress] = useAtom(progressAtom);
  const [expanded, setExpanded] = useState(false);
  const cancelRef = useRef<(() => void) | null>(null);

  const handleRun = useCallback(() => {
    setIsRunning(true);
    setProgress([]);
    setLiveResults([]);

    const req = Object.fromEntries(
      Object.entries(keys).filter(([, v]) => v.length > 0),
    );

    cancelRef.current = runExperiment(
      req,
      (msg) => setProgress((prev) => [...prev, msg]),
      (results) => setLiveResults(results),
      () => setIsRunning(false),
      (err) => {
        setProgress((prev) => [...prev, `error: ${err}`]);
        setIsRunning(false);
      },
    );
  }, [keys, setIsRunning, setProgress, setLiveResults]);

  const handleCancel = useCallback(() => {
    cancelRef.current?.();
    setIsRunning(false);
    setProgress((prev) => [...prev, "cancelled"]);
  }, [setIsRunning, setProgress]);

  const hasAnyKey = Object.values(keys).some((v) => v.length > 0);

  return (
    <div className="rounded-lg border border-gray-200 bg-gray-50/50">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center justify-between px-4 py-3 text-left"
      >
        <div className="flex items-center gap-2">
          <Play className="h-4 w-4 text-emerald-500" />
          <span className="font-semibold text-gray-900">
            Run Live Experiment
          </span>
          <span className="text-xs text-gray-400">
            enter API keys to run with your own accounts
          </span>
        </div>
        <span className="text-gray-400">{expanded ? "−" : "+"}</span>
      </button>

      {expanded && (
        <div className="border-t border-gray-200 px-4 py-4">
          <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
            {PROVIDERS.map((p) => (
              <div key={p.key}>
                <label className="mb-1 block text-xs font-medium text-gray-500">
                  <div className="flex items-center gap-1">
                    <Key className="h-3 w-3" />
                    {p.label}
                  </div>
                </label>
                <input
                  type="password"
                  placeholder={p.placeholder}
                  value={keys[p.key]}
                  onChange={(e) =>
                    setKeys({ ...keys, [p.key]: e.target.value })
                  }
                  className="w-full rounded border border-gray-200 bg-white px-3 py-1.5 text-sm font-mono text-gray-700 placeholder-gray-300 focus:border-emerald-300 focus:ring-1 focus:ring-emerald-300 focus:outline-none"
                />
              </div>
            ))}
          </div>

          <p className="mb-4 text-xs text-gray-400">
            Local models (MiniLM, BGE, GTE) always run. Cloud models are
            skipped if no key is provided. Keys are only sent to the local
            server — never stored.
          </p>

          <div className="flex items-center gap-3">
            {isRunning ? (
              <button
                onClick={handleCancel}
                className="flex items-center gap-2 rounded bg-red-500 px-4 py-1.5 text-sm font-medium text-white hover:bg-red-600"
              >
                Cancel
              </button>
            ) : (
              <button
                onClick={handleRun}
                className="flex items-center gap-2 rounded bg-emerald-500 px-4 py-1.5 text-sm font-medium text-white hover:bg-emerald-600 disabled:opacity-50"
              >
                <Play className="h-3.5 w-3.5" />
                Run Experiment
                {!hasAnyKey && (
                  <span className="text-xs opacity-70">(local only)</span>
                )}
              </button>
            )}
            {isRunning && (
              <span className="flex items-center gap-1.5 text-xs text-gray-400">
                <Loader2 className="h-3 w-3 animate-spin" />
                running...
              </span>
            )}
          </div>

          {progress.length > 0 && (
            <div className="mt-4 max-h-40 overflow-y-auto rounded bg-gray-900 p-3 font-mono text-xs text-gray-300">
              <div className="mb-1 flex items-center gap-1 text-gray-500">
                <Terminal className="h-3 w-3" />
                progress
              </div>
              {progress.map((msg, i) => (
                <div key={i} className="py-0.5">
                  <span className="mr-2 text-gray-600">$</span>
                  {msg}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
