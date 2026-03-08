import type { ModelResult, RunRequest } from "./types";

export async function fetchPrerecordedResults(): Promise<ModelResult[]> {
  const res = await fetch("/api/results");
  if (!res.ok) throw new Error(`failed to fetch results: ${res.status}`);
  return res.json();
}

export function runExperiment(
  keys: RunRequest,
  onProgress: (msg: string) => void,
  onResults: (results: ModelResult[]) => void,
  onDone: () => void,
  onError: (err: string) => void,
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(keys),
        signal: controller.signal,
      });

      if (!res.ok) {
        onError(`server error: ${res.status}`);
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        onError("no response body");
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        let eventType = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            eventType = line.slice(7);
          } else if (line.startsWith("data: ")) {
            const data = line.slice(6);
            if (eventType === "progress") {
              onProgress(data);
            } else if (eventType === "results") {
              onResults(JSON.parse(data));
            } else if (eventType === "done") {
              onDone();
            }
            eventType = "";
          }
        }
      }
    } catch (e) {
      if (!controller.signal.aborted) {
        onError(e instanceof Error ? e.message : "unknown error");
      }
    }
  })();

  return () => controller.abort();
}
