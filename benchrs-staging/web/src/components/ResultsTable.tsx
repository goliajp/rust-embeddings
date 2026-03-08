import {
  Trophy,
  Cpu,
  Cloud,
  ArrowUpDown,
  Search,
  Languages,
  Globe,
  Shield,
  Layers,
} from "lucide-react";
import type { ModelResult } from "../lib/types";

const METRICS: {
  key: keyof ModelResult;
  label: string;
  icon: typeof Trophy;
  format: (v: number, r: ModelResult) => string;
  higherBetter: boolean;
  group: string;
}[] = [
  {
    key: "dimensions",
    label: "Dims",
    icon: Layers,
    format: (v) => String(v),
    higherBetter: false,
    group: "info",
  },
  {
    key: "similarity_rho",
    label: "Spearman ρ",
    icon: ArrowUpDown,
    format: (v) => v.toFixed(2),
    higherBetter: true,
    group: "similarity",
  },
  {
    key: "discrimination_gap",
    label: "Discrimination",
    icon: ArrowUpDown,
    format: (v) => v.toFixed(2),
    higherBetter: true,
    group: "similarity",
  },
  {
    key: "retrieval_top1",
    label: "Retrieval",
    icon: Search,
    format: (v, r) => `${v}/${r.retrieval_total}`,
    higherBetter: true,
    group: "retrieval",
  },
  {
    key: "mrr",
    label: "MRR",
    icon: Search,
    format: (v) => v.toFixed(2),
    higherBetter: true,
    group: "retrieval",
  },
  {
    key: "en_rho",
    label: "EN ρ",
    icon: Languages,
    format: (v) => v.toFixed(2),
    higherBetter: true,
    group: "multilingual",
  },
  {
    key: "zh_rho",
    label: "ZH ρ",
    icon: Languages,
    format: (v) => v.toFixed(2),
    higherBetter: true,
    group: "multilingual",
  },
  {
    key: "ja_rho",
    label: "JA ρ",
    icon: Languages,
    format: (v) => v.toFixed(2),
    higherBetter: true,
    group: "multilingual",
  },
  {
    key: "crosslingual_avg",
    label: "Cross-lang",
    icon: Globe,
    format: (v) => v.toFixed(2),
    higherBetter: true,
    group: "crosslingual",
  },
  {
    key: "robustness_avg",
    label: "Robustness",
    icon: Shield,
    format: (v) => v.toFixed(2),
    higherBetter: true,
    group: "robustness",
  },
  {
    key: "cluster_separation",
    label: "Cluster sep.",
    icon: Layers,
    format: (v) => `${v.toFixed(2)}x`,
    higherBetter: true,
    group: "clustering",
  },
];

function isLocal(name: string) {
  return [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "bge-small-en-v1.5",
    "gte-small",
  ].includes(name);
}

function bestValue(results: ModelResult[], key: keyof ModelResult, higher: boolean) {
  const values = results.map((r) => r[key] as number);
  return higher ? Math.max(...values) : Math.min(...values);
}

export default function ResultsTable({ results }: { results: ModelResult[] }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-gray-200">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 bg-gray-50">
            <th className="sticky left-0 z-10 bg-gray-50 px-4 py-2.5 text-left font-semibold text-gray-700">
              Model
            </th>
            {METRICS.map((m) => (
              <th
                key={m.key}
                className="px-3 py-2.5 text-right font-medium text-gray-500"
              >
                <div className="flex items-center justify-end gap-1">
                  <m.icon className="h-3 w-3" />
                  {m.label}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr
              key={r.name}
              className="border-b border-gray-100 transition-colors hover:bg-gray-50/50"
            >
              <td className="sticky left-0 z-10 bg-white px-4 py-2.5 font-medium text-gray-900 whitespace-nowrap">
                <div className="flex items-center gap-2">
                  {isLocal(r.name) ? (
                    <Cpu className="h-3.5 w-3.5 text-blue-500" />
                  ) : (
                    <Cloud className="h-3.5 w-3.5 text-violet-500" />
                  )}
                  {r.name}
                  {r.name === "all-MiniLM-L6-v2" && (
                    <span className="rounded bg-blue-50 px-1 py-0.5 text-[10px] font-medium text-blue-600">
                      local default
                    </span>
                  )}
                  {r.name === "text-embedding-3-small" && (
                    <span className="rounded bg-violet-50 px-1 py-0.5 text-[10px] font-medium text-violet-600">
                      cloud default
                    </span>
                  )}
                </div>
              </td>
              {METRICS.map((m) => {
                const val = r[m.key] as number;
                const best = bestValue(results, m.key, m.higherBetter);
                const isBest = m.group !== "info" && val === best;
                return (
                  <td
                    key={m.key}
                    className={`px-3 py-2.5 text-right tabular-nums whitespace-nowrap ${
                      isBest
                        ? "bg-emerald-50/70 font-semibold text-emerald-600"
                        : "text-gray-600"
                    }`}
                  >
                    <div className="flex items-center justify-end gap-1">
                      {isBest && <Trophy className="h-3 w-3 text-amber-400" />}
                      {m.format(val, r)}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
