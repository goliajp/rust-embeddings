import { BookOpen, Hash } from "lucide-react";

const DIMENSIONS = [
  {
    num: 1,
    name: "Graded Similarity",
    metric: "Spearman ρ + discrimination gap",
    desc: "32 human-scored pairs (0-5) across EN/ZH/JA",
  },
  {
    num: 2,
    name: "Retrieval",
    metric: "Top-1 accuracy + MRR",
    desc: "9 queries × 4 candidates (EN/ZH/JA)",
  },
  {
    num: 3,
    name: "Multilingual",
    metric: "ρ per language",
    desc: "English (21), Chinese (11), Japanese (5) pairs",
  },
  {
    num: 4,
    name: "Cross-lingual",
    metric: "Cosine similarity",
    desc: "6 sentence groups × 3 languages (same meaning)",
  },
  {
    num: 5,
    name: "Length Sensitivity",
    metric: "ρ by text length",
    desc: "Short (<50), medium (50-200), long (200+) chars",
  },
  {
    num: 6,
    name: "Robustness",
    metric: "Cosine similarity",
    desc: "15 variants: typos, casing, word order changes",
  },
  {
    num: 7,
    name: "Clustering",
    metric: "NN purity + separation ratio",
    desc: "4 topics × 5 texts each",
  },
  {
    num: 8,
    name: "Throughput",
    metric: "texts/sec",
    desc: "500 identical-length texts (live only)",
  },
];

export default function Methodology() {
  return (
    <div id="methodology" className="scroll-mt-20">
      <div className="mb-4 flex items-center gap-2">
        <BookOpen className="h-5 w-5 text-gray-400" />
        <h2 className="text-xl font-semibold text-gray-900">Methodology</h2>
      </div>
      <p className="mb-4 text-sm text-gray-500">
        Each model is evaluated across 8 dimensions using 184 unique test texts.
        Pre-recorded data ships with the experiment so anyone can reproduce the
        analysis without API keys.
      </p>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {DIMENSIONS.map((d) => (
          <div
            key={d.num}
            className="rounded-lg border border-gray-200 p-3 transition-colors hover:border-gray-300"
          >
            <div className="mb-1 flex items-center gap-1.5">
              <Hash className="h-3 w-3 text-gray-400" />
              <span className="text-xs font-medium text-gray-400">
                {d.num}
              </span>
              <span className="text-sm font-medium text-gray-900">
                {d.name}
              </span>
            </div>
            <p className="mb-1 text-xs font-medium text-emerald-600">
              {d.metric}
            </p>
            <p className="text-xs text-gray-400">{d.desc}</p>
          </div>
        ))}
      </div>

      <div className="mt-6 rounded-lg border border-amber-100 bg-amber-50/50 p-4">
        <h3 className="mb-1 text-sm font-semibold text-amber-800">Decision</h3>
        <div className="space-y-2 text-sm text-amber-700">
          <p>
            <strong>local() → all-MiniLM-L6-v2</strong> — 23MB (only model
            small enough for app embedding), best clustering separation 8.73x,
            100% retrieval, EN ρ=0.92.
          </p>
          <p>
            <strong>cloud() → OpenAI text-embedding-3-small</strong> — best
            discrimination 0.58, 100% retrieval, balanced multilingual
            (EN/ZH/JA all &gt;0.88), cheapest at $0.02/1M tokens.
          </p>
        </div>
      </div>

      <div className="mt-4 text-xs text-gray-400">
        <p>
          Reproduce locally:{" "}
          <code className="rounded bg-gray-100 px-1 py-0.5 text-gray-500">
            cargo run -p benchrs --bin embedding_models --release
          </code>
        </p>
      </div>
    </div>
  );
}
