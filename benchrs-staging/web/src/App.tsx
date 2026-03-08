import { useQuery } from "@tanstack/react-query";
import { useAtomValue } from "jotai";
import { fetchPrerecordedResults } from "./lib/api";
import { liveResultsAtom } from "./lib/store";
import Header from "./components/Header";
import ResultsTable from "./components/ResultsTable";
import RunPanel from "./components/RunPanel";
import Methodology from "./components/Methodology";

export default function App() {
  const { data: prerecorded, isLoading } = useQuery({
    queryKey: ["results"],
    queryFn: fetchPrerecordedResults,
  });

  const liveResults = useAtomValue(liveResultsAtom);
  const results = liveResults.length > 0 ? liveResults : prerecorded;

  return (
    <div className="min-h-screen bg-white">
      <Header />
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <section className="mb-10">
          <h2 className="mb-2 text-xl font-semibold text-gray-900">
            Embedding Model Comparison
          </h2>
          <p className="mb-6 text-sm text-gray-500">
            8 dimensions × 8 models — 184 unique texts across English, Chinese,
            and Japanese.{" "}
            {liveResults.length > 0 ? (
              <span className="font-medium text-emerald-600">
                Showing live results
              </span>
            ) : (
              <span className="text-gray-400">
                Showing pre-recorded results (2026-03-08)
              </span>
            )}
          </p>
          {isLoading ? (
            <div className="py-20 text-center text-gray-400">
              Loading pre-recorded data...
            </div>
          ) : results ? (
            <ResultsTable results={results} />
          ) : null}
        </section>

        <section className="mb-10">
          <RunPanel />
        </section>

        <section>
          <Methodology />
        </section>
      </main>

      <footer className="border-t border-gray-100 py-6 text-center text-xs text-gray-400">
        benchrs — part of{" "}
        <a
          href="https://github.com/goliajp/airs"
          className="text-gray-500 hover:text-gray-700"
        >
          airs
        </a>{" "}
        (AI in Rust Series)
      </footer>
    </div>
  );
}
