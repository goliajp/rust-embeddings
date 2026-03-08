export interface ModelResult {
  name: string;
  dimensions: number;
  similarity_rho: number;
  discrimination_gap: number;
  similar_avg: number;
  dissimilar_avg: number;
  retrieval_top1: number;
  retrieval_total: number;
  mrr: number;
  en_rho: number;
  zh_rho: number;
  ja_rho: number;
  crosslingual_avg: number;
  crosslingual_min: number;
  robustness_avg: number;
  robustness_min: number;
  cluster_purity: number;
  cluster_total: number;
  cluster_separation: number;
}

export interface RunRequest {
  openai_key?: string;
  gemini_key?: string;
  cohere_key?: string;
  voyage_key?: string;
}
