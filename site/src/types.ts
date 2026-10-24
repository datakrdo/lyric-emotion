export interface Top { emotion: string; score: number; distinct: boolean }
export interface Compared {
  emotion: string; higher: string; median_a: number; median_b: number;
  significant: boolean; p_adj: number; pair_share_a: number;
}
export interface Findings {
  bands: [string, string];
  verdict: string;
  window: { n_songs: number; n_a: number; n_b: number; n_studio_albums: number;
            start_year: number; end_year: number };
  profile: Record<string, { top: Top[]; sadness_rank: number }>;
  emotions_compared: Compared[];
  valence: { status: string; delta: number; ci_low: number; ci_high: number };
  style: { status: string; auc: number; p_value: number; auc_without_question_labels: number };
  emotions: { status: string };
  poles: Record<string, string[]>;
  question_line_share: Record<string, number>;
}
export interface Album {
  artist: string; album: string; year: number; n: number; kind: string;
  valence_pct: number; stands_out: string; cover_url: string;
}
export type Song = {
  artist: string; title: string; album: string; year: number;
  valence_z: number; valence_pct: number; style: number;
} & Record<string, number | string>;
export interface Bundle { findings: Findings; albums: Album[]; songs: Song[] }
export interface Validation { status: string; summary?: string }
