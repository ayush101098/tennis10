/**
 * In-browser inference for the trained pre-match neural network.
 *
 * Weights are produced by train_nn_prematch.py (repo root) and served from
 * /nn_model.json. The network is an MLP over symmetric rank/surface features,
 * Platt-calibrated; out-of-sample (2024) it beats the Elo formula on log loss
 * and Brier score, which is what matters for Kelly staking.
 *
 * True P (pre-match) = 0.6 × NN + 0.4 × Elo when both ranks are known.
 * The NN averages both player orientations so P(p1) + P(p2) === 1 exactly.
 */

interface NNLayer { W: number[][]; b: number[] }

interface NNModelFile {
  feature_names: string[];
  scaler_mean: number[];
  scaler_std: number[];
  layers: NNLayer[];
  platt: { A: number; B: number };
  meta?: { trained_on?: string; oos_2024?: { acc: number; logloss: number; brier: number } };
}

let _modelPromise: Promise<NNModelFile | null> | null = null;

export function loadNNModel(): Promise<NNModelFile | null> {
  if (_modelPromise) return _modelPromise;
  _modelPromise = (async () => {
    try {
      const res = await fetch("/nn_model.json");
      if (!res.ok) return null;
      const model: NNModelFile = await res.json();
      console.log(`[nn] model loaded — ${model.meta?.trained_on ?? "unknown"} (oos brier ${model.meta?.oos_2024?.brier})`);
      return model;
    } catch (e) {
      console.warn("[nn] failed to load /nn_model.json", e);
      return null;
    }
  })();
  return _modelPromise;
}

/** Everything the model needs about one match, from A's perspective. */
export interface PrematchInputs {
  rankA: number;         // >0 required
  rankB: number;
  pointsA?: number;      // ranking points (0 if unknown)
  pointsB?: number;
  surface: string;       // "Hard" | "Clay" | "Grass"
  bestOf: number;        // 3 | 5
  bigEvent?: boolean;    // Slam / Masters / Finals
}

function features(inp: PrematchInputs): number[] {
  const lr = Math.log(inp.rankB) - Math.log(inp.rankA);
  const lp = Math.log1p(inp.pointsA ?? 0) - Math.log1p(inp.pointsB ?? 0);
  const clay = inp.surface === "Clay" ? 1 : 0;
  const grass = inp.surface === "Grass" ? 1 : 0;
  const hard = inp.surface === "Hard" ? 1 : 0;
  const bo5 = inp.bestOf === 5 ? 1 : 0;
  const big = inp.bigEvent ? 1 : 0;
  // age/height diffs unknown at schedule time → 0 (the training-set fill value)
  return [lr, lp, 0, 0, clay, grass, hard, bo5, big, lr * bo5, lr * clay, lr * grass];
}

function forward(model: NNModelFile, x: number[]): number {
  let a = x.map((v, i) => (v - model.scaler_mean[i]) / model.scaler_std[i]);
  for (let li = 0; li < model.layers.length; li++) {
    const { W, b } = model.layers[li];
    const out = new Array(b.length).fill(0);
    for (let j = 0; j < b.length; j++) {
      let s = b[j];
      for (let i = 0; i < a.length; i++) s += a[i] * W[i][j];
      out[j] = li < model.layers.length - 1 ? Math.max(0, s) : s; // ReLU hidden, linear out
    }
    a = out;
  }
  const raw = 1 / (1 + Math.exp(-a[0]));
  // Platt calibration on the output logit
  const z = Math.log(Math.min(1 - 1e-6, Math.max(1e-6, raw)) / Math.max(1e-6, 1 - raw));
  return 1 / (1 + Math.exp(-(model.platt.A * z + model.platt.B)));
}

/** P(A beats B) from the NN — averaged over both orientations for symmetry. */
export function nnMatchProb(model: NNModelFile, inp: PrematchInputs): number {
  const pAB = forward(model, features(inp));
  const pBA = forward(model, features({
    ...inp,
    rankA: inp.rankB, rankB: inp.rankA,
    pointsA: inp.pointsB, pointsB: inp.pointsA,
  }));
  const p = (pAB + (1 - pBA)) / 2;
  return Math.min(0.97, Math.max(0.03, p));
}
