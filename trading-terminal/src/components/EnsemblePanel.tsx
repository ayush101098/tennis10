"use client";

import type { EnsembleData, ModelContributionData } from "@/lib/types";
import clsx from "clsx";

const MODEL_LABELS: Record<string, string> = {
  markov: "Markov Chain",
  lr_sackmann: "LR Sackmann (14f)",
  lr_advanced: "LR Advanced (6f)",
  rf_advanced: "RF Advanced (6f)",
};

const MODEL_COLORS: Record<string, string> = {
  markov: "bg-terminal-cyan",
  lr_sackmann: "bg-terminal-green",
  lr_advanced: "bg-terminal-yellow",
  rf_advanced: "bg-purple-500",
};

function ContribBar({ model }: { model: ModelContributionData }) {
  const label = MODEL_LABELS[model.name] || model.name;
  const barColor = MODEL_COLORS[model.name] || "bg-slate-500";

  return (
    <div className="flex items-center gap-2 text-xs">
      <div className="w-[90px] truncate text-terminal-muted text-[10px]" title={label}>
        {label}
      </div>
      <div className="flex-1 h-3 bg-terminal-border/40 rounded overflow-hidden relative">
        <div
          className={clsx("h-full rounded transition-all duration-500", barColor)}
          style={{ width: `${model.probability * 100}%`, opacity: model.available ? 1 : 0.2 }}
        />
      </div>
      <div
        className={clsx(
          "w-[42px] text-right font-mono text-[11px]",
          model.available ? "text-slate-200" : "text-terminal-muted"
        )}
      >
        {model.available ? `${(model.probability * 100).toFixed(1)}%` : "—"}
      </div>
      <div className="w-[32px] text-right text-[9px] text-terminal-muted">
        w={model.available ? (model.weight * 100).toFixed(0) : "0"}%
      </div>
    </div>
  );
}

export default function EnsemblePanel({ ensemble }: { ensemble: EnsembleData | null }) {
  if (!ensemble) {
    return (
      <div className="h-full flex items-center justify-center text-terminal-muted text-[10px]">
        Ensemble data unavailable
      </div>
    );
  }

  const totalAvailable = ensemble.models.filter((m) => m.available).length;

  return (
    <div className="h-full overflow-y-auto px-2 py-1.5">
      <div className="flex justify-between items-center mb-2">
        <div className="text-[10px] font-bold text-terminal-muted uppercase tracking-wider">
          ML ENSEMBLE ({totalAvailable}/{ensemble.models.length} active)
        </div>
        <div className="text-[10px] text-terminal-cyan">
          blend: {(ensemble.blend_weight * 100).toFixed(0)}%
        </div>
      </div>

      {/* Blend ramp indicator */}
      <div className="mb-2">
        <div className="text-[9px] text-terminal-muted mb-0.5">Data confidence ramp</div>
        <div className="h-1.5 bg-terminal-border rounded-full overflow-hidden">
          <div
            className="h-full bg-terminal-cyan rounded-full transition-all duration-500"
            style={{ width: `${ensemble.blend_weight * 100}%` }}
          />
        </div>
        <div className="flex justify-between text-[8px] text-terminal-muted mt-0.5">
          <span>Career</span>
          <span>Live data</span>
        </div>
      </div>

      {/* Model contributions */}
      <div className="space-y-1.5">
        {ensemble.models.map((model) => (
          <ContribBar key={model.name} model={model} />
        ))}
      </div>

      {/* Blended stats summary */}
      {ensemble.blended_stats?.player1 && (
        <div className="mt-3 pt-2 border-t border-terminal-border">
          <div className="text-[9px] text-terminal-muted uppercase mb-1">Blended Input Stats</div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px]">
            <div className="text-terminal-muted">P1 SPW</div>
            <div>{((ensemble.blended_stats.player1.serve_pct ?? 0) * 100).toFixed(1)}%</div>
            <div className="text-terminal-muted">P2 SPW</div>
            <div>{((ensemble.blended_stats.player2?.serve_pct ?? 0) * 100).toFixed(1)}%</div>
            <div className="text-terminal-muted">P1 BP Save</div>
            <div>{((ensemble.blended_stats.player1.bp_save_pct ?? 0) * 100).toFixed(1)}%</div>
            <div className="text-terminal-muted">P2 BP Save</div>
            <div>{((ensemble.blended_stats.player2?.bp_save_pct ?? 0) * 100).toFixed(1)}%</div>
          </div>
        </div>
      )}
    </div>
  );
}
