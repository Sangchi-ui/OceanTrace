"use client";

import React from 'react';

type ResultsPanelProps = {
  result: any;
};

export function ResultsPanel({ result }: ResultsPanelProps) {
  if (!result) {
    return (
      <div className="bg-gray-800 rounded-lg p-3 shadow-lg border border-gray-700 flex flex-col gap-2 opacity-50">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">Analysis Results</h3>
        <div className="text-sm text-gray-500 italic">Run analysis to see results</div>
      </div>
    );
  }

  const h = result.hindcast;
  const f = result.forecast;

  return (
    <div className="bg-gray-800 rounded-lg p-3 shadow-lg border border-gray-700 flex flex-col gap-2">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">Analysis Results</h3>
      
      <div className="space-y-3">
        {h?.enabled && h.best_candidate && (
          <div className="bg-gray-900 rounded p-2 border border-gray-700">
            <h4 className="text-[11px] font-bold text-orange-400 uppercase">Hindcast: Best Origin</h4>
            <div className="flex justify-between mt-1 text-sm">
              <span className="text-gray-400">Match Score:</span>
              <span className="text-gray-200 font-mono">{(h.best_candidate.composite_score * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Release:</span>
              <span className="text-gray-200 font-mono text-xs">{h.best_candidate.release_timestamp.replace('T', ' ').slice(0, 16)}</span>
            </div>
            {h.engine_used && (
              <div className="flex justify-between text-[10px] mt-1">
                <span className="text-gray-500">Engine:</span>
                <span className="text-gray-400">{h.engine_used}</span>
              </div>
            )}
          </div>
        )}

        {f?.enabled && f.horizons && f.horizons.length > 0 && (
          <div className="bg-gray-900 rounded p-2 border border-gray-700">
            <h4 className="text-[11px] font-bold text-teal-400 uppercase">Forecast: Max Horizon (+{f.horizons[f.horizons.length-1].horizon_hours}h)</h4>
            <div className="flex justify-between mt-1 text-sm">
              <span className="text-gray-400">Drift Dist:</span>
              <span className="text-gray-200 font-mono">{f.horizons[f.horizons.length-1].drift_distance_km.toFixed(1)} km</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Spread Area:</span>
              <span className="text-gray-200 font-mono">{f.horizons[f.horizons.length-1].spread_area_km2.toFixed(1)} km²</span>
            </div>
            {f.engine_used && (
              <div className="flex justify-between text-[10px] mt-1">
                <span className="text-gray-500">Engine:</span>
                <span className="text-gray-400">{f.engine_used}</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
