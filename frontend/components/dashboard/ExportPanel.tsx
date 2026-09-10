"use client";

import React, { useState } from 'react';

type ExportPanelProps = {
  result: any;
};

export function ExportPanel({ result }: ExportPanelProps) {
  const [exporting, setExporting] = useState(false);

  const handleExportGeojson = async () => {
    if (!result) return;
    setExporting(true);
    try {
      const res = await fetch('http://localhost:8000/api/v2/export/geojson', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(result)
      });
      const data = await res.json();
      
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `oceantrace_export_${result.input_reference?.spill_id || 'spill'}.geojson`;
      a.click();
    } catch (e) {
      console.error(e);
    } finally {
      setExporting(false);
    }
  };

  const handleExportSitrep = async () => {
    if (!result) return;
    setExporting(true);
    try {
      const res = await fetch('http://localhost:8000/api/v2/export/sitrep', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(result)
      });
      const data = await res.json();
      
      const blob = new Blob([data.markdown], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `SITREP_${result.input_reference?.spill_id || 'spill'}.md`;
      a.click();
    } catch (e) {
      console.error(e);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-3 shadow-lg border border-gray-700 flex flex-col gap-2">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">Export Products</h3>
      <div className="flex flex-col gap-2">
        <button 
          onClick={handleExportGeojson}
          disabled={!result || exporting}
          className="bg-gray-700 hover:bg-gray-600 disabled:opacity-50 text-white py-1.5 px-3 rounded text-sm flex items-center justify-center"
        >
          📄 Download GeoJSON
        </button>
        <button 
          onClick={handleExportSitrep}
          disabled={!result || exporting}
          className="bg-indigo-700 hover:bg-indigo-600 disabled:opacity-50 text-white py-1.5 px-3 rounded text-sm flex items-center justify-center"
        >
          📋 Generate SITREP
        </button>
      </div>
    </div>
  );
}
