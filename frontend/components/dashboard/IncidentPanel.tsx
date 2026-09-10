"use client";

import React, { useState, useEffect } from 'react';

type IncidentPanelProps = {
  onRunAnalysis: (params: any) => void;
  isLoading: boolean;
};

export function IncidentPanel({ onRunAnalysis, isLoading }: IncidentPanelProps) {
  const [formData, setFormData] = useState({
    spill_id: 'SPILL-TEST-001',
    lon: '103.82',
    lat: '1.35',
    area_km2: '12.5',
    confidence: '0.85',
    oil_type: 'medium_crude',
    forcing_mode: 'mock',
    mode: 'both',
    source_satellite: 'Sentinel-1A',
    observation_timestamp: new Date().toISOString().slice(0, 16)
  });

  const loadFromDetection = () => {
    try {
      const stored = localStorage.getItem('last_agent1_result');
      if (stored) {
        const data = JSON.parse(stored);
        if (data && data.metrics) {
          // just mock data loading
          setFormData(prev => ({
            ...prev,
            area_km2: (data.metrics.oil_pixels * 0.0001).toFixed(2),
            confidence: data.metrics.mean_confidence.toFixed(2),
            observation_timestamp: data.metadata?.acquisition_time || prev.observation_timestamp
          }));
        }
      }
    } catch (e) {
      console.error("Could not load agent1 result", e);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onRunAnalysis(formData);
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 shadow-lg border border-gray-700 flex flex-col gap-4">
      <div className="flex justify-between items-center border-b border-gray-700 pb-2">
        <h2 className="text-lg font-semibold text-gray-100">Incident Config</h2>
        <button 
          onClick={loadFromDetection}
          className="text-xs bg-gray-700 hover:bg-gray-600 text-gray-200 py-1 px-2 rounded"
          type="button"
        >
          Load Detection
        </button>
      </div>
      
      <form onSubmit={handleSubmit} className="flex flex-col gap-3 text-sm">
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-gray-400 text-xs mb-1">Spill ID</label>
            <input 
              value={formData.spill_id} 
              onChange={e => setFormData({...formData, spill_id: e.target.value})}
              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-gray-200"
            />
          </div>
          <div>
            <label className="block text-gray-400 text-xs mb-1">Time (UTC)</label>
            <input 
              type="datetime-local"
              value={formData.observation_timestamp} 
              onChange={e => setFormData({...formData, observation_timestamp: e.target.value})}
              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-gray-200"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-gray-400 text-xs mb-1">Longitude</label>
            <input 
              value={formData.lon} 
              onChange={e => setFormData({...formData, lon: e.target.value})}
              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-gray-200"
            />
          </div>
          <div>
            <label className="block text-gray-400 text-xs mb-1">Latitude</label>
            <input 
              value={formData.lat} 
              onChange={e => setFormData({...formData, lat: e.target.value})}
              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-gray-200"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-gray-400 text-xs mb-1">Oil Type</label>
            <select 
              value={formData.oil_type}
              onChange={e => setFormData({...formData, oil_type: e.target.value})}
              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-gray-200"
            >
              <option value="light_crude">Light Crude</option>
              <option value="medium_crude">Medium Crude</option>
              <option value="heavy_crude">Heavy Crude</option>
              <option value="diesel">Diesel</option>
            </select>
          </div>
          <div>
            <label className="block text-gray-400 text-xs mb-1">Forcing Mode</label>
            <select 
              value={formData.forcing_mode}
              onChange={e => setFormData({...formData, forcing_mode: e.target.value})}
              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-gray-200"
            >
              <option value="mock">Mock Forcing</option>
              <option value="cmems">CMEMS Live</option>
              <option value="composite">Composite</option>
            </select>
          </div>
        </div>

        <button 
          type="submit" 
          disabled={isLoading}
          className={`mt-2 w-full py-2 rounded font-semibold text-white ${isLoading ? 'bg-teal-700 opacity-70 cursor-not-allowed' : 'bg-teal-600 hover:bg-teal-500'}`}
        >
          {isLoading ? 'Running Analysis...' : '▶ Run OpenDrift Analysis'}
        </button>
      </form>
    </div>
  );
}
