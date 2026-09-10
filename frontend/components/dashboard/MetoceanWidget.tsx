"use client";

import React, { useEffect, useState } from 'react';

type MetoceanData = {
  wind: { speed: number; dir: number };
  current: { speed: number; dir: number };
  sst: number;
};

export function MetoceanWidget() {
  const [data, setData] = useState<MetoceanData | null>(null);

  useEffect(() => {
    // Poll metocean endpoint
    fetch('http://localhost:8000/api/v2/metocean?bbox=0,0,0,0')
      .then(res => res.json())
      .then(d => setData(d))
      .catch(console.error);
  }, []);

  if (!data) return <div className="p-2 text-gray-500 text-sm">Loading Metocean...</div>;

  return (
    <div className="bg-gray-800 rounded-lg p-3 shadow-lg border border-gray-700 flex flex-col gap-2">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Metocean Conditions</h3>
      <div className="grid grid-cols-3 gap-2 text-center mt-1">
        <div className="bg-gray-900 rounded p-2">
          <div className="text-gray-500 text-[10px] uppercase">Wind</div>
          <div className="text-gray-200 text-sm font-bold">{data.wind.speed.toFixed(1)} <span className="text-[10px] font-normal">m/s</span></div>
          <div className="text-blue-400 text-xs">↗ {data.wind.dir}°</div>
        </div>
        <div className="bg-gray-900 rounded p-2">
          <div className="text-gray-500 text-[10px] uppercase">Currents</div>
          <div className="text-gray-200 text-sm font-bold">{data.current.speed.toFixed(2)} <span className="text-[10px] font-normal">m/s</span></div>
          <div className="text-teal-400 text-xs">↗ {data.current.dir}°</div>
        </div>
        <div className="bg-gray-900 rounded p-2">
          <div className="text-gray-500 text-[10px] uppercase">SST</div>
          <div className="text-gray-200 text-sm font-bold">{data.sst.toFixed(1)} <span className="text-[10px] font-normal">°C</span></div>
          <div className="text-orange-400 text-xs">Surface</div>
        </div>
      </div>
    </div>
  );
}
