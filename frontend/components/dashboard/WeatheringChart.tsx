"use client";

import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

type WeatheringChartProps = {
  data: any[];
};

export function WeatheringChart({ data }: WeatheringChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 shadow-lg border border-gray-700 h-64 flex items-center justify-center">
        <span className="text-gray-500">No weathering data available</span>
      </div>
    );
  }

  // Format data for recharts
  const chartData = data.map((d, i) => ({
    name: `+${i}h`,
    evaporated: d.evaporated_fraction * 100,
    water: d.water_content * 100,
    surface: d.surface_oil_fraction * 100
  }));

  return (
    <div className="bg-gray-800 rounded-lg p-4 shadow-lg border border-gray-700 h-64 flex flex-col">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Weathering Profile</h3>
      <div className="flex-1 w-full min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="name" stroke="#9CA3AF" tick={{fontSize: 10}} />
            <YAxis stroke="#9CA3AF" tick={{fontSize: 10}} />
            <Tooltip 
              contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '4px' }}
              itemStyle={{ fontSize: '12px' }}
              labelStyle={{ fontSize: '12px', color: '#9CA3AF' }}
            />
            <Legend wrapperStyle={{ fontSize: '10px' }} />
            <Line type="monotone" dataKey="surface" name="Surface Oil %" stroke="#f59e0b" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="evaporated" name="Evaporated %" stroke="#3b82f6" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="water" name="Water %" stroke="#10b981" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
