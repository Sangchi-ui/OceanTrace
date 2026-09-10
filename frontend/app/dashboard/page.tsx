"use client";

import React, { useState } from 'react';
import { IncidentPanel } from '@/components/dashboard/IncidentPanel';
import { OceanMap } from '@/components/dashboard/OceanMap';
import { TemporalScrubber } from '@/components/dashboard/TemporalScrubber';
import { WeatheringChart } from '@/components/dashboard/WeatheringChart';
import { MetoceanWidget } from '@/components/dashboard/MetoceanWidget';
import { ResultsPanel } from '@/components/dashboard/ResultsPanel';
import { ExportPanel } from '@/components/dashboard/ExportPanel';

export default function DashboardPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [currentFrame, setCurrentFrame] = useState<any>(null);
  
  const handleRunAnalysis = async (params: any) => {
    setIsLoading(true);
    try {
      // Mock poly if missing
      if (!params.geojson_polygon) {
        const offset = Math.sqrt(parseFloat(params.area_km2)) / 111;
        const lon = parseFloat(params.lon);
        const lat = parseFloat(params.lat);
        params.geojson_polygon = {
          type: "Polygon",
          coordinates: [[
            [lon - offset, lat - offset],
            [lon + offset, lat - offset],
            [lon + offset, lat + offset],
            [lon - offset, lat + offset],
            [lon - offset, lat - offset]
          ]]
        };
        params.centroid = [lon, lat];
      }

      const res = await fetch('http://localhost:8000/api/v2/hindcast-forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      const data = await res.json();
      setResult(data);
      if (data.animation_frames && data.animation_frames.length > 0) {
        setCurrentFrame(data.animation_frames[0]);
      } else {
        setCurrentFrame(null);
      }
    } catch (e) {
      console.error(e);
      alert("Failed to run analysis. Is backend running?");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-1 overflow-hidden p-4 gap-4 bg-gray-950 text-gray-200">
      
      {/* Left Sidebar - Controls & Status */}
      <div className="w-80 flex flex-col gap-4 overflow-y-auto no-scrollbar">
        <IncidentPanel onRunAnalysis={handleRunAnalysis} isLoading={isLoading} />
        <MetoceanWidget />
        <ResultsPanel result={result} />
        <ExportPanel result={result} />
      </div>

      {/* Center/Right - Map & Analytics */}
      <div className="flex-1 flex flex-col gap-4 min-w-0">
        
        {/* Main Map Area */}
        <div className="flex-1 relative rounded-lg border border-gray-700 shadow-xl overflow-hidden">
          <OceanMap 
            geojsonCollection={result?.geojson_collection} 
            currentFrame={currentFrame} 
          />
          
          {/* Overlay loading state */}
          {isLoading && (
            <div className="absolute inset-0 bg-gray-900 bg-opacity-70 flex items-center justify-center z-10">
              <div className="flex flex-col items-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-teal-500 mb-4"></div>
                <span className="text-teal-400 font-medium">Running OpenDrift Engine...</span>
              </div>
            </div>
          )}
        </div>

        {/* Bottom Panel - Scrubber & Charts */}
        <div className="h-64 flex gap-4 shrink-0">
          <div className="flex-1">
            <TemporalScrubber 
              frames={result?.animation_frames || []} 
              onFrameChange={(frame) => setCurrentFrame(frame)}
            />
          </div>
          <div className="flex-1">
            <WeatheringChart data={result?.weathering || []} />
          </div>
        </div>

      </div>
    </div>
  );
}
