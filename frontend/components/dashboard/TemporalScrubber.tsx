"use client";

import React, { useState, useEffect } from 'react';
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react';

type TemporalScrubberProps = {
  frames: any[];
  onFrameChange: (frame: any, index: number) => void;
};

export function TemporalScrubber({ frames, onFrameChange }: TemporalScrubberProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (frames.length > 0) {
      onFrameChange(frames[currentIndex], currentIndex);
    }
  }, [currentIndex, frames]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying && frames.length > 0) {
      interval = setInterval(() => {
        setCurrentIndex(prev => {
          if (prev >= frames.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 500); // 500ms per frame
    }
    return () => clearInterval(interval);
  }, [isPlaying, frames]);

  const handlePlayPause = () => {
    if (currentIndex >= frames.length - 1) {
      setCurrentIndex(0); // Restart if at end
    }
    setIsPlaying(!isPlaying);
  };

  if (!frames || frames.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-3 shadow-lg border border-gray-700 flex flex-col gap-2 opacity-50">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">Time Controls</h3>
        <div className="text-sm text-gray-500 italic">No animation frames available</div>
      </div>
    );
  }

  const currentFrame = frames[currentIndex];

  return (
    <div className="bg-gray-800 rounded-lg p-3 shadow-lg border border-gray-700 flex flex-col gap-3">
      <div className="flex justify-between items-center">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Time Controls</h3>
        <span className="text-xs font-mono text-gray-300 bg-gray-900 px-2 py-1 rounded">
          {currentFrame?.t ? currentFrame.t.replace('T', ' ').slice(0, 16) : 'N/A'}
        </span>
      </div>
      
      <div className="flex items-center gap-4">
        <button 
          onClick={() => setCurrentIndex(0)}
          className="text-gray-400 hover:text-white"
        >
          <SkipBack size={16} />
        </button>
        <button 
          onClick={handlePlayPause}
          className="bg-teal-600 hover:bg-teal-500 text-white rounded-full p-2"
        >
          {isPlaying ? <Pause size={16} /> : <Play size={16} />}
        </button>
        <button 
          onClick={() => setCurrentIndex(frames.length - 1)}
          className="text-gray-400 hover:text-white"
        >
          <SkipForward size={16} />
        </button>
        
        <input 
          type="range" 
          min="0" 
          max={frames.length - 1} 
          value={currentIndex}
          onChange={(e) => setCurrentIndex(parseInt(e.target.value))}
          className="flex-1 accent-teal-500"
        />
      </div>
      <div className="flex justify-between text-[10px] text-gray-500 font-mono">
        <span>{frames[0]?.t.slice(5, 16).replace('T', ' ')}</span>
        <span>{frames[frames.length-1]?.t.slice(5, 16).replace('T', ' ')}</span>
      </div>
    </div>
  );
}
