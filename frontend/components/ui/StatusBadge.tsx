import React from 'react';

type StatusBadgeProps = {
  status: 'opendrift' | 'rk4' | 'mock' | 'cmems' | 'error' | string;
};

export function StatusBadge({ status }: StatusBadgeProps) {
  let colorClass = 'bg-gray-500 text-white';
  let label = status;

  if (status === 'opendrift') {
    colorClass = 'bg-green-600 text-white';
    label = '● OpenDrift Active';
  } else if (status === 'rk4') {
    colorClass = 'bg-amber-500 text-white';
    label = '● RK4 Fallback';
  } else if (status === 'mock') {
    colorClass = 'bg-blue-600 text-white';
    label = '● Mock Forcing';
  } else if (status === 'cmems') {
    colorClass = 'bg-green-600 text-white';
    label = '● CMEMS Live';
  } else if (status === 'error') {
    colorClass = 'bg-red-600 text-white';
    label = '⚠ Error';
  }

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${colorClass}`}>
      {label}
    </span>
  );
}
