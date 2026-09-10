"use client";

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { StatusBadge } from './ui/StatusBadge';

export function GlobalNav() {
  const pathname = usePathname();
  const [engineStatus, setEngineStatus] = useState('rk4');
  
  useEffect(() => {
    fetch('http://localhost:8000/api/v2/health')
      .then(res => res.json())
      .then(data => {
        if (data.opendrift_available) {
          setEngineStatus('opendrift');
        } else {
          setEngineStatus('rk4');
        }
      })
      .catch(err => {
        setEngineStatus('error');
      });
  }, []);

  return (
    <nav className="flex items-center justify-between p-4 bg-gray-900 border-b border-gray-800 text-white shadow-md">
      <div className="flex items-center space-x-6">
        <Link href="/" className="flex items-center space-x-2">
          <span className="text-xl font-bold tracking-wider">🌊 OceanTrace</span>
        </Link>
        <div className="hidden md:flex space-x-4">
          <Link 
            href="/" 
            className={`px-3 py-2 rounded-md text-sm font-medium ${pathname === '/' ? 'bg-gray-800 text-white' : 'text-gray-300 hover:bg-gray-700 hover:text-white'}`}
          >
            Detection
          </Link>
          <Link 
            href="/dashboard" 
            className={`px-3 py-2 rounded-md text-sm font-medium ${pathname?.startsWith('/dashboard') ? 'bg-gray-800 text-white' : 'text-gray-300 hover:bg-gray-700 hover:text-white'}`}
          >
            Dashboard
          </Link>
        </div>
      </div>
      <div>
        <StatusBadge status={engineStatus} />
      </div>
    </nav>
  );
}
