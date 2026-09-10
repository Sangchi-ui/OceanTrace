"use client";

import React, { useMemo } from 'react';
import Map, { Source, Layer, NavigationControl } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';

type OceanMapProps = {
  geojsonCollection: any;
  currentFrame: any;
};

export function OceanMap({ geojsonCollection, currentFrame }: OceanMapProps) {

  const particleGeojson = useMemo(() => {
    if (!currentFrame || !currentFrame.particles) return null;
    return {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          geometry: {
            type: "MultiPoint",
            coordinates: currentFrame.particles
          },
          properties: {
            phase: currentFrame.phase
          }
        }
      ]
    };
  }, [currentFrame]);

  // Find centroid to set initial map view if available
  const initialViewState = useMemo(() => {
    let lon = 103.82;
    let lat = 1.35;
    let zoom = 9;

    if (geojsonCollection?.features) {
      const observed = geojsonCollection.features.find((f: any) => f.properties?.layer === 'observed_spill');
      if (observed && observed.geometry?.coordinates) {
        // Just grab the first coordinate of the polygon for simplicity
        try {
          const coords = observed.geometry.coordinates[0][0];
          lon = coords[0];
          lat = coords[1];
        } catch(e) {}
      }
    }
    return { longitude: lon, latitude: lat, zoom };
  }, [geojsonCollection]);

  return (
    <div className="w-full h-full relative bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
      <Map
        initialViewState={initialViewState}
        mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
        attributionControl={false}
      >
        <NavigationControl position="bottom-right" />

        {/* Static GeoJSON Layers */}
        {geojsonCollection && (
          <Source id="agent2-layers" type="geojson" data={geojsonCollection}>
            
            {/* Probable Origin Fill */}
            <Layer 
              id="origin-fill" 
              type="fill" 
              filter={['==', 'layer', 'probable_origin_region']}
              paint={{
                'fill-color': ['get', 'fill'],
                'fill-opacity': ['get', 'fill_opacity']
              }} 
            />
            {/* Probable Origin Outline */}
            <Layer 
              id="origin-line" 
              type="line" 
              filter={['==', 'layer', 'probable_origin_region']}
              paint={{
                'line-color': ['get', 'stroke'],
                'line-width': 2
              }} 
            />

            {/* Observed Spill */}
            <Layer 
              id="observed-fill" 
              type="fill" 
              filter={['==', 'layer', 'observed_spill']}
              paint={{
                'fill-color': ['get', 'fill'],
                'fill-opacity': ['get', 'fill_opacity']
              }} 
            />

            {/* Best Hindcast Trajectory */}
            <Layer 
              id="hindcast-line" 
              type="line" 
              filter={['==', 'layer', 'best_hindcast_trajectory']}
              paint={{
                'line-color': ['get', 'stroke'],
                'line-width': ['get', 'stroke_width']
              }} 
            />

            {/* Forecast Envelopes 50% */}
            <Layer 
              id="forecast-50-fill" 
              type="fill" 
              filter={['in', 'forecast_envelope_50', ['get', 'layer']]}
              paint={{
                'fill-color': ['get', 'fill'],
                'fill-opacity': ['get', 'fill_opacity']
              }} 
            />
            
            {/* Forecast Envelopes 90% */}
            <Layer 
              id="forecast-90-fill" 
              type="fill" 
              filter={['in', 'forecast_envelope_90', ['get', 'layer']]}
              paint={{
                'fill-color': ['get', 'fill'],
                'fill-opacity': ['get', 'fill_opacity']
              }} 
            />

            {/* Forecast Centroids */}
            <Layer 
              id="forecast-centroids" 
              type="circle" 
              filter={['in', 'forecast_centroid', ['get', 'layer']]}
              paint={{
                'circle-color': '#ffffff',
                'circle-radius': 4,
                'circle-stroke-width': 1,
                'circle-stroke-color': '#000000'
              }} 
            />

          </Source>
        )}

        {/* Dynamic Particles */}
        {particleGeojson && (
          <Source id="dynamic-particles" type="geojson" data={particleGeojson}>
            <Layer
              id="particles-point"
              type="circle"
              paint={{
                'circle-color': [
                  'match',
                  ['get', 'phase'],
                  'hindcast', '#f4a261', // Orange for hindcast
                  'forecast', '#2a9d8f', // Teal for forecast
                  '#ffffff'
                ],
                'circle-radius': 2,
                'circle-opacity': 0.8
              }}
            />
          </Source>
        )}
        
      </Map>
    </div>
  );
}
