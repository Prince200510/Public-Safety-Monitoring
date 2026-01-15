import React, { useMemo } from 'react'
import { GoogleMap, Marker, useJsApiLoader } from '@react-google-maps/api'

const containerStyle = { width: '100%', height: '100%', minHeight: '400px', borderRadius: '1rem' };

const defaultCenter = { lat: 19.2183,  lng: 72.8367 };

export default function Map({ locations = [] }) {
  const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;

  if (!apiKey || apiKey === '') {
    return (
      <div className="w-full h-96 bg-slate-100 flex flex-col gap-2 items-center justify-center rounded-xl text-slate-500 border border-slate-200 p-4 text-center">
        <p className="font-semibold">Google Maps API Key Missing</p>
        <p className="text-sm">Please add VITE_GOOGLE_MAPS_API_KEY to frontend/.env</p>
        <div className="mt-4 text-xs text-left w-full overflow-auto bg-slate-200 p-2 rounded">
          <strong>Locations to display:</strong>
          <pre>{JSON.stringify(locations, null, 2)}</pre>
        </div>
      </div>
    )
  }

  const { isLoaded, loadError } = useJsApiLoader({
    id: 'google-map-script',
    googleMapsApiKey: apiKey,
  })

  const center = useMemo(() => {
    return locations.length > 0
      ? { lat: locations[0].latitude, lng: locations[0].longitude }
      : defaultCenter
  }, [locations])

  if (loadError) {
    return (
      <div className="w-full h-96 bg-slate-100 flex flex-col gap-2 items-center justify-center rounded-xl text-slate-500 border border-slate-200 p-4 text-center">
        <p className="font-semibold">Failed to load Google Maps</p>
        <p className="text-sm">{String(loadError?.message || loadError)}</p>
      </div>
    )
  }

  if (!isLoaded) {
    return (
      <div className="w-full h-96 bg-slate-100 flex flex-col gap-2 items-center justify-center rounded-xl text-slate-500 border border-slate-200 p-4 text-center">
        <p className="font-semibold">Loading map…</p>
      </div>
    )
  }

  return (
    <div className="h-96 w-full rounded-xl overflow-hidden shadow-sm border border-slate-200/70">
      <GoogleMap mapContainerStyle={containerStyle} center={center} zoom={13}>
        {locations.map((loc) => (
          <Marker key={loc.user_email} position={{ lat: loc.latitude, lng: loc.longitude }} title={`${loc.user_email} (${loc.timestamp})`} />
        ))}
      </GoogleMap>
    </div>
  )
}
