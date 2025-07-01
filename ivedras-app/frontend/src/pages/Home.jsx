import React, { useEffect, useState } from 'react';
import { getComplaints } from '../api/api';
import { Link } from 'react-router-dom';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';

const markerIcon = new L.Icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
  className: 'blue-marker',
});

function Home() {
  const [complaints, setComplaints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setError('');
      try {
        const data = await getComplaints();
        setComplaints(data);
      } catch (err) {
        setError('Erro ao obter queixas.');
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  // Only show complaints with valid coordinates on the map
  const recentComplaints = complaints.slice(0, 10);
  const complaintsWithCoords = recentComplaints.filter(c => c.latitude && c.longitude);
  const mapCenter = complaintsWithCoords.length
    ? [
        complaintsWithCoords.reduce((sum, c) => sum + c.latitude, 0) / complaintsWithCoords.length,
        complaintsWithCoords.reduce((sum, c) => sum + c.longitude, 0) / complaintsWithCoords.length
      ]
    : [39.0917, -9.2589];

  return (
    <div style={{ padding: '3.5rem 0', maxWidth: 700, margin: '0 auto', width: '100%' }}>
      <div style={{ textAlign: 'center', marginBottom: '2.5rem', padding: '0 0.5em' }}>
        <img src="/iVedras.png" alt="iVedras logo" style={{ width: '160px', height: '160px', objectFit: 'contain', marginBottom: '1.2rem', filter: 'drop-shadow(0 4px 24px #00aae933)' }} />
        <div style={{ color: '#1a2a36', fontSize: '1em', marginBottom: '1.2em', fontWeight: 400 }}>Bem-vindo ao portal de queixas de Torres Vedras</div>
        <div style={{ margin: '0.5em 0 0.5em 0', fontSize: '1em', display: 'flex', justifyContent: 'center', gap: 12 }}>
          <Link to="/dashboard" style={{ color: '#00aae9', fontWeight: 600 }}>Dashboard</Link>
          <span style={{ color: '#e3eaf2' }}>|</span>
          <Link to="/submit" style={{ color: '#00aae9', fontWeight: 600 }}>Submeter Queixa</Link>
        </div>
      </div>
      <div className="card" style={{ boxShadow: '0 4px 24px rgba(0,170,233,0.10)', padding: '1.2em 0.7em', marginBottom: '1.5em', width: '100%', maxWidth: '100vw' }}>
        <h2 style={{ fontSize: '1.1em', color: '#00aae9', fontWeight: 700, margin: 0, marginBottom: '1em', letterSpacing: '-0.01em' }}>Mapa das Queixas Recentes</h2>
        {complaintsWithCoords.length > 0 ? (
          <MapContainer center={mapCenter} zoom={11} style={{ height: '300px', width: '100%', borderRadius: '10px', minWidth: 0 }}>
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            {complaintsWithCoords.map((c, i) => (
              <Marker key={i} position={[c.latitude, c.longitude]} icon={markerIcon}>
                <Popup>
                  <b>{c.problem}</b><br />
                  {c.topic && (<span><b>Categoria:</b> {c.topic}<br /></span>)}
                  {c.timestamp ? new Date(c.timestamp).toLocaleString() : ''}
                </Popup>
              </Marker>
            ))}
          </MapContainer>
        ) : (
          <div style={{ color: '#aaa', textAlign: 'center', padding: '2em 0' }}>Sem queixas recentes com localização.</div>
        )}
      </div>
      <div style={{ borderTop: '1px solid #e3eaf2', margin: '1.5em 0 1em 0' }} />
      <h2 style={{ fontSize: '1em', color: '#00aae9', fontWeight: 700, margin: 0, marginBottom: '1em', letterSpacing: '-0.01em', paddingLeft: 8 }}>Queixas Recentes</h2>
      {loading && <p style={{ color: '#00aae9', textAlign: 'center' }}>A carregar...</p>}
      {error && <p style={{ color: '#FF3B30', textAlign: 'center' }}>{error}</p>}
      {!loading && !error && (
        <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
          {recentComplaints.map((c, i) => (
            <li key={i} style={{ marginBottom: '1.2rem', padding: '0 0.5em' }}>
              <div style={{ fontWeight: 600, fontSize: '1em', color: '#1a2a36', marginBottom: 2 }}><span style={{ fontStyle: 'italic' }}>&quot;{c.problem}&quot;</span></div>
              {c.topic && (<span style={{ color: '#00aae9', fontWeight: 500, fontSize: '0.95em' }}>Categoria: {c.topic}</span>)}
              <div style={{ fontSize: '0.9em', color: '#7a8ca3', marginTop: 2 }}>{c.timestamp ? new Date(c.timestamp).toLocaleString() : ''}</div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default Home; 