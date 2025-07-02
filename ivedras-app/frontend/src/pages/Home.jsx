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
        <img 
          src="/iVedras.png" 
          alt="iVedras logo" 
          style={{ 
            width: '160px', 
            height: '160px', 
            objectFit: 'contain', 
            marginBottom: '1.2rem', 
            filter: 'drop-shadow(0 4px 24px #00aae933)',
            maxWidth: '80vw',
            ...(window.innerWidth < 700 ? { width: '100px', height: '100px', marginBottom: '0.7rem' } : {})
          }} 
        />
        <div style={{ color: '#1a2a36', fontSize: '1em', marginBottom: '1.2em', fontWeight: 400 }}>Vedras é uma app que mostra como a Inteligência Artificial pode tornar as Juntas de Freguesia mais próximas, rápidas e transparentes. Com modelos que classificam pedidos por tema e urgência, e uma plataforma simples de gestão, esta é a prova de que inovar pode ser fácil — e útil.</div>
        <div style={{ display: 'flex', flexDirection: window.innerWidth < 700 ? 'column' : 'row', alignItems: 'center', justifyContent: 'center', gap: 12, marginBottom: '1.5em' }}>
          <a href="https://github.com/valterjpcaldeira/iVedras" target="_blank" rel="noopener noreferrer" style={{ display: 'flex', alignItems: 'center', gap: 8, textDecoration: 'none', color: '#24292f', fontWeight: 600, fontSize: '1.08em', background: '#f6f8fa', borderRadius: 8, padding: '0.4em 1em', boxShadow: '0 2px 8px #00aae911', transition: 'background 0.2s', marginBottom: window.innerWidth < 700 ? 8 : 0 }}>
            <svg height="24" width="24" viewBox="0 0 16 16" fill="currentColor" style={{ display: 'inline', verticalAlign: 'middle' }} aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.19 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>
            <span>Open Source on GitHub</span>
          </a>
        </div>
        <div style={{ margin: '0.5em 0 0.5em 0', fontSize: '1em', display: 'flex', flexDirection: window.innerWidth < 700 ? 'column' : 'row', justifyContent: 'center', alignItems: 'center', gap: 12 }}>
          <Link to="/dashboard" style={{ color: '#00aae9', fontWeight: 600 }}>Dashboard</Link>
          <span style={{ color: '#e3eaf2', display: window.innerWidth < 700 ? 'none' : 'inline' }}>|</span>
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