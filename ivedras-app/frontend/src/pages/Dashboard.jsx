import React, { useEffect, useState } from 'react';
import { getComplaints } from '../api/api';
import { MapContainer, TileLayer, Marker, Popup, useMap, CircleMarker, Tooltip } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet.heat';
import { Bar, Doughnut, Line } from 'react-chartjs-2';
import { Chart, CategoryScale, LinearScale, BarElement, ArcElement, Tooltip as ChartTooltip, Legend, LineElement, PointElement, TimeScale } from 'chart.js';
import 'chartjs-adapter-date-fns';
import MarkerClusterGroup from 'react-leaflet-markercluster';
import 'leaflet.markercluster/dist/MarkerCluster.css';
import 'leaflet.markercluster/dist/MarkerCluster.Default.css';

Chart.register(CategoryScale, LinearScale, BarElement, ArcElement, ChartTooltip, Legend, LineElement, PointElement, TimeScale);

const markerIcon = new L.Icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

// Helper: group complaints by rounded lat/lng (zone)
function groupByZone(complaints, precision = 2) {
  const zones = {};
  complaints.forEach(c => {
    if (!c.latitude || !c.longitude) return;
    const lat = c.latitude.toFixed(precision);
    const lng = c.longitude.toFixed(precision);
    const key = `${lat},${lng}`;
    if (!zones[key]) zones[key] = { lat: +lat, lng: +lng, count: 0 };
    zones[key].count++;
  });
  return Object.values(zones);
}

function HeatmapLayer({ points }) {
  const map = useMap();
  useEffect(() => {
    if (!points.length) return;
    const heatLayer = L.heatLayer(points, {
      radius: 48,
      blur: 38,
      gradient: {
        0.0: 'rgba(0,0,0,0)',
        0.3: '#00aae9',
        0.6: '#ffe066',
        0.85: '#ff3b30',
        1.0: '#d90429'
      },
      minOpacity: 0.18,
      max: 1.0,
      maxZoom: 18,
    }).addTo(map);
    return () => { map.removeLayer(heatLayer); };
  }, [points, map]);
  return null;
}

function Dashboard() {
  const [complaints, setComplaints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [topicFilter, setTopicFilter] = useState('');
  const [urgencyFilter, setUrgencyFilter] = useState('');

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

  const validComplaints = complaints.filter(c => c.latitude && c.longitude);
  const filteredComplaints = validComplaints.filter(c =>
    (!topicFilter || c.topic === topicFilter) &&
    (!urgencyFilter || c.urgency === urgencyFilter)
  );
  const mapCenter = filteredComplaints.length
    ? [
        filteredComplaints.reduce((sum, c) => sum + c.latitude, 0) / filteredComplaints.length,
        filteredComplaints.reduce((sum, c) => sum + c.longitude, 0) / filteredComplaints.length
      ]
    : [39.0917, -9.2589];

  // --- Bubble Map by zone ---
  const bubbleZones = groupByZone(filteredComplaints, 2); // precision=2 ~ 1km
  const bubbleData = bubbleZones.map(z => {
    const complaintsInZone = filteredComplaints.filter(c => c.latitude && c.longitude && c.latitude.toFixed(2) == z.lat.toFixed(2) && c.longitude.toFixed(2) == z.lng.toFixed(2));
    const votes = complaintsInZone.reduce((sum, c) => sum + (c.votes || 0), 0);
    return {
      lat: z.lat,
      lng: z.lng,
      count: z.count,
      votes,
      tooltip: `${z.count} pedidos${votes > 0 ? `, ${votes} votos` : ''}`
    };
  });

  // --- Charts ---
  // 1. Complaints over time (last 30 days)
  const now = new Date();
  const days = Array.from({ length: 30 }, (_, i) => {
    const d = new Date(now);
    d.setDate(now.getDate() - (29 - i));
    return d;
  });
  const complaintsByDay = days.map(day => {
    const dayStr = day.toISOString().slice(0, 10);
    return filteredComplaints.filter(c => c.timestamp && c.timestamp.slice(0, 10) === dayStr).length;
  });

  // 2. Stacked bar: category x urgency
  const topicUrgency = {};
  filteredComplaints.forEach(c => {
    if (!c.topic) return;
    if (!topicUrgency[c.topic]) topicUrgency[c.topic] = {};
    topicUrgency[c.topic][c.urgency || 'Sem urgência'] = (topicUrgency[c.topic][c.urgency || 'Sem urgência'] || 0) + 1;
  });
  const allTopics = Object.keys(topicUrgency);
  const allUrgencies = Array.from(new Set(filteredComplaints.map(c => c.urgency || 'Sem urgência')));
  const stackedData = {
    labels: allTopics,
    datasets: allUrgencies.map((urg, idx) => ({
      label: urg,
      data: allTopics.map(t => topicUrgency[t][urg] || 0),
      backgroundColor: ['#00aae9', '#0086b3', '#e3eaf2', '#1a2a36'][idx % 4],
      borderRadius: 6,
      maxBarThickness: 28,
    })),
  };

  // 3. Pie/Bar for topics/urgency
  const topicCounts = {};
  const urgencyCounts = {};
  filteredComplaints.forEach(c => {
    if (c.topic) topicCounts[c.topic] = (topicCounts[c.topic] || 0) + 1;
    if (c.urgency) urgencyCounts[c.urgency] = (urgencyCounts[c.urgency] || 0) + 1;
  });
  const topicLabels = Object.keys(topicCounts);
  const topicData = Object.values(topicCounts);
  const urgencyLabels = Object.keys(urgencyCounts);
  const urgencyData = Object.values(urgencyCounts);

  // Pending and solved complaints
  const pendingComplaints = complaints.filter(c => (c.status || 'pending') === 'pending');
  const solvedComplaints = complaints.filter(c => c.status === 'solved');

  // Pagination for pending complaints
  const [page, setPage] = useState(1);
  const pageSize = 10;
  const sortedPending = [...pendingComplaints].sort((a, b) => (b.votes || 0) - (a.votes || 0));
  const pagedPending = sortedPending.slice((page - 1) * pageSize, page * pageSize);

  // --- Status chart data (handle any status code and show percentage) ---
  const statusCounts = complaints.reduce((acc, c) => {
    const status = (c.status || 'Pendente').toLowerCase();
    acc[status] = (acc[status] || 0) + 1;
    return acc;
  }, {});
  const statusLabels = Object.keys(statusCounts).map(s => s.charAt(0).toUpperCase() + s.slice(1));
  const statusData = Object.values(statusCounts);
  const statusTotal = statusData.reduce((a, b) => a + b, 0);
  const statusChartData = {
    labels: statusLabels.map((l, i) => `${l} (${((statusData[i] / statusTotal) * 100).toFixed(1)}%)`),
    datasets: [{
      data: statusData,
      backgroundColor: ['#ff9800', '#2ecc40', '#00aae9', '#ffe066', '#ff3b30', '#b2e0f7'],
      borderWidth: 0,
    }],
  };

  // Bar chart for requests per status
  const barStatusChartData = {
    labels: statusLabels,
    datasets: [{
      label: 'Pedidos',
      data: statusData,
      backgroundColor: ['#ff9800', '#2ecc40', '#00aae9', '#ffe066', '#ff3b30', '#b2e0f7'],
      borderRadius: 8,
    }],
  };

  // Table: show all requests that are not in the state 'Resolvido'
  const notSolved = complaints.filter(c => (c.status || '').toLowerCase() !== 'resolvido');
  const sortedNotSolved = [...notSolved].sort((a, b) => (b.votes || 0) - (a.votes || 0));
  const pagedNotSolved = sortedNotSolved.slice((page - 1) * pageSize, page * pageSize);

  // --- Layout ---
  return (
    <div style={{ padding: '2.5rem 0', maxWidth: 900, margin: '0 auto', width: '100%' }}>
      <h1 style={{ color: '#00aae9', fontWeight: 800, fontSize: '2.1em', marginBottom: '0.5em', padding: '0 0.5em' }}>Dashboard iVedras</h1>
      <div className="card" style={{ background: '#fff', boxShadow: '0 4px 24px rgba(0,170,233,0.10)', padding: '1.2em 0.7em', minWidth: 0, marginBottom: '2.2rem', width: '100%', maxWidth: '100vw' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center', marginBottom: 14 }}>
          <span style={{ fontWeight: 700, color: '#00aae9', fontSize: '1.08em' }}>Mapa de Calor por Zona</span>
          <select value={topicFilter} onChange={e => setTopicFilter(e.target.value)} style={{ marginLeft: 'auto', minWidth: 120 }}>
            <option value="">Todas as Categorias</option>
            {allTopics.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
          <select value={urgencyFilter} onChange={e => setUrgencyFilter(e.target.value)} style={{ minWidth: 120 }}>
            <option value="">Todas as Urgências</option>
            {allUrgencies.map(u => <option key={u} value={u}>{u}</option>)}
          </select>
        </div>
        <MapContainer center={mapCenter} zoom={11} style={{ height: '320px', width: '100%', borderRadius: '10px', minWidth: 0 }}>
          <TileLayer
            attribution='&copy; <a href="https://carto.com/attributions">CARTO</a>'
            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          />
          <MarkerClusterGroup>
            {filteredComplaints.map((c, i) => (
              <Marker key={c._id} position={[c.latitude, c.longitude]} icon={markerIcon}>
                <Popup>
                  <b>{c.problem}</b><br />
                  {c.topic && (<span><b>Categoria:</b> {c.topic}<br /></span>)}
                  {c.urgency && (<span><b>Urgência:</b> {c.urgency}<br /></span>)}
                  <span><b>Votos:</b> {c.votes || 0}<br /></span>
                  <span><b>Status:</b> {c.status || 'Pendente'}<br /></span>
                  {c.timestamp ? new Date(c.timestamp).toLocaleString() : ''}
                </Popup>
              </Marker>
            ))}
          </MarkerClusterGroup>
        </MapContainer>
        <div style={{ marginTop: 8, textAlign: 'right', fontSize: '0.95em', color: '#0077a9', opacity: 0.8 }}>
          <span style={{ background: 'linear-gradient(90deg, #b2e0f7 0%, #00aae9 60%, #ff3b30 100%)', borderRadius: 8, padding: '0.2em 1.2em', marginRight: 8, display: 'inline-block', height: 12, verticalAlign: 'middle' }}></span>
          <span>Mais afetado</span>
        </div>
      </div>
      <div style={{ marginBottom: '2.2rem', padding: '0 0.5em' }}>
        <h2 style={{ color: '#00aae9', fontWeight: 700, fontSize: '1.1em', margin: '1.2em 0 0.7em 0' }}>Evolução das Queixas (30 dias)</h2>
        <div className="card" style={{ background: '#fff', boxShadow: '0 4px 24px rgba(0,170,233,0.10)', padding: '1em 0.7em', minHeight: 80, maxHeight: 120, width: '100%', maxWidth: '100vw' }}>
          <Line
            data={{
              labels: days.map(d => d.toISOString().slice(0, 10)),
              datasets: [{
                label: 'Queixas/dia',
                data: complaintsByDay,
                fill: true,
                backgroundColor: 'rgba(0,170,233,0.10)',
                borderColor: '#00aae9',
                tension: 0.3,
                pointRadius: 2.5,
              }],
            }}
            options={{
              plugins: { legend: { display: false } },
              scales: { x: { grid: { display: false }, ticks: { color: '#7a8ca3', maxTicksLimit: 7 } }, y: { beginAtZero: true, grid: { color: '#e3eaf2' } } },
              responsive: true,
              maintainAspectRatio: false,
              height: 80,
            }}
            height={80}
          />
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem', marginBottom: '2.2rem', padding: '0 0.5em' }}>
        <div className="card" style={{ background: '#fff', boxShadow: '0 4px 24px rgba(0,170,233,0.10)', padding: '1.1em 0.8em', flex: 1, minWidth: 0, maxHeight: 220, overflowY: 'auto', width: '100%', maxWidth: '100vw' }}>
          <span style={{ fontWeight: 700, color: '#00aae9', fontSize: '1.02em' }}>Categorias x Urgência</span>
          <Bar
            data={stackedData}
            options={{
              plugins: { legend: { position: 'bottom', labels: { color: '#1a2a36', font: { weight: 500 } } } },
              responsive: true,
              maintainAspectRatio: false,
              height: 180,
              scales: {
                x: { stacked: true, grid: { display: false }, ticks: { color: '#7a8ca3', maxRotation: 30, minRotation: 30, autoSkip: false } },
                y: { stacked: true, beginAtZero: true, grid: { color: '#e3eaf2' } },
              },
            }}
            height={180}
          />
        </div>
        <div style={{ display: 'flex', flexDirection: 'row', gap: '1.2rem', flexWrap: 'wrap', width: '100%' }}>
          <div className="card" style={{ background: '#fff', boxShadow: '0 4px 24px rgba(0,170,233,0.10)', padding: '1.1em 0.8em', minHeight: 80, minWidth: 0, maxHeight: 220, overflowY: 'auto', flex: 1, width: '100%', maxWidth: '100vw' }}>
            <span style={{ fontWeight: 700, color: '#00aae9', fontSize: '1.02em' }}>Categorias</span>
            <Bar
              data={{
                labels: topicLabels,
                datasets: [{
                  label: 'Queixas',
                  data: topicData,
                  backgroundColor: '#00aae9',
                  borderRadius: 8,
                }],
              }}
              options={{
                plugins: { legend: { display: false } },
                scales: { x: { grid: { display: false }, ticks: { color: '#7a8ca3', maxRotation: 30, minRotation: 30, autoSkip: false } }, y: { beginAtZero: true, grid: { color: '#e3eaf2' } } },
                responsive: true,
                maintainAspectRatio: false,
                height: 180,
              }}
              height={180}
            />
          </div>
          <div className="card" style={{ background: '#fff', boxShadow: '0 4px 24px rgba(0,170,233,0.10)', padding: '1.1em 0.8em', minHeight: 80, minWidth: 0, maxHeight: 220, overflowY: 'auto', flex: 1, width: '100%', maxWidth: '100vw' }}>
            <span style={{ fontWeight: 700, color: '#00aae9', fontSize: '1.02em' }}>Urgência</span>
            <Doughnut
              data={{
                labels: urgencyLabels,
                datasets: [{
                  data: urgencyData,
                  backgroundColor: ['#00aae9', '#0086b3', '#e3eaf2'],
                  borderWidth: 0,
                }],
              }}
              options={{
                plugins: { legend: { position: 'bottom', labels: { color: '#1a2a36', font: { weight: 500 } } } },
                cutout: '70%',
                responsive: true,
                maintainAspectRatio: false,
                height: 80,
              }}
              height={180}
            />
          </div>
        </div>
      </div>
      <div className="card" style={{ background: '#fff', boxShadow: '0 4px 24px rgba(0,170,233,0.10)', padding: '1.2em 1em', width: '100%', maxWidth: '100vw' }}>
        <h3 style={{ color: '#00aae9', fontWeight: 700, fontSize: '1.1em', marginBottom: 18 }}>Pedidos Não Resolvidos (ordenados por votos)</h3>
        <div style={{ maxWidth: 300, margin: '0 auto 1.5em auto' }}>
          <Doughnut data={statusChartData} options={{ plugins: { legend: { position: 'bottom', labels: { color: '#1a2a36', font: { weight: 500 } } } }, cutout: '70%', responsive: true, maintainAspectRatio: false, height: 80 }} height={180} />
        </div>
        <div style={{ maxWidth: 500, margin: '0 auto 2em auto' }}>
          <Bar data={barStatusChartData} options={{
            plugins: { legend: { display: false } },
            scales: { x: { grid: { display: false }, ticks: { color: '#7a8ca3' } }, y: { beginAtZero: true, grid: { color: '#e3eaf2' } } },
            responsive: true,
            maintainAspectRatio: false,
            height: 180,
          }} height={180} />
        </div>
        {loading && <p style={{ color: '#00aae9' }}>A carregar...</p>}
        {error && <p style={{ color: '#FF3B30' }}>{error}</p>}
        {!loading && !error && (
          <>
            <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '1rem', fontSize: '1em', wordBreak: 'break-word' }}>
              <thead>
                <tr style={{ background: '#f7fbfd' }}>
                  <th style={{ padding: '0.5rem', border: '1px solid #e3eaf2' }}>Data</th>
                  <th style={{ padding: '0.5rem', border: '1px solid #e3eaf2' }}>Queixa</th>
                  <th style={{ padding: '0.5rem', border: '1px solid #e3eaf2' }}>Urgência</th>
                  <th style={{ padding: '0.5rem', border: '1px solid #e3eaf2' }}>Categoria</th>
                  <th style={{ padding: '0.5rem', border: '1px solid #e3eaf2' }}>Votos</th>
                </tr>
              </thead>
              <tbody>
                {pagedNotSolved.map((c, i) => (
                  <tr key={c._id}>
                    <td style={{ padding: '0.5rem', border: '1px solid #e3eaf2' }}>{c.timestamp ? new Date(c.timestamp).toLocaleString() : ''}</td>
                    <td style={{ padding: '0.5rem', border: '1px solid #e3eaf2', fontStyle: 'italic' }}>&quot;{c.problem}&quot;</td>
                    <td style={{ padding: '0.5rem', border: '1px solid #e3eaf2' }}>{c.urgency || '-'}</td>
                    <td style={{ padding: '0.5rem', border: '1px solid #e3eaf2' }}>{c.topic || '-'}</td>
                    <td style={{ padding: '0.5rem', border: '1px solid #e3eaf2', color: '#00aae9', fontWeight: 700 }}>{c.votes || 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ display: 'flex', justifyContent: 'center', gap: 8, marginTop: 16 }}>
              <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}>Anterior</button>
              <span style={{ fontWeight: 600, color: '#00aae9' }}>{page} / {Math.ceil(sortedNotSolved.length / pageSize)}</span>
              <button onClick={() => setPage(p => Math.min(Math.ceil(sortedNotSolved.length / pageSize), p + 1))} disabled={page === Math.ceil(sortedNotSolved.length / pageSize)}>Próxima</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default Dashboard; 