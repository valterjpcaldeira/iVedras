import React, { useState, useEffect, useRef } from 'react';
import { submitComplaint, classifyComplaint } from '../api/api';
import { useNavigate } from 'react-router-dom';
import { MapContainer, TileLayer, Marker, useMapEvents } from 'react-leaflet';
import L from 'leaflet';

// Fix default marker icon for leaflet in React
const markerIcon = new L.Icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

function LocationMarker({ location, setLocation }) {
  useMapEvents({
    click(e) {
      setLocation({ lat: e.latlng.lat, lng: e.latlng.lng });
    },
  });
  return <Marker position={[location.lat, location.lng]} icon={markerIcon} />;
}

function MapStep({ location, setLocation, setStep }) {
  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <p><b>Selecione a localização do problema no mapa (clique para marcar):</b></p>
      <MapContainer
        center={[location.lat, location.lng]}
        zoom={15}
        style={{ width: '100%', height: 300, borderRadius: '12px' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <LocationMarker location={location} setLocation={setLocation} />
      </MapContainer>
      <div style={{ marginTop: '1rem' }}>
        <label>Latitude: <input type="number" value={location.lat} onChange={e => setLocation({ ...location, lat: parseFloat(e.target.value) })} /></label>
        <label style={{ marginLeft: '1rem' }}>Longitude: <input type="number" value={location.lng} onChange={e => setLocation({ ...location, lng: parseFloat(e.target.value) })} /></label>
      </div>
      <button style={{ marginTop: '1.5rem' }} disabled={!location.lat || !location.lng} onClick={() => setStep(2)}>Próximo</button>
    </div>
  );
}

function TextStep({ text, setText, aiCategory, aiUrgency, aiCategoryConfidence, aiUrgencyConfidence, aiLoading, aiError, setStep }) {
  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <p><b>Descreva o problema:</b></p>
      <textarea
        rows={6}
        style={{ width: '100%', borderRadius: '8px', padding: '0.5rem' }}
        value={text}
        onChange={e => setText(e.target.value)}
        placeholder="Descreva o problema aqui..."
        autoFocus
      />
      {/* Live AI suggestion */}
      {text.length > 10 && (
        <div style={{ margin: '1rem 0', background: '#f1f8ff', borderRadius: '10px', padding: '0.7em 1em' }}>
          <b>Sugestão de categoria:</b> <span style={{ color: '#00aae9' }}>{aiLoading ? 'A analisar...' : aiCategory || '[Sem sugestão]'}</span>
          {aiCategoryConfidence !== null && !aiLoading && (
            <span style={{ color: '#888', marginLeft: 8 }}>(confiança: {(aiCategoryConfidence * 100).toFixed(1)}%)</span>
          )}<br />
          <b>Sugestão de urgência:</b> <span style={{ color: '#FF3B30' }}>{aiLoading ? 'A analisar...' : aiUrgency || '[Sem sugestão]'}</span>
          {aiUrgencyConfidence !== null && !aiLoading && (
            <span style={{ color: '#888', marginLeft: 8 }}>(confiança: {(aiUrgencyConfidence * 100).toFixed(1)}%)</span>
          )}
          {aiError && <div style={{ color: 'red', marginTop: 6 }}>{aiError}</div>}
        </div>
      )}
      <div style={{ marginTop: '1.5rem' }}>
        <button onClick={() => setStep(1)}>Voltar</button>
        <button style={{ marginLeft: '1rem' }} disabled={text.length < 10} onClick={() => setStep(3)}>Próximo</button>
      </div>
    </div>
  );
}

function ReviewStep({ location, text, aiCategory, aiUrgency, aiCategoryConfidence, aiUrgencyConfidence, error, setStep, handleSubmit, submitting }) {
  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <h3>Rever e Submeter</h3>
      <p><b>Localização:</b> {location.lat}, {location.lng}</p>
      <p><b>Queixa:</b> {text}</p>
      {/* AI suggestions */}
      <p><b>Categoria sugerida:</b> {aiCategory ? `${aiCategory} (${(aiCategoryConfidence * 100).toFixed(1)}%)` : '[Sem sugestão]'}</p>
      <p><b>Urgência sugerida:</b> {aiUrgency ? `${aiUrgency} (${(aiUrgencyConfidence * 100).toFixed(1)}%)` : '[Sem sugestão]'}</p>
      <div style={{ marginTop: '1.5rem' }}>
        <button onClick={() => setStep(2)}>Voltar</button>
        <button style={{ marginLeft: '1rem' }} onClick={handleSubmit} disabled={submitting}>{submitting ? 'A Submeter...' : 'Submeter Queixa'}</button>
      </div>
      {error && <div style={{ color: 'red', marginTop: '1rem' }}>{error}</div>}
    </div>
  );
}

function ComplaintWizard() {
  const [step, setStep] = useState(1);
  const [location, setLocation] = useState({ lat: 39.0917, lng: -9.2589 });
  const [locationSelected, setLocationSelected] = useState(false);
  const [text, setText] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState('');
  const [aiCategory, setAiCategory] = useState('');
  const [aiUrgency, setAiUrgency] = useState('');
  const [aiCategoryConfidence, setAiCategoryConfidence] = useState(null);
  const [aiUrgencyConfidence, setAiUrgencyConfidence] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiError, setAiError] = useState('');
  const navigate = useNavigate();
  const debounceRef = useRef();

  useEffect(() => {
    if (text.length < 10) {
      setAiCategory('');
      setAiUrgency('');
      setAiCategoryConfidence(null);
      setAiUrgencyConfidence(null);
      setAiError('');
      return;
    }
    setAiLoading(true);
    setAiError('');
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      try {
        const res = await classifyComplaint(text);
        setAiCategory(res.topic);
        setAiUrgency(res.urgency);
        setAiCategoryConfidence(res.topic_confidence);
        setAiUrgencyConfidence(res.urgency_confidence);
        setAiLoading(false);
      } catch (err) {
        setAiError('Erro ao obter sugestão AI.');
        setAiLoading(false);
      }
    }, 600);
    return () => clearTimeout(debounceRef.current);
  }, [text]);

  // Center map on user's current location if available
  useEffect(() => {
    if (step === 1 && !locationSelected && navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          setLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude });
        },
        () => {},
        { enableHighAccuracy: true, timeout: 10000 }
      );
    }
  }, [step, locationSelected]);

  // Step 1: Location picker (map click)
  const handleMapClick = (e) => {
    setLocation({ lat: e.latLng.lat(), lng: e.latLng.lng() });
    setLocationSelected(true);
  };

  // Step 2: Complaint text (with live suggestion placeholder)
  // (You can add live suggestion logic here later)

  // Step 3: Review & submit
  const handleSubmit = async () => {
    setSubmitting(true);
    setError('');
    try {
      await submitComplaint({
        problem: text,
        location: `Lat: ${location.lat}, Lng: ${location.lng}`,
        latitude: location.lat,
        longitude: location.lng,
        topic: aiCategory,
        urgency: aiUrgency,
        // Add more fields as needed
      });
      setSuccess(true);
    } catch (err) {
      setError('Erro ao submeter queixa.');
    } finally {
      setSubmitting(false);
    }
  };

  if (success) {
    return (
      <div style={{ padding: '2rem' }}>
        <h2>✅ Queixa registada com sucesso!</h2>
        <button onClick={() => navigate('/')}>Ver Dashboard</button>
      </div>
    );
  }

  return (
    <div style={{ padding: '2rem', maxWidth: 600, margin: '0 auto' }}>
      <h1>Submeter Queixa</h1>
      <div style={{ margin: '2rem 0' }}>
        {step === 1 && (
          <MapStep location={location} setLocation={setLocation} setStep={setStep} />
        )}
        {step === 2 && (
          <TextStep
            text={text}
            setText={setText}
            aiCategory={aiCategory}
            aiUrgency={aiUrgency}
            aiCategoryConfidence={aiCategoryConfidence}
            aiUrgencyConfidence={aiUrgencyConfidence}
            aiLoading={aiLoading}
            aiError={aiError}
            setStep={setStep}
          />
        )}
        {step === 3 && (
          <ReviewStep
            location={location}
            text={text}
            aiCategory={aiCategory}
            aiUrgency={aiUrgency}
            aiCategoryConfidence={aiCategoryConfidence}
            aiUrgencyConfidence={aiUrgencyConfidence}
            error={error}
            setStep={setStep}
            handleSubmit={handleSubmit}
            submitting={submitting}
          />
        )}
      </div>
    </div>
  );
}

export default ComplaintWizard; 