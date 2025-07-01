import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Dashboard from './pages/Dashboard';
import ComplaintWizard from './pages/ComplaintWizard';
import 'leaflet/dist/leaflet.css';
import Header from './components/Header';

function App() {
  return (
    <div style={{ minHeight: '100vh', background: '#c4eafa' }}>
      <Router>
        <Header />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/submit" element={<ComplaintWizard />} />
        </Routes>
      </Router>
    </div>
  );
}

export default App; 