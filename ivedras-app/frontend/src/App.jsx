import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Dashboard from './pages/Dashboard';
import ComplaintWizard from './pages/ComplaintWizard';
import 'leaflet/dist/leaflet.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/submit" element={<ComplaintWizard />} />
      </Routes>
    </Router>
  );
}

export default App; 