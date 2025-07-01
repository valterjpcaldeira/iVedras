import React from 'react';
import { Link } from 'react-router-dom';

const Header = () => (
  <header style={{
    width: '100%',
    background: 'rgba(255,255,255,0.85)',
    borderBottom: '1px solid #b2e0f7',
    padding: '0.5rem 0',
    position: 'sticky',
    top: 0,
    zIndex: 100,
    boxShadow: '0 2px 12px #00aae911',
    display: 'flex',
    alignItems: 'center',
    minHeight: 64,
  }}>
    <div style={{ marginLeft: 24 }}>
      <Link to="/">
        <img src="/iVedras.png" alt="iVedras logo" style={{ height: 44, width: 'auto', objectFit: 'contain', verticalAlign: 'middle', filter: 'drop-shadow(0 2px 8px #00aae933)' }} />
      </Link>
    </div>
  </header>
);

export default Header; 