import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import { bootstrap } from './state/store.js';
import './styles/tokens.css';
import './styles/app.css';

bootstrap();

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
