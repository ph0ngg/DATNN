import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Home from './components/home';
import MCT from './components/mct';
import SCT from './components/sct';
import MCT_RT from './components/mct_rt'
import './App.css';

const App = () => {
  const [showNav, setShowNav] = useState(true);

  return (
    <Router>
      <div className="App">
        {showNav && (
          <nav>
            <ul>
              <li>
                <Link to="/">Home</Link>
              </li>
              <li>
                <Link to="/mct">MCT</Link>
              </li>
              <li>
                <Link to="/sct">SCT</Link>
              </li>
              <li>
                <Link to="/mct_rt">MCT_RT</Link>
              </li>
            </ul>
          </nav>
        )}
        <button onClick={() => setShowNav(!showNav)}>
          {showNav ? 'Hide Menu' : 'Show Menu'}
        </button>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/mct" element={<MCT />} />
          <Route path="/sct" element={<SCT />} />
          <Route path='/mct_rt' element={<MCT_RT />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;