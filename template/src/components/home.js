import React from 'react';
import { useNavigate } from 'react-router-dom';
import Button from 'react-bootstrap/Button';
import './home.css';

const Home = () => {
  const navigate = useNavigate();

  const handleMCTClick = () => {
  navigate('/mct');
  };

  const handleSCTClick = () => {
    navigate('/sct');
  };

  return (
    <div className='container'>
      <h1>Select an option:</h1>
      <div>
        <Button variant="outline-dark" onClick={handleMCTClick}>Go to MCT</Button>{' '}
        <Button variant="outline-dark" onClick={handleSCTClick}>Go to SCT</Button>{' '}
      </div>
    </div>
  );
};

export default Home;