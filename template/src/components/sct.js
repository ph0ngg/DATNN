import React, {useState} from 'react';
import Upload from './upload'
import Stream from './stream'

const App = () => {
    const [uploaded, setUploaded] = useState(false);
  
    const handleUpload = () => {
      setUploaded(true);
    };
    console.log(uploaded)
    return (
      <div className="App"  >
        <Upload onUpload={handleUpload} />
        {uploaded && <Stream />}
      </div>
    );
  };
  
  export default App;