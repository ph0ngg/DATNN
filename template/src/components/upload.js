import React, { useState, useEffect } from "react";

const Upload = ({onUpload}) => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploadMessage, setUploadMessage] = useState("");


  const handleFileChange = (event) => {
    setSelectedFiles(event.target.files);
  };

  const handleUpload = async () => {
    if (selectedFiles.length > 0) {
      const formData = new FormData();
      for (let i = 0; i < selectedFiles.length; i++) {
        formData.append("videos", selectedFiles[i]);
      }

      try {
        const response = await fetch("http://localhost:8000/uploadsct/", {
          method: "POST",
          body: formData,
        });
        if (response.ok) {
          onUpload()
        }
        setUploadMessage("Videos uploaded successfully");
      } catch (error) {
        console.error("Error:", error);
        setUploadMessage("Failed to upload videos");
      }
    }  
  };
  return (
    <div>
      <h1>SingleCam Video Uploader Demo</h1>
      <input type="file" accept="video/*" multiple onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload Videos</button>
      <p>{uploadMessage}</p>
    </div>
  );
}

export default Upload;