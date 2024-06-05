
import React, { useState } from "react";
import './mct.css'

function MCT() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploadMessage, setUploadMessage] = useState("");
  const [processedVideoURLs, setProcessedVideoURLs] = useState([]);

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
        const response = await fetch("http://localhost:8000/upload/", {
          method: "POST",
          body: formData,
        });
        const data = await response.json();
        setUploadMessage("Videos uploaded successfully");
        const processedURLs = data.processed_video_paths.map((path) => {
          console.log(path); // In ra giá trị của path
          console.log(path.split("\\").pop())
          return `http://localhost:8000/processed_video/${path.split("\\").pop()}`;
        });
        setProcessedVideoURLs(processedURLs);
      } catch (error) {
        console.error("Error:", error);
        setUploadMessage("Failed to upload videos");
      }
    }
  };

  return (
    <div>
      <h1>MultiCam Video Uploader Demo</h1>
      <input type="file" accept="video/*" multiple onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload Videos</button>
      <p>{uploadMessage}</p>
      <div style={{ display: "flex", flexWrap: "wrap" }}>
        {processedVideoURLs.map((url, index) => (
          <video key={index} controls style={{ width: "450px", margin: "10px" }}>
            <source src={url} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        ))}
      </div>
    </div>
  );
}

export default MCT;