import React, { useEffect, useState, useRef } from 'react';

function Stream() {
  const [imageSrc1, setImageSrc1] = useState('');
  const [imageSrc2, setImageSrc2] = useState('');
  const imageUrlRef1 = useRef('');
  const imageUrlRef2 = useRef('');

  useEffect(() => {
    let isSubscribed = true;

    const fetchImages = async () => {
      if (!isSubscribed) return;

      try {
        const response = await fetch('http://localhost:8000/mct_rt/', {
          headers: {
            'Accept': 'multipart/x-mixed-replace; boundary=frame'
          }
        });

        if (response.status === 204) {
          console.log('Video stream ended');
          return;
        }

        if (!response.ok) {
          throw new Error('Network response was not ok');
        }

        const reader = response.body.getReader();
        let chunks = [];

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
        }

        const combinedChunks = new Uint8Array(chunks.reduce((acc, val) => acc.concat(Array.from(val)), []));
        const boundary = '--frame';
        const boundaryBytes = new TextEncoder().encode(boundary);
        const images = [];

        let start = 0;
        for (let i = 0; i < combinedChunks.length; i++) {
          if (combinedChunks.subarray(i, i + boundaryBytes.length).every((value, index) => value === boundaryBytes[index])) {
            const imagePart = combinedChunks.subarray(start, i);
            const headerEndIndex = imagePart.indexOf(0xff, 1);  // Skip first byte which is part of JPEG start sequence
            if (headerEndIndex !== -1) {
              const imageData = imagePart.subarray(headerEndIndex - 1);
              images.push(new Blob([imageData], { type: 'image/jpeg' }));
            }
            start = i + boundaryBytes.length;
          }
        }

        if (images.length >= 2) {
          const imageObjectURL1 = URL.createObjectURL(images[0]);
          const imageObjectURL2 = URL.createObjectURL(images[1]);

          if (imageUrlRef1.current) {
            URL.revokeObjectURL(imageUrlRef1.current);
          }

          if (isSubscribed) {
            imageUrlRef1.current = imageObjectURL1;
            setImageSrc1(imageObjectURL1);
          }

          if (imageUrlRef2.current) {
            URL.revokeObjectURL(imageUrlRef2.current);
          }

          if (isSubscribed) {
            imageUrlRef2.current = imageObjectURL2;
            setImageSrc2(imageObjectURL2);
          }
        }
      } catch (error) {
        console.error('Error fetching images:', error);
      }

      if (isSubscribed) {
        setTimeout(fetchImages, 100); // Adjust the interval as necessary
      }
    };

    fetchImages(); // Start fetching images

    return () => {
      isSubscribed = false;
      if (imageUrlRef1.current) {
        URL.revokeObjectURL(imageUrlRef1.current); // Clean up on unmount
      }
      if (imageUrlRef2.current) {
        URL.revokeObjectURL(imageUrlRef2.current); // Clean up on unmount
      }
    };
  }, []);

  return (
    <div className="App">
      <img src={imageSrc1} alt="Video Frame 1" style={{ width: "450px", margin: "10px" }} />
      <img src={imageSrc2} alt="Video Frame 2" style={{ width: "450px", margin: "10px" }} />
    </div>
  );
}

export default Stream;