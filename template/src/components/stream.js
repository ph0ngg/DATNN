import React, { useEffect, useState, useRef } from 'react';

function Stream() {
  const [imageSrc, setImageSrc] = useState('');
  const imageUrlRef = useRef('');

  useEffect(() => {
    let isSubscribed = true;

    const fetchImage = async () => {
      if (!isSubscribed) return;

      try {
        const response = await fetch('http://localhost:8000/video_stream/', {
          headers: {
            'Accept': 'image/jpeg'
          }
        });
        if (response.status == 204) {
          console.log('video ended')
          return;
        }

        if (!response.ok) {
          throw new Error('Network response was not ok');
        }

        const imageBlob = await response.blob();
        const imageObjectURL = URL.createObjectURL(imageBlob);

        // Clean up the old object URL
        if (imageUrlRef.current) {
          URL.revokeObjectURL(imageUrlRef.current);
        }

        if (isSubscribed) {
          imageUrlRef.current = imageObjectURL;
          setImageSrc(imageObjectURL);
        }
      } catch (error) {
        console.error('Error fetching image:', error);
      }

      // Schedule the next fetch
      if (isSubscribed) {
        setTimeout(fetchImage, 100); // Adjust the interval as necessary
      }
    };

    fetchImage(); // Start fetching images

    return () => {
      isSubscribed = false;
      if (imageUrlRef.current) {
        URL.revokeObjectURL(imageUrlRef.current); // Clean up on unmount
      }
    };
  }, []);

  return (
    <div className="App">
      <img src={imageSrc} alt="Video Frame" style={{width: "450px", margin: "10px"}} />
    </div>
  );
}

export default Stream;