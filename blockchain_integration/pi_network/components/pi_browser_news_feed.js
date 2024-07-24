import React, { useState, useEffect } from 'eact';
import { PiBrowser } from '@pi-network/pi-browser-sdk';

const PiBrowserNewsFeed = () => {
  const [newsFeed, setNewsFeed] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch news feed from Pi Browser API
    PiBrowser.getNewsFeed().then(feed => {
      setNewsFeed(feed);
      setLoading(false);
    });
  }, []);

  return (
    <div>
      <h1>Pi Browser News Feed</h1>
      {loading? (
        <p>Loading...</p>
      ) : (
        <ul>
          {newsFeed.map(article => (
            <li key={article.id}>
              <h2>
                <a href={article.url}>{article.title}</a>
              </h2>
              <p>{article.summary}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default PiBrowserNewsFeed;
