import { useState, useEffect } from 'eact';
import { useBlockchain } from '@pi-nexus/blockchain-react';
import { useSocialNetwork } from '@pi-nexus/social-network-react';

const PiNexusSocialNetwork = () => {
  const [socialData, setSocialData] = useState(null);
  const { blockchain } = useBlockchain();
  const { socialNetwork } = useSocialNetwork();

  useEffect(() => {
    const fetchSocialData = async () => {
      const data = await blockchain.getSocialData();
      setSocialData(data);
    };

    fetchSocialData();
  }, [blockchain]);

  const handlePost = async (content) => {
    const postData = await socialNetwork.createPost(content);
    setSocialData((prevData) => ({
      ...prevData,
      posts: [...prevData.posts, postData],
    }));
  };

  return (
    <div>
      <h1>Pi Nexus Social Network</h1>
      {socialData && (
        <SocialDataViewer data={socialData} />
      )}
      <PostForm onSubmit={handlePost} />
    </div>
  );
};

export default PiNexusSocialNetwork;
