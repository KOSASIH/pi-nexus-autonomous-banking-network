const SocialDataViewer = ({ data }) => {
  return (
    <div>
      <h2>Posts:</h2>
      {data.posts.map((post) => (
        <PostViewer key={post.id} data={post} />
      ))}
    </div>
  );
};

export default SocialDataViewer;
