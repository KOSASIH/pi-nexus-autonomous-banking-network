class AstralPlaneAPI {
  async getAssets() {
    const response = await fetch('/api/assets');
    const data = await response.json();
    return data;
  }

  async buyAsset(assetId) {
    const response = await fetch(`/api/assets/${assetId}/buy`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    const data = await response.json();
    return data;
  }
}

export default AstralPlaneAPI;
