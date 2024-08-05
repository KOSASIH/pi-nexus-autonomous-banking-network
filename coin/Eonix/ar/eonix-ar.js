// Eonix AR
const EonixAR = {
  // Engine
  engine: 'ARKit',
  // Models
  models: [
    {
      name: 'EonixLogo',
      type: '3DModel',
      file: 'eonix_logo.obj',
    },
    {
      name: 'EonixAvatar',
      type: '3DModel',
      file: 'eonix_avatar.obj',
    },
  ],
  // Tracking
  tracking: {
    type: 'Markerless',
    parameters: {
      camera: 'rear',
      resolution: '1080p',
    },
  },
};
