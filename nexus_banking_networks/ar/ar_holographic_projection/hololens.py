import hololens

class ARHolographicProjection:
    def __init__(self):
        self.hololens = hololens.HoloLens()

    def project_hologram(self, hologram_data):
        # Project 3D hologram
        self.hololens.project_hologram(hologram_data)

class AdvancedARHolographicProjection:
    def __init__(self, ar_holographic_projection):
        self.ar_holographic_projection = ar_holographic_projection

    def enable_holographic_banking_experience(self, hologram_data):
        # Enable holographic banking experience
        self.ar_holographic_projection.project_hologram(hologram_data)
