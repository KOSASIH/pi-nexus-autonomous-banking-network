import explainable_computer_vision

# Define an explainable computer vision model
def explainable_computer_vision_model():
    model= explainable_computer_vision.ExplainableComputerVisionModel()
    return model

# Use the explainable computer vision model to analyze images
def analyze_image(model, image):
    explanation = model.analyze_image(image)
    return explanation
