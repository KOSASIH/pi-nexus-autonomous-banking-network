import lime

# Define an XAI explainer for the AI model
def xai_explainer(ai_model):
    explainer = lime.LIME(ai_model)
    return explainer

# Generate explanations for AI-driven decisions
def generate_explanation(explainer, input_data, output_data):
    explanation = explainer.explain(input_data, output_data)
    return explanation
