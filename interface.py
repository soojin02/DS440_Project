import gradio as gr
from model import create_model, predict_toxicity, load_model_and_tokenizer

# Load the pre-trained model and tokenizer
model, tokenizer = load_model_and_tokenizer('model.h5', 'tokenizer.pkl')

# Gradio interface function for prediction
def toxicity_predictor(text):
    result = predict_toxicity(model, tokenizer, text)
    return result

# Create the Gradio interface
iface = gr.Interface(fn=toxicity_predictor, 
                     inputs=gr.Textbox(lines=2, placeholder="Enter your text here..."), 
                     outputs="json")

# Launch the Gradio interface
if __name__ == "__main__":
    iface.launch()
