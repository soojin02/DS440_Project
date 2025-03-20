import gradio as gr
import matplotlib.pyplot as plt
from model import predict_toxicity, load_model_and_tokenizer

# Load the trained model and tokenizer
model, tokenizer = load_model_and_tokenizer('model.h5', 'tokenizer.pkl')

# Gradio interface function
def toxicity_predictor(text):
    if not text.strip():
        return "Please enter valid text."
    
    result = predict_toxicity(model, tokenizer, text)
    labels = list(result.keys())
    percentages = [value * 100 for value in result.values()]  # Convert to percentage

    # Create bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, percentages, color=['red' if p > 50 else 'skyblue' for p in percentages])
    ax.set_ylabel('Percentage')
    ax.set_title('Toxicity Prediction')
    ax.set_ylim(0, 100)  # Set y-axis limit to 100%

    # Show percentages on bars
    for i, v in enumerate(percentages):
        ax.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=10)

    return fig

# Create Gradio interface
iface = gr.Interface(fn=toxicity_predictor, 
                     inputs=gr.Textbox(lines=3, placeholder="Enter your text here..."), 
                     outputs=gr.Plot())

# Launch the Gradio interface
if __name__ == "__main__":
    iface.launch()
