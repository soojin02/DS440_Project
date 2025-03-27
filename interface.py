import gradio as gr
import matplotlib.pyplot as plt
import logging
from model import predict_toxicity, load_model_and_tokenizer

# Load the trained model and tokenizer
model, tokenizer = load_model_and_tokenizer('model.h5', 'tokenizer.pkl')

# Set up logging for error tracking
logging.basicConfig(filename="error.log", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

# Gradio interface function with error handling
def toxicity_predictor(text, chart_type):
    try:
        if not text.strip():
            return "Error: Please enter valid text."

        result = predict_toxicity(model, tokenizer, text)
        labels = list(result.keys())
        percentages = [round(value * 100, 2) for value in result.values()]  # Convert to percentage and round to 2 decimals

        # Check if all values are very low (below 1%)
        if sum(percentages) < 1:
            return "No significant toxicity detected."

        # Create the selected chart
        fig, ax = plt.subplots(figsize=(8, 4))

        if chart_type == "Bar Chart":
            ax.bar(labels, percentages, color=['red' if p > 50 else 'skyblue' for p in percentages])
            ax.set_ylabel('Percentage')
            ax.set_title('Toxicity Prediction')
            ax.set_ylim(0, 100)  # Set y-axis limit to 100%

            # Show percentages on bars
            for i, v in enumerate(percentages):
                ax.text(i, v + 2, f"{v:.2f}%", ha='center', fontsize=10)

        elif chart_type == "Pie Chart":
            # Only keep meaningful values (ignore 0%)
            filtered_data = [(p, l) for p, l in zip(percentages, labels) if p > 0]
            if not filtered_data:  # If everything is 0%, return a message
                return "No significant toxicity detected."

            sorted_data = sorted(filtered_data, reverse=True)  # Sort in descending order
            sorted_percentages, sorted_labels = zip(*sorted_data)

            # Assign colors dynamically based on value rankings
            colors = ["red", "orange", "skyblue"][:len(sorted_percentages)]
            
            ax.pie(sorted_percentages, labels=sorted_labels, autopct='%1.1f%%', colors=colors)
            ax.set_title('Toxicity Prediction')

        return fig

    except Exception as e:
        logging.error(f"Unexpected error: {e}")  # Log the error
        return "Error: Something went wrong. Please try again."

# Create Gradio interface
iface = gr.Interface(
    fn=toxicity_predictor,
    inputs=[
        gr.Textbox(lines=3, placeholder="Enter your text here..."),
        gr.Radio(choices=["Bar Chart", "Pie Chart"], label="Select Chart Type", value="Bar Chart")
    ],
    outputs=gr.Plot()
)

# Launch the Gradio interface
if __name__ == "__main__":
    iface.launch()
