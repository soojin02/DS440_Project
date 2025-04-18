import gradio as gr
import matplotlib.pyplot as plt
from model import predict_toxicity, load_model_and_tokenizer

# Load the model and tokenizer
model, tokenizer = load_model_and_tokenizer('model.h5', 'tokenizer.pkl')

# Toxicity prediction function
def toxicity_predictor(text, chart_type):
    text = text.strip()

    if not text:
        return None, "No input detected. Please provide some text."
    if len(text) > 200:
        return None, "Input text is too long. Please shorten your text."
    if not chart_type:
        return None, "Please select a chart type (Bar Chart or Pie Chart)."

    try:
        result = predict_toxicity(model, tokenizer, text)
        labels, percentages = zip(*[(k, round(v * 100, 2)) for k, v in result.items()])

        if sum(percentages) < 1:
            return None, "No significant toxicity detected."

        fig, ax = plt.subplots(figsize=(8, 4))

        if chart_type == "Bar Chart":
            ax.bar(labels, percentages, color=['red' if p > 50 else 'skyblue' for p in percentages])
            ax.set_ylabel('Percentage')
            ax.set_title('Toxicity Prediction')
            ax.set_ylim(0, 100)

            for i, v in enumerate(percentages):
                ax.text(i, v + 2, f"{v:.2f}%", ha='center', fontsize=10)

        elif chart_type == "Pie Chart":
            filtered_data = [(p, l) for p, l in zip(percentages, labels) if p > 0]
            if not filtered_data:
                return None, "No significant toxicity detected."

            sorted_data = sorted(filtered_data, reverse=True)
            sorted_percentages, sorted_labels = zip(*sorted_data)
            colors = ["red", "orange", "skyblue"][:len(sorted_percentages)]

            ax.pie(sorted_percentages, labels=sorted_labels, autopct='%1.1f%%', colors=colors)
            ax.set_title('Toxicity Prediction')

        return fig, None

    except Exception as e:
        return None, f"An unexpected error occurred: {str(e)}"

# Gradio app layout
iface = gr.Interface(
    fn=toxicity_predictor,
    inputs=[
        gr.Textbox(lines=3, placeholder="Enter your text here..."),
        gr.Radio(choices=["Bar Chart", "Pie Chart"], label="Select Chart Type", value="Bar Chart")
    ],
    outputs=[
        gr.Plot(),  # Graph output
        gr.Textbox(label="Error Message", interactive=False)  # Error message display
    ],
    title="CleanComment",  # Title added here
)

# Launch app
if __name__ == "__main__":
    iface.launch(share=True)
