import gradio as gr
import torch
from PIL import Image
from processor import MultiModalProcessor
from inference import test_inference
from load_model import load_hf_model

# Load model and processor
MODEL_PATH = "./weights"
TOKENIZER_PATH = "./weights"
device = "cuda" if torch.cuda.is_available() else "cpu"

model, tokenizer = load_hf_model(MODEL_PATH, TOKENIZER_PATH, device)
model = model.eval()

num_image_tokens = model.config.vision_config.num_image_tokens
image_size = model.config.vision_config.image_size
max_length = 512
processor = MultiModalProcessor(tokenizer, num_image_tokens, image_size, max_length)

def generate_caption(image, prompt, max_tokens=300, temperature=0.8, top_p=0.9, do_sample=False):
    # Save the input image temporarily
    temp_image_path = "temp_image.jpg"
    Image.fromarray(image).save(temp_image_path)
    
    # Use the existing test_inference function
    result = []
    def capture_print(text):
        result.append(text)

    import builtins
    original_print = builtins.print
    builtins.print = capture_print

    test_inference(
        model,
        processor,
        device,
        prompt,
        temp_image_path,
        max_tokens,
        temperature,
        top_p,
        do_sample
    )

    builtins.print = original_print

    # Return the captured output
    return "".join(result)

# Define Gradio demo
with gr.Blocks(title="VLM-o Demo", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown(
        """
        # Interact with VLM-o
        This demo uses the VLM-o model to let you chat with an image.
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("Generate Response"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="numpy", label="Upload Image")
                    prompt_input = gr.Textbox(label="Prompt", placeholder="Enter question here")
                
                with gr.Column(scale=1):
                    with gr.Group():
                        max_tokens_input = gr.Slider(1, 500, value=300, step=1, label="Max Tokens")
                        temperature_input = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                        top_p_input = gr.Slider(0.1, 1.0, value=0.9, step=0.1, label="Top P")
                        do_sample_input = gr.Checkbox(label="Do Sample")
                    
                    generate_button = gr.Button("Generate Response")
            
            output = gr.Textbox(label="Response: ", lines=5)
        
        with gr.TabItem("About"):
            gr.Markdown(
                """
                ## How to use:
                1. Upload an image in the 'Generate Response' tab.
                2. Enter a prompt to guide the model response.
                3. Adjust the hyperparameters if desired.
                4. Click 'Generate Response' to see the results.

                ## Model Details:
                - Model: VLM-o
                - Type: Multimodal (Text + Image) input -> Text output
                - Task: Image Captioning, Image analysis, Question answering.
                """
            )

    generate_button.click(
        generate_caption,
        inputs=[image_input, prompt_input, max_tokens_input, temperature_input, top_p_input, do_sample_input],
        outputs=output
    )

# Launch the demo
if __name__ == "__main__":
    demo.launch()