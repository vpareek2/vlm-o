import streamlit as st
import torch
from PIL import Image
import os

from processor import MultiModalProcessor
from model.multimodal.multimodal_model import PaliGemmaForConditionalGeneration
from load_model import load_hf_model
from inference import test_inference

st.set_page_config(page_title="Multimodal VLM Demo", layout="wide")

@st.cache_resource
def load_model_and_processor():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    
    model_path = "./model_weights"
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    max_length = 512
    processor = MultiModalProcessor(tokenizer, num_image_tokens, image_size, max_length)
    
    return model, processor, device

st.title("Multimodal Vision-Language Model Demo")

# Load model and processor
model, processor, device = load_model_and_processor()

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Text input for prompt
prompt = st.text_input("Enter your prompt:", "Be descriptive with your response. What is happening in the photo?")

# Parameters
col1, col2, col3 = st.columns(3)
with col1:
    max_tokens = st.slider("Max tokens to generate", 10, 500, 300)
with col2:
    temperature = st.slider("Temperature", 0.0, 1.0, 0.8)
with col3:
    top_p = st.slider("Top P", 0.0, 1.0, 0.9)
do_sample = st.checkbox("Use sampling", value=False)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Response"):
        with st.spinner("Generating response..."):
            # Save the uploaded image temporarily
            temp_image_path = "temp_image.jpg"
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Capture the printed output
            import io
            import sys

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            # Run inference
            with torch.no_grad():
                test_inference(
                    model,
                    processor,
                    device,
                    prompt,
                    temp_image_path,
                    max_tokens,
                    temperature,
                    top_p,
                    do_sample,
                )
            
            # Get the captured output
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

            # Remove the temporary image
            os.remove(temp_image_path)

        st.success("Response generated!")
        st.write(output)

st.markdown("---")
st.write("This demo uses a custom Multimodal Vision-Language Model.")