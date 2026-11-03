import time
import torch
from PIL import Image
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(
    page_title="Image Captioning System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for a clean, professional aesthetic
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .caption-box {
        background-color: #f8f9fa;
        border-left: 4px solid #0d6efd;
        padding: 1rem 1.25rem;
        border-radius: 4px;
        font-size: 1.1rem;
        font-weight: 500;
        color: #212529;
        margin-top: 1rem;
    }
    .metadata-label {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metadata-value {
        font-size: 1rem;
        font-weight: 600;
        color: #212529;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource(show_spinner=False)
def load_captioning_model(model_identifier: str = "Salesforce/blip-image-captioning-base"):
    """
    Loads and caches the BLIP vision-language processor and model.
    Defaults to 'base' for optimal memory usage and inference latency on standard server instances.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(model_identifier)
    model = BlipForConditionalGeneration.from_pretrained(model_identifier).to(device)
    model.eval()
    return processor, model, device

# Sidebar: Configuration Parameters
st.sidebar.header("Inference Configuration")

model_choice = st.sidebar.selectbox(
    "Model Architecture",
    options=[
        "Salesforce/blip-image-captioning-base",
        "Salesforce/blip-image-captioning-large"
    ],
    index=0,
    help="The base model is recommended for deployment on shared CPU hosts due to memory constraints."
)

decoding_strategy = st.sidebar.radio(
    "Decoding Strategy",
    options=["Beam Search", "Nucleus Sampling"],
    index=0,
    help="Beam Search produces deterministic, factual captions. Nucleus Sampling yields more varied phrasing."
)

st.sidebar.subheader("Hyperparameters")

max_new_tokens = st.sidebar.slider(
    "Maximum Sequence Length",
    min_value=10,
    max_value=60,
    value=30,
    step=5
)

min_length = st.sidebar.slider(
    "Minimum Sequence Length",
    min_value=5,
    max_value=25,
    value=8,
    step=1
)

repetition_penalty = st.sidebar.slider(
    "Repetition Penalty",
    min_value=1.0,
    max_value=2.0,
    value=1.2,
    step=0.05
)

if decoding_strategy == "Beam Search":
    num_beams = st.sidebar.slider("Beam Width", min_value=1, max_value=10, value=5, step=1)
    temperature = None
    top_p = None
else:
    num_beams = 1
    temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.05)
    top_p = st.sidebar.slider("Top-p (Cumulative Probability)", min_value=0.5, max_value=1.0, value=0.9, step=0.05)

# Initialize Model
with st.spinner("Initializing vision-language pipeline..."):
    processor, model, device = load_captioning_model(model_choice)

# Main Application Layout
st.markdown('<div class="main-header">Image Captioning System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated visual description generator powered by Transformer-based Vision-Language Models.</div>', unsafe_allow_html=True)

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("Input Image")
    uploaded_file = st.file_uploader(
        "Upload an image file (PNG, JPG, JPEG, WEBP)",
        type=["png", "jpg", "jpeg", "webp"],
        help="Select a local image for descriptive analysis."
    )

    optional_prompt = st.text_input(
        "Optional Conditioning Prompt",
        value="",
        placeholder="e.g., a photo of",
        help="Provide contextual guidance to condition the initial sequence generation."
    )

    generate_button = st.button("Generate Caption", type="primary", use_container_width=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col_output:
    st.subheader("Analysis & Generation")

    if uploaded_file is None:
        st.info("Upload an image on the left panel to begin caption generation.")
    elif generate_button:
        with st.spinner("Processing image and generating description..."):
            start_time = time.time()

            # Preprocess inputs
            if optional_prompt.strip():
                inputs = processor(images=image, text=optional_prompt.strip(), return_tensors="pt").to(device)
            else:
                inputs = processor(images=image, return_tensors="pt").to(device)

            # Generate caption
            with torch.no_grad():
                if decoding_strategy == "Beam Search":
                    generation_output = model.generate(
                        **inputs,
                        num_beams=num_beams,
                        max_new_tokens=max_new_tokens,
                        min_length=min_length,
                        repetition_penalty=repetition_penalty
                    )
                else:
                    generation_output = model.generate(
                        **inputs,
                        do_sample=True,
                        top_p=top_p,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        min_length=min_length,
                        repetition_penalty=repetition_penalty
                    )

            elapsed_time = time.time() - start_time
            caption_text = processor.decode(generation_output[0], skip_special_tokens=True).strip()

        # Display result
        st.markdown('<div class="metadata-label">Generated Description</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="caption-box">{caption_text}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Execution Metrics")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.markdown('<div class="metadata-label">Compute Device</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metadata-value">{device.upper()}</div>', unsafe_allow_html=True)
        with metric_col2:
            st.markdown('<div class="metadata-label">Latency</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metadata-value">{elapsed_time:.2f} s</div>', unsafe_allow_html=True)
        with metric_col3:
            st.markdown('<div class="metadata-label">Token Count</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metadata-value">{len(generation_output[0])}</div>', unsafe_allow_html=True)
