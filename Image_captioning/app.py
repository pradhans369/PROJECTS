import torch
from PIL import Image
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(
    page_title="Image Captioning System",
    layout="wide"
)


st.markdown(
    """
    <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
        <div style='display: inline-flex; align-items: center; background-color: rgba(255, 255, 255, 0.05); 
                    padding: 6px 14px; border-radius: 25px; border: 1px solid rgba(255, 255, 255, 0.1);'>
            <b style='margin-right: 12px; font-size: 15px;'>Credits :</b>
            <a href='https://github.com/pradhans369' target='_blank'>
                <img src='https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white' 
                     style='border-radius: 6px; margin-right: 8px;'>
            </a>
            <a href='https://www.linkedin.com/in/pradhans369/' target='_blank'>
                <img src='https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white' 
                     style='border-radius: 6px;'>
            </a>
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)


# Custom styling
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
        margin-bottom: 2rem;
    }
    .caption-box {
        background-color: #f8f9fa;
        border-left: 4px solid #0d6efd;
        padding: 1.25rem;
        border-radius: 4px;
        font-size: 1.2rem;
        font-weight: 500;
        color: #212529;
        margin-top: 1rem;
    }
    .caption-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)



@st.cache_resource(show_spinner=False)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    model.eval()
    return processor, model, device

with st.spinner("Loading model..."):
    processor, model, device = load_model()

# Header
st.markdown('<div class="main-header">Image Captioning System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'An automated visual description engine powered by the Salesforce BLIP (Bootstrapping Language-Image Pre-training) '
    'multimodal architecture. Upload an image to generate a syntactically grounded, context-aware description.'
    '</div>',
    unsafe_allow_html=True
)

with st.expander("Technical Specifications"):
    st.markdown(
        """
        - **Model Architecture:** Salesforce BLIP (`blip-image-captioning-base`)
        - **Visual Backbone:** Vision Transformer (ViT) with spatial patch representation
        - **Text Generation:** Causal Transformer Decoder with cross-attention
        - **Inference Strategy:** Deterministic Beam Search (Beam Width: 5, Repetition Penalty: 1.2)
        """
    )



# Two-column layout
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
    else:
        st.info("Upload an image (PNG, JPG, JPEG, WEBP) to begin.")

with col_right:
    if uploaded_file is not None:
        # Button placed on the right side of the photo
        generate_clicked = st.button("Generate Caption", type="primary", use_container_width=True)

        # Track previous caption in session state to persist across renders
        if "last_file" not in st.session_state or st.session_state["last_file"] != uploaded_file.name:
            st.session_state["last_file"] = uploaded_file.name
            st.session_state["caption"] = None

        if generate_clicked:
            with st.spinner("Analyzing image..."):
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    output_tokens = model.generate(
                        **inputs,
                        num_beams=5,
                        max_new_tokens=30,
                        min_length=8,
                        repetition_penalty=1.2
                    )

                st.session_state["caption"] = processor.decode(output_tokens[0], skip_special_tokens=True).strip()

        if st.session_state.get("caption"):
            st.markdown(
                f"""
                <div class="caption-box">
                    <div class="caption-title">Generated Caption</div>
                    {st.session_state['caption']}
                </div>
                """,
                unsafe_allow_html=True
            )
