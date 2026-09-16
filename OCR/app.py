import streamlit as st
import pytesseract as pyt
import pypdfium2 as pdfium
from PIL import Image


st.set_page_config('OCR', layout='wide')

# ---------------------------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------------------------

st.title('OCR - pytesserect and pypdfium')


# ---------------------------------------------------------------------------------------------------

st.write(
    "**Optical Character Recognition (OCR)** is a technology that converts images containing text—"
    "such as scanned paper documents, digital photos, PDF files, or handwritten notes—into "
    "editable, searchable, and machine-readable digital text."
)
st.write(
    "This project showcases the use of OCR using **pytesseract** and **pypdfium2**"
    " for extracting texts from image and PDF-based files."
)
st.write("---")

# ---------------------------------------------------------------------------------------------------

# taking input
st.markdown("### Select file")
file = st.file_uploader("Image and PDFs only", type=['.jpg','.jpeg','.png','.pdf'])

col1, col2 = st.columns([1,2])

if file is not None:
    file_name = file.name.lower()

    # for image input
    if file_name.endswith(('.jpg','.jpeg','.png')):             # for multiple values the '.endswith' only takes tuples
        img = Image.open(file)
        with col1:
            st.image(img)
        with col2:
            text = pyt.pytesseract.image_to_string(img)
            st.text(text)

    # for pdf input
    elif file_name.endswith('.pdf'):
        pdf = pdfium.PdfDocument(file)
        with col1:
            st.header("Images of the pages", divider='gray')
        with col2:
            st.header("Text from each pages", divider='gray')

        for i, page in enumerate(pdf):
            temp = page.get_textpage()
            text = temp.get_text_range()
            with col1:
                img = page.render(scale=2).to_pil()
                st.image(img, caption=f"Page {i+1}")
            with col2:
                st.subheader(f"Page {i+1}", divider='gray')
                st.text(text)
                st.write("\n")









