


import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image
import fitz  # PyMuPDF
import anthropic
import base64
import requests
from io import BytesIO
from datetime import datetime
import openai

load_dotenv()  # Load environment variables from .env file
openai.api_key = os.getenv('OPENAI_API_KEY')
claude = anthropic.Anthropic(api_key=os.getenv('CLOADE_API_KEY'))

def encode_image(image_path):
    """Convert image to base64 string"""
    with Image.open(image_path) as image:
        buffered = BytesIO()
        image.save(buffered, format=image.format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_text_from_image_openai(image):
    """Extract text from image using OpenAI's Vision model"""
    try:
        # Convert uploaded image to base64
        if isinstance(image, BytesIO):
            base64_image = base64.b64encode(image.getvalue()).decode('utf-8')
        else:
            base64_image = encode_image(image)

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please extract all the medical information from this image. Include all text you can see, maintaining the medical terminology and formatting. Extract it as raw text, don't try to interpret or reorganize it yet."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4000
        )
        return response.choices[0].message['content']
    except Exception as e:
        st.error(f"Error extracting text from image: {str(e)}")
        return None

def generate_pathology_report_claude_instant(text):
    """Generate a structured pathology report using Claude Instant"""
    try:
        response = claude.messages.create(
            model="claude-3-sonnet-20240229",  # Using the lower cost Claude Instant model
            max_tokens=4000,
            temperature=0.3,
            messages=[
                {
                    "role": "assistant",
                    "content": f"""As an expert pathologist doctor, please create a comprehensive pathology report from the following medical information according to pathology report writing rules. 
                    Follow these guidelines:

                    1. Key Sections Required:
                    - Patient Demographics
                    - Clinical History
                    - Specimen Details
                    - Gross Description 
                    - Microscopic Findings
                    - Diagnosis
                    - Comments

                    2. Requirements:
                    - Use standard medical terminology
                    - Include measurements where available
                    - Note any abnormal findings
                    - Add relevant clinical recommendations

                    Raw Medical Information:
                    {text}

                    Please format this into a clear, professional pathology report ."""
                }
            ]
        )
        return response.content[0].text
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        return None

# Streamlit App Layout
st.title("Medical Report Extractor and Formatter")
st.write("Upload an image of a medical report for professional pathology report formatting.")

# Upload image
uploaded_file = st.file_uploader("Upload a medical report image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Medical Report", use_column_width=True)
    
    # Extract text using OpenAI Vision
    with st.spinner("Extracting text from image using OpenAI Vision..."):
        extracted_text = extract_text_from_image_openai(uploaded_file)
        
        if extracted_text:
            st.subheader("Extracted Medical Information")
            st.write(extracted_text)
            
            # Allow editing of extracted text
            edited_text = st.text_area(
                "Review and edit the extracted text if needed:",
                value=extracted_text,
                height=300
            )
            
            if st.button("Generate Pathology Report"):
                with st.spinner("Generating professional pathology report..."):
                    formatted_report = generate_pathology_report_claude_instant(edited_text)
                    
                    if formatted_report:
                        # Add report metadata
                        report_header = f"""
                        # PATHOLOGY REPORT
                        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        Report ID: PATH-{datetime.now().strftime('%Y%m%d-%H%M%S')}
                        _______________________________________________

                        """
                        
                        # Display the formatted report
                        st.subheader("Final Pathology Report")
                        st.markdown(report_header + formatted_report)
                        
                        # Add download button for the report
                        st.download_button(
                            label="Download Report as Text",
                            data=report_header + formatted_report,
                            file_name=f"pathology_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
