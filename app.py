import streamlit as st
from hybrid_predict import hybrid_prediction
import PyPDF2
import docx


st.set_page_config(page_title="Fake Job Scam Detector")

st.title("🚨 Fake Job / Internship Scam Detection System")

st.write("Check whether a job or internship posting is real or fake.")

# FILE TEXT EXTRACTION FUNCTIONS
def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text


# INPUT OPTIONS
option = st.radio("Choose Input Type:", ["Text Input", "Upload File"])

job_text = ""

# TEXT INPUT
if option == "Text Input":
    job_text = st.text_area("Paste Job Description", height=200)

# FILE INPUT
elif option == "Upload File":
    uploaded_file = st.file_uploader("Upload file", type=["txt", "pdf", "docx"])

    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1]

        if file_type == "txt":
            job_text = uploaded_file.read().decode("utf-8")

        elif file_type == "pdf":
            job_text = extract_text_from_pdf(uploaded_file)

        elif file_type == "docx":
            job_text = extract_text_from_docx(uploaded_file)

        st.write("📄 File Content Preview:")
        st.text(job_text[:500])

# BUTTON
if st.button("Check Job"):

    if job_text.strip() == "":
        st.warning("Please enter or upload a job description.")
    else:
        label, confidence, reasons = hybrid_prediction(job_text)

        # RESULT
        st.subheader("Result:")
        st.success(label)

        # CONFIDENCE
        st.subheader("Confidence:")
        st.progress(int(confidence * 100))
        st.write(f"{round(confidence*100,2)} %")

        # REASONS
        st.subheader("Reasoning:")
        if reasons:
            for r in reasons:
                st.write("•", r)
        else:
            st.write("No strong scam indicators detected")

        # MODEL INFO
        st.info("Model: Ensemble ML + Explainable Hybrid System")