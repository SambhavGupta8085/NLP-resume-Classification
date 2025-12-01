import streamlit as st
import pickle
import docx
import PyPDF2
import re

try:
    svc_model = pickle.load(open('clf.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    le = pickle.load(open('encoder.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files missing. Ensure clf.pkl, tfidf.pkl, encoder.pkl are in the same directory.")
    st.stop()

def cleanResume(txt):
    txt = re.sub('http\S+\s', ' ', txt)
    txt = re.sub('RT|cc', ' ', txt)
    txt = re.sub('#\S+\s', ' ', txt)
    txt = re.sub('@\S+', ' ', txt)
    txt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    txt = re.sub('\s+', ' ', txt)
    return txt

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc_file = docx.Document(file)
    text = ''
    for paragraph in doc_file.paragraphs:
        text += paragraph.text + '\n'
    return text

def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        return file.read().decode('latin-1')

def handle_file_upload(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif ext == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif ext == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type.")

def pred(input_resume):
    cleaned = cleanResume(input_resume)
    vector = tfidf.transform([cleaned]).toarray()
    result = svc_model.predict(vector)
    return le.inverse_transform(result)[0]

def main():
    st.set_page_config(page_title="Resume Classification", layout="wide")

    st.title("Intelligent Resume Classification System")
    st.markdown("---")

    st.header("About the Project")
    st.info(
        "This system uses Natural Language Processing and Machine Learning to automatically categorize resumes "
        "into job domains. It was created to simplify large-scale resume screening."
    )
    st.markdown("---")

    st.header("Upload Resume")
    uploaded_file = st.file_uploader("Select a File (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Process Status")
            try:
                resume_text = handle_file_upload(uploaded_file)
                st.success("Text extracted successfully.")
                category = pred(resume_text)
                st.success("Prediction completed.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.stop()

        with col2:
            st.subheader("Predicted Category")
            st.success(f"{category}")

        st.markdown("---")
        with st.expander("Extracted Resume Text"):
            st.text_area("Text Content", resume_text, height=200, disabled=True)

if __name__ == "__main__":
    main()
