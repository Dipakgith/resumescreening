import streamlit as st
import pickle
import re
import nltk
import PyPDF2
from io import BytesIO

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load models
def load_models():
    try:
        clf = pickle.load(open('clf.pkl', 'rb'))
        tfidf_vectorizer = pickle.load(open('tfidf.pkl', 'rb'))
        st.success("Models loaded successfully.")
        return clf, tfidf_vectorizer
    except FileNotFoundError as e:
        st.error(f"Model file not found. Please check the file path. Error: {e}")
        return None, None

# Clean resume text
def clean_resume(resume_text):
    clean_text = re.sub(r'http\S+\s*', ' ', resume_text)
    clean_text = re.sub(r'RT|cc', ' ', clean_text)
    clean_text = re.sub(r'#\S+', '', clean_text)
    clean_text = re.sub(r'@\S+', ' ', clean_text)
    clean_text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text.strip()

# Extract text from uploaded file
def extract_text_from_uploaded_file(uploaded_file):
    try:
        if uploaded_file.type == 'text/plain':
            return uploaded_file.read().decode('utf-8')
        elif uploaded_file.type == 'application/pdf':
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            pdf_text = ''
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()
            return pdf_text
        else:
            st.error("Unsupported file type.")
            return ""
    except Exception as e:
        st.error(f"Failed to extract text: {e}")
        return ""

# Main function to run the Streamlit app
def main():
    st.title("Tech Developer Resume Screening App")

    # Load models
    clf, tfidf_vectorizer = load_models()

    if clf is None or tfidf_vectorizer is None:
        st.error("Model loading failed. Please upload valid model files.")
        return

    # File upload section
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        resume_text = extract_text_from_uploaded_file(uploaded_file)
        cleaned_resume = clean_resume(resume_text)

        # Debugging: Check if resume text is cleaned and available
        if cleaned_resume:
            st.text("Cleaned Resume Text:")
            st.write(cleaned_resume)

            try:
                # Vectorize the resume text
                input_features = tfidf_vectorizer.transform([cleaned_resume])

                # Make prediction
                prediction_id = clf.predict(input_features)[0]

                # Map category ID to tech developer roles only
                tech_category_mapping = {
                    0: "Full Stack Developer",
                    15: "Java Developer",
                    20: "Python Developer",
                    3: "Blockchain",
                    8: "DevOps Engineer",
                    9: "DotNet Developer",
                    7: "Database",
                    6: "Data Science",
                    17: "Network Security Engineer",
                    21: "SAP Developer",
                    13: "Hadoop",
                    10: "ETL Developer",
                }

                # Check if predicted category is a tech developer role
                category_name = tech_category_mapping.get(prediction_id, "Unknown")

                if category_name == "Unknown":
                    st.warning("The uploaded resume does not match a tech developer role.")
                else:
                    st.success(f"Predicted Tech Developer Role: {category_name}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.write("No text extracted from the resume.")

# Python main
if __name__ == "__main__":
    main()
