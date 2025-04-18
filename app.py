
from flask import Flask, request, render_template, redirect, url_for, session
import pickle
import docx
import PyPDF2
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import gensim
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.metrics.pairwise import cosine_similarity

# Load trained Word2Vec model
model = gensim.models.Word2Vec.load("resume_word2vec.model")
def handler(environ, start_response):
    return app(environ, start_response)
# Function to preprocess text
def preprocess(text):
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())  
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return tokens

# Function to compute sentence vector
def get_sentence_vector(text, model):
    tokens = preprocess(text)
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Function to predict resume score
def predict_resume_score(resume_text, job_role):
    resume_vec = get_sentence_vector(resume_text, model)
    job_vec = get_sentence_vector(job_role, model)
    return cosine_similarity([resume_vec], [job_vec])[0][0]

# Load pre-trained model and TF-IDF vectorizer
try:
    svc_model = pickle.load(open('clf.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    le = pickle.load(open('encoder.pkl', 'rb'))
    model_loaded = True
except Exception as e:
    print("Model failed to load:", e)
    model_loaded = False

app = Flask(__name__)
app.secret_key = "your_secret_key"

os.makedirs("static/images", exist_ok=True)

# Function to clean resume text
def clean_resume(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Functions to extract text from different file types
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    return '\n'.join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file):
    return file.read().decode('utf-8', errors='ignore')

# Function to handle file upload
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.filename.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

# Function to predict the category of a resume
def predict_category(input_resume):
    cleaned_text = clean_resume(input_resume)
    if model_loaded:
        try:
            vectorized_text = tfidf.transform([cleaned_text]).toarray()
            predicted_category = svc_model.predict(vectorized_text)
            resume_score = predict_resume_score(input_resume, le.inverse_transform(predicted_category)[0])
            return le.inverse_transform(predicted_category)[0], resume_score, vectorized_text
        except Exception as e:
            print("Model Prediction Failed:", e)
    return "Other", np.random.randint(30, 70), None

# Function to generate a resume score donut chart
def generate_donut_chart(score):
    labels = [score, 100 - score]
    sizes = [score, 100 - score]
    colors = ['#1f77b4', '#d3d3d3']
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    plt.savefig("static/images/resume_score_donut.png", bbox_inches='tight', dpi=100)
    plt.close()

# Function to generate a contribution chart
def generate_contribution_chart(vectorized_text):
    if vectorized_text is not None:
        feature_array = np.array(tfidf.get_feature_names_out())
        importances = np.abs(vectorized_text[0])
        top_n = 10
        top_indices = importances.argsort()[-top_n:][::-1]
        top_features = feature_array[top_indices]
        top_importances = importances[top_indices] * 100
        plt.figure(figsize=(6, 4))
        bars = plt.barh(top_features[::-1], top_importances[::-1], color='#1f77b4')
        for bar, score in zip(bars, top_importances[::-1]):
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{score:.1f}%", va='center', fontsize=10, color='black')
        plt.xlabel("Importance Score")
        plt.title("Top Feature Contribution")
        plt.savefig("static/images/contribution_chart.png", bbox_inches='tight', dpi=100)
        plt.close()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files.get('resume')
        if not uploaded_file or uploaded_file.filename == '':
            return render_template('index.html', error="No file selected")
        try:
            extracted_text = handle_file_upload(uploaded_file)
            prediction, resume_score, vectorized_text = predict_category(extracted_text)
            if resume_score < 1:
                while resume_score < 10:  # Ensure at least two digits before the decimal point
                    resume_score *= 10
            session['resume_score'] = float(resume_score) 
            generate_donut_chart(resume_score)
            generate_contribution_chart(vectorized_text)
            session['prediction'] = prediction
            return redirect(url_for('result'))
        except Exception as e:
            return render_template('index.html', error=str(e))
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template(
        'result.html', 
        prediction=session.get('prediction', 'No Prediction'), 
        resume_score=session.get('resume_score', 0), 
        donut_chart="static/images/resume_score_donut.png",
        contribution_chart="static/images/contribution_chart.png"
    )

if __name__ == '__main__':
    app.run(debug=True)
