import os
import joblib
import re
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Input directory containing resumes
input_directory = input("Enter the path of the directory containing resumes: ")


# Load the trained model and TF-IDF vectorizer
clf = joblib.load('best_clf.pkl') # must be put in the same directory where you save script.py 
tfidf_vectorizer = joblib.load('tfidf.pkl') # it loaded for TF-IDF model which are create in notebook.
# Define the text cleaning and preprocessing function
def resume_cleaning(text):
    
    # Remove HTML tags 
    cleaned_text = re.sub(r'<.*?>', ' ', text)
    
    # Remove non-english characters, punctuation,special characters, digits, continous underscores and extra whitespace
    cleaned_text = re.sub('[^a-zA-Z]', ' ', cleaned_text)
    cleaned_text = re.sub(r'[^\w\s]|_', ' ', cleaned_text)
    cleaned_text = re.sub(r'\d+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    cleaned_text=re.sub('http\S+\s', " ", cleaned_text)
    
    # Convert to lowercase
    cleaned_text = cleaned_text.lower()
    
    # Tokenize the cleaned text
    words = word_tokenize(cleaned_text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Apply stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    
    # Remove extra words
    extra_words = ['compani', 'name', 'citi', 'state', 'work', 'manag']
    final_words = [word for word in stemmed_words if word not in extra_words]
    
    final_words = ' '.join(final_words)
    
    return final_words


category_mapping={'ACCOUNTANT': 0,
    'ADVOCATE': 1,
    'AGRICULTURE': 2,
    'APPAREL': 3,
    'ARTS': 4,
    'AUTOMOBILE': 5,
    'AVIATION': 6,
    'BANKING': 7,
    'BPO': 8,
    'BUSINESS-DEVELOPMENT': 9,
    'CHEF': 10,
    'CONSTRUCTION': 11,
    'CONSULTANT': 12,
    'DESIGNER': 13,
    'DIGITAL-MEDIA': 14,
    'ENGINEERING': 15,
    'FINANCE': 16,
    'FITNESS': 17,
    'HEALTHCARE': 18,
    'HR': 19,
    'INFORMATION-TECHNOLOGY': 20,
    'PUBLIC-RELATIONS': 21,
    'SALES': 22,
    'TEACHER': 23
    }

reverse_lookup = {v: k for k, v in category_mapping.items()}

# Create an empty DataFrame to store data for the CSV file
csv_data = pd.DataFrame(columns=['finalname', 'category'])

# Process each resume file in the input directory
for resume_file in os.listdir(input_directory):
    resume_path = os.path.join(input_directory, resume_file)
    if resume_file.endswith('.pdf'):
        # Open and read the PDF file using PyPDF2 in binary mode
        with open(resume_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()
        
        # Preprocess the PDF text
        preprocessed_text = resume_cleaning(pdf_text)
        
        # Vectorize the preprocessed text
        # Vectorize the preprocessed text
        resume_tfidf = tfidf_vectorizer.transform([preprocessed_text])
        
        # Predict the category
        predicted_category = clf.predict(resume_tfidf)[0]
        
        # Convert predicted_category to a string
        predicted_category_str = reverse_lookup.get(predicted_category)
        
        # Create output subdirectory for the category if not exists
        output_category_dir = os.path.join(input_directory, predicted_category_str)
        os.makedirs(output_category_dir, exist_ok=True)
        
        # Move the PDF file to the output category folder
        new_resume_path = os.path.join(output_category_dir, resume_file)
        os.rename(resume_path, new_resume_path)
        # Add data to the DataFrame
        csv_data = csv_data.append({'finalname': resume_file, 'category': predicted_category_str}, ignore_index=True)

        print(f"Moved '{resume_file}' to '{predicted_category_str}' folder.")

# Save the DataFrame to a CSV file in the input directory
csv_file_path = os.path.join(input_directory, 'categorize_resume.csv')
csv_data.to_csv(csv_file_path, index=False)

print(f"CSV file 'categorize_resume.csv' created in '{input_directory}'.")
