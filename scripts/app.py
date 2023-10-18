"""
Author : Harshvardhan Raju G
Deevia Software
Date : 17/10/23
"""

from transformers import AutoTokenizer, AutoModel
# import fitz  # PyMuPDF
import PyPDF2
import numpy as np
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModel, AutoTokenizer


# Load a pre-trained BERT model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

tokenizer_roberta = AutoTokenizer.from_pretrained("roberta-base")
model_roberta = AutoModel.from_pretrained("roberta-base")

# # Function to extract text from a PDF file
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     doc = fitz.open(pdf_path)
#     for page_num in range(doc.page_count):
#         page = doc.load_page(page_num)
#         text += page.get_text()
#     return text

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(file_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)
        for page_num in range(pdf_reader.numPages):
            text += pdf_reader.getPage(page_num).extractText()
    return text
# Function to compute similarity between two texts using BERT embeddings
def compute_similarity(text1, text2):
    # Tokenize the text
    text1_tokens = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
    text2_tokens = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)

    # Get BERT embeddings for the tokens
    with np.errstate(divide='ignore', invalid='ignore'):
        text1_embeddings = model(**text1_tokens).last_hidden_state.mean(dim=1)
        text2_embeddings = model(**text2_tokens).last_hidden_state.mean(dim=1)

    # Convert PyTorch tensors to NumPy arrays
    text1_embeddings = text1_embeddings.detach().numpy()
    text2_embeddings = text2_embeddings.detach().numpy()

    # Calculate cosine similarity between embeddings
    similarity = cosine_similarity(text1_embeddings, text2_embeddings)

    return similarity[0][0]


def improved_compute_similarity(text1, text2):


    # Tokenize the text
    text1_tokens = tokenizer_roberta(text1, return_tensors="pt", padding=True, truncation=True)
    text2_tokens = tokenizer_roberta(text2, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        # Get RoBERTa embeddings for the tokens
        text1_embeddings = model_roberta(**text1_tokens).last_hidden_state
        text2_embeddings = model_roberta(**text2_tokens).last_hidden_state

    # Calculate cosine similarity between embeddings
    similarity = cosine_similarity(text1_embeddings, text2_embeddings)

    return similarity[0][0]



# Paths to the PDF documents
jd_path = r"g:\jd_Cv\JD's-20231016T065558Z-001\JD_s\PDF\Application Security Specialist.pdf"
resume_pdf_folder = r"g:\jd_Cv\Profiles-20231016T064917Z-001\test"

# Extract text from the resume
jd_text = extract_text_from_pdf(jd_path)
# print("Resume text :", resume_text )

# Function to calculate similarity for a given JD
def calculate_jd_similarity( resume_text, jd_text):
    similarity_score = compute_similarity(resume_text, jd_text)
    return similarity_score

# List JDs in the JD folder
resume_files = [f for f in os.listdir(resume_pdf_folder) if f.endswith(".pdf")]

# Dictionary to store JD names and their similarity scores
jd_scores = {}

# Calculate similarity for each resume_files
for resume_file in resume_files:
    resume_path = os.path.join(resume_pdf_folder, resume_file)
    resume_text = extract_text_from_pdf(resume_path)
    similarity_score = calculate_jd_similarity(resume_text, jd_text)
    jd_scores[resume_path] = similarity_score 

# Print the similarity scores for each JD
for resume_file, score in jd_scores.items():
    print(f"JD: {resume_file}, Similarity Score: {score:.2f}")


print(f"Semantic Similarity Score: {similarity_score:.2f}")
