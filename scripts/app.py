from transformers import AutoTokenizer, AutoModel
import fitz  # PyMuPDF
import numpy as np
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModel, AutoTokenizer


# Load a pre-trained BERT model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
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
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModel.from_pretrained("roberta-base")

    # Tokenize the text
    text1_tokens = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
    text2_tokens = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        # Get RoBERTa embeddings for the tokens
        text1_embeddings = model(**text1_tokens).last_hidden_state
        text2_embeddings = model(**text2_tokens).last_hidden_state

    # Calculate cosine similarity between embeddings
    similarity = cosine_similarity(text1_embeddings, text2_embeddings)

    return similarity[0][0]



# Paths to the PDF documents
resume_path = r"g:\jd_Cv\Profiles-20231016T064917Z-001\Profiles\Application Security Specialist\Ganesh_s Resume.pdf"
jd_pdf_folder = r"g:\jd_Cv\JD's-20231016T065558Z-001\JD_s\PDF"

# Extract text from the resume
resume_text = extract_text_from_pdf(resume_path)
# print("Resume text :", resume_text )

# Function to calculate similarity for a given JD
def calculate_jd_similarity(jd_path, resume_text):
    jd_text = extract_text_from_pdf(jd_path)
    similarity_score = improved_compute_similarity(resume_text, jd_text)
    return similarity_score

# List JDs in the JD folder
jd_files = [f for f in os.listdir(jd_pdf_folder) if f.endswith(".pdf")]

# Dictionary to store JD names and their similarity scores
jd_scores = {}

# Calculate similarity for each JD
for jd_file in jd_files:
    jd_path = os.path.join(jd_pdf_folder, jd_file)
    similarity_score = calculate_jd_similarity(jd_path, resume_text)
    jd_scores[jd_file] = similarity_score 

# Print the similarity scores for each JD
for jd_file, score in jd_scores.items():
    print(f"JD: {jd_file}, Similarity Score: {score:.2f}")


print(f"Semantic Similarity Score: {similarity_score:.2f}")
