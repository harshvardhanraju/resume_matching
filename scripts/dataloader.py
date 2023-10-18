import os
import PyPDF2
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# Define a mapping of JDs to class IDs
jd_to_class_mapping = {
    "Software Engineer": 0,
    "Data Scientist": 1,
    "Product Manager": 2,
    # Add more mappings as needed
}

class PDFDataset(Dataset):
    def __init__(self, resume_dir, jd_dir, tokenizer, jd_to_class_mapping):
        self.resume_dir = resume_dir
        self.jd_dir = jd_dir
        self.tokenizer = tokenizer
        self.jd_to_class_mapping = jd_to_class_mapping
        self.pdf_files = os.listdir(resume_dir)

    def __len__(self):
        return len(self.pdf_files)

    def __getitem__(self, idx):
        # Read resume and JD text from PDF files
        resume_file_path = os.path.join(self.resume_dir, self.pdf_files[idx])
        jd_file_path = os.path.join(self.jd_dir, self.pdf_files[idx])

        resume_text = self.extract_text_from_pdf(resume_file_path)
        jd_text = self.extract_text_from_pdf(jd_file_path)

        # Tokenize the text data
        inputs = self.tokenizer(resume_text, jd_text, return_tensors='pt', padding=True, truncation=True)

        # Map JD to class ID
        jd_class = self.jd_to_class_mapping.get(jd_text, -1)  # -1 if JD not found in mapping

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': jd_class,
        }

    def extract_text_from_pdf(self, file_path):
        text = ""
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfFileReader(pdf_file)
            for page_num in range(pdf_reader.numPages):
                text += pdf_reader.getPage(page_num).extractText()
        return text

# Define your directories containing resume and JD PDFs
resume_directory = "path/to/resume_pdfs"
jd_directory = "path/to/jd_pdfs"

# Initialize the BERT tokenizer and create the dataset
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = PDFDataset(resume_directory, jd_directory, tokenizer, jd_to_class_mapping)

# Create a DataLoader for training
batch_size = 4  # Adjust as needed
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)