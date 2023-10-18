import torch
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import os
import PyPDF2
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# Define the model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Define your fine-tuning dataset and data loaders
# You'll need to implement a DataLoader for your dataset

# Fine-tuning loop
num_epochs = 3  # You can adjust the number of epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Define a mapping of JDs to class IDs
jd_to_class_mapping = {
    "Application Security Specialist": 0,
    "Application Support Engineer": 1,
    "AWS DevOps Engineer": 2,
    "Azure Active Directory": 3,
    "Oracle DBA": 4,
    "Redhat Linux": 5,
    "Vmware Admin": 6
    # Add more mappings as needed
}

class PDFDataset_old(Dataset):
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
        
        # Extract JD name from the JD file name
        jd_name = os.path.splitext(os.path.basename(jd_file_path))[0]
        
        # Tokenize the text data
        inputs = self.tokenizer(resume_text, jd_text, return_tensors='pt', padding=True, truncation=True)

        # Map JD to class ID
        jd_class = self.jd_to_class_mapping.get(jd_name, -1)  # -1 if JD not found in mapping

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

class PDFDataset(Dataset):
    def __init__(self, resume_dir, jd_dir, tokenizer, jd_to_class_mapping, num_resumes_per_class=10):
        self.resume_dir = resume_dir
        self.jd_dir = jd_dir
        self.tokenizer = tokenizer
        self.jd_to_class_mapping = jd_to_class_mapping
        self.num_resumes_per_class = num_resumes_per_class
        self.data = self.prepare_data()
        
    def prepare_data(self):
        data = []  # List to store data points

        for jd_name, class_id in self.jd_to_class_mapping.items():
            for res_num in range(self.num_resumes_per_class):
                resume_file = os.path.join(self.resume_dir, f"{jd_name}_resume_{res_num + 1}.pdf")
                jd_file = os.path.join(self.jd_dir, f"{jd_name}_jd.pdf")

                resume_text = self.extract_text_from_pdf(resume_file)
                jd_text = self.extract_text_from_pdf(jd_file)

                inputs = self.tokenizer(resume_text, jd_text, return_tensors='pt', padding=True, truncation=True)

                data.append({
                    'input_ids': inputs['input_ids'].squeeze(),
                    'attention_mask': inputs['attention_mask'].squeeze(),
                    'labels': class_id,
                })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def extract_text_from_pdf(self, file_path):
        text = ""
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfFileReader(pdf_file)
            for page_num in range(pdf_reader.numPages):
                text += pdf_reader.getPage(page_num).extractText()
        return text

class PDFDataset_new(Dataset):
    def __init__(self, resume_dir, jd_dir, tokenizer, jd_to_class_mapping):
        self.resume_dir = resume_dir
        self.jd_dir = jd_dir
        self.tokenizer = tokenizer
        self.jd_to_class_mapping = jd_to_class_mapping
        self.data = self.prepare_data()
        
    def prepare_data(self):
        data = []  # List to store data points

        for jd_name, class_id in self.jd_to_class_mapping.items():
            # Find all resume files for the current JD
            resume_files = [f for f in os.listdir(self.resume_dir) if f.startswith(f"{jd_name}_resume_")]
            jd_file = os.path.join(self.jd_dir, f"{jd_name}_jd.pdf")

            for resume_file in resume_files:
                # Construct the full file paths
                resume_file_path = os.path.join(self.resume_dir, resume_file)

                resume_text = self.extract_text_from_pdf(resume_file_path)
                jd_text = self.extract_text_from_pdf(jd_file)

                inputs = self.tokenizer(resume_text, jd_text, return_tensors='pt', padding=True, truncation=True)

                data.append({
                    'input_ids': inputs['input_ids'].squeeze(),
                    'attention_mask': inputs['attention_mask'].squeeze(),
                    'labels': class_id,
                })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def extract_text_from_pdf(self, file_path):
        text = ""
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfFileReader(pdf_file)
            for page_num in range(pdf_reader.numPages):
                text += pdf_reader.getPage(page_num).extractText()
        return text


# Define your directories containing resume and JD PDFs
resume_directory = r"g:\jd_Cv\Profiles-20231016T064917Z-001\PDFs"
jd_directory = r"g:\jd_Cv\JD's-20231016T065558Z-001\JD_s\PDF"

# Initialize the BERT tokenizer and create the dataset
# dataset = PDFDataset_old(resume_directory, jd_directory, tokenizer, jd_to_class_mapping)
# num_resumes_per_class = 10  # Number of resumes per JD class
# dataset = PDFDataset(resume_directory, jd_directory, tokenizer, jd_to_class_mapping, num_resumes_per_class)
dataset = PDFDataset_new(resume_directory, jd_directory, tokenizer, jd_to_class_mapping)

# Create a DataLoader for training
batch_size = 4  # Adjust as needed
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_dataloader  = dataloader

# Define the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)  # You can adjust the learning rate
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * num_epochs)

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()


for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}, Loss: {average_loss}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model")
