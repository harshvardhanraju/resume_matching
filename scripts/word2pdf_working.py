import os
import subprocess

def convert_word_to_pdf(word_file, pdf_file):
    subprocess.call(["unoconv", "-f", "pdf", "-o", pdf_file, word_file])

def batch_convert_folder_to_pdf(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".doc") or file.endswith(".docx"):
                word_file = os.path.join(root, file)
                pdf_file = os.path.join(output_folder, file.replace(".doc", ".pdf").replace(".docx", ".pdf"))
                print(f"Converting {word_file} to PDF...")
                convert_word_to_pdf(word_file, pdf_file)
                print(f"Conversion complete.")

# Specify the folder containing the Word documents
input_folder =  r"g:\jd_Cv\JD's-20231016T065558Z-001\JD_s"

# Specify the folder where you want to save the PDFs
output_folder = r"g:\jd_Cv\JD's-20231016T065558Z-001\JD_s"

# Convert all Word documents to PDF in the input folder
batch_convert_folder_to_pdf(input_folder, output_folder)
