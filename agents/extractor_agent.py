import os
import fitz
import docx
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from langgraph_workflows.graph import graph
def ocr_pdf(filepath):
    pages = convert_from_path(filepath)
    text = ""
    for page in pages:
        img = page.convert("RGB") if page.mode != "RGB" else page
        text += pytesseract.image_to_string(img)
    return text

def extract_text_from_pdf(filepath):
    text=""
    doc=fitz.open(filepath)
    for page in doc:
        text+=page.get_text()
    return text

def ocr_image(filepath):
    try:
        image = Image.open(filepath)
        image = image.convert("RGB")
        text = pytesseract.image_to_string(image)
        return text if text.strip() else ""
    except Exception as e:
        return ""

def is_scanned_pdf(filepath):
    doc=fitz.open(filepath)
    for page in doc:
        text=page.get_text().strip()
        if len(text)>20:
            return False
        
    return True

def extract_docx(filepath):
    doc=docx.Document(filepath)
    text=[]
    for para in doc.paragraphs:
        text.append(para.text)
    return "\n".join(text)     

def extract_file(filepath):
    ext=filepath.split(".")[-1].lower()
    if ext=="pdf":
        if is_scanned_pdf(filepath):
            return ocr_pdf(filepath)
        else:
            return extract_text_from_pdf(filepath)
    elif ext=="docx":
        return extract_docx(filepath)
    elif ext in ['png','jpg','jpeg']:
        return ocr_image(filepath)
    elif ext=="txt" or ext=="json":
        with open(filepath,"r") as f:
            return f.read()
    else:
        return "Unsupported format"    

def run_extraction_pipeline(patient_id,patient_files):
    print("[extractor] extractor agent invoked")
    extracted_data={}
    for file_type,file_path in patient_files.items():
        text=extract_file(file_path)
        extracted_data[file_type]=text

    config = {
        "configurable":{
            "thread_id": patient_id
        }
    }
    graph.invoke({
        "patient_id":patient_id,
        "raw_data":{
            "bill":extracted_data['bills'],
            "doctor":extracted_data['doctor_reports'],
            "lab":extracted_data['lab_reports']
        }
    },config)

