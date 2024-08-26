from extract_features import extract_features
import json
from flask import current_app
from pdf_blueprint.fpdf_structure import PDF
import os

# Checks if a vlid file type is given or not
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

# Define the image processing function 
def process_image(filepath):
    image_features = extract_features(filepath)
    return json.dumps(image_features,indent=4)

def generate_pdf(text,name):

    source_path = f"reports\\report_{name}.pdf" 
    complete_source_path = f"app\\reports\\report_{name}.pdf"
    pdf = PDF()
    pdf.add_page()
    pdf.chapter_title('Analysis')
    pdf.chapter_body(text)

    directory = os.path.dirname(complete_source_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    pdf.output(complete_source_path,dest="F")
    return source_path
