from flask import Blueprint, request, render_template,current_app,send_file
from werkzeug.utils import secure_filename
import os
from app.utils import allowed_file, process_image,generate_pdf
from agent import get_traits_from_AI
main = Blueprint('main', __name__)

@main.route('/', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        name = request.form['name']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            handwriting_traits = process_image(filepath)
            result = get_traits_from_AI(handwriting_traits)
            pdf_path = generate_pdf(result,name)
            return send_file(pdf_path,as_attachment=True,download_name="report.pdf")
            
