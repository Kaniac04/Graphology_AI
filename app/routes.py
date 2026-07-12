from flask import Blueprint, request, render_template,current_app,send_file, after_this_request
from werkzeug.utils import secure_filename
import os
from app.utils import allowed_file, process_image,generate_pdf
from agent import get_traits_from_AI
main = Blueprint('main', __name__)

@main.route('/', methods=["GET",'POST'])
def upload_image():
    if request.method == "GET" :
        return render_template('index.html')
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        name = request.form['name']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                handwriting_traits = process_image(filepath)
                if os.path.exists(filepath):
                    os.remove(filepath)

                result = get_traits_from_AI(handwriting_traits)
                if not result:
                    return "Error: AI Failed to analyse traits", 500
                pdf_path = generate_pdf(result,name)

                @after_this_request
                def remove_report(response):
                    try:
                        if os.path.exists(pdf_path):
                            os.remove(pdf_path)
                    except Exception as error:
                        current_app.logger.error(f"Error removing cleanup file: {error}")
                    return response

                return send_file(pdf_path,as_attachment=True,download_name="report.pdf")
            
            except Exception as e:
                if os.path.exists(filepath) : os.remove(filepath)
                return "An error occured during processing. Please ensure the image is clear.", 500
            
