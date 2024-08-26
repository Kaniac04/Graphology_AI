from flask import Flask
import os

def create_app():
    app = Flask(__name__)

    # Set the upload folder and allowed extensions
    app.config['UPLOAD_FOLDER'] = 'uploads/'
    app.config['ALLOWED_EXTENSIONS'] = ['png', 'jpg', 'jpeg', 'gif']
    app.config['REPORTS_FOLDER'] = "app//reports"
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)
    # Register routes
    from .routes import main
    app.register_blueprint(main)

    return app
