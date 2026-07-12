[English](README.md) | [日本語](README.ja.md)
# Handwriting Analysis and Psychological Profiling System

## Overview
This application leverages computer vision and Large Language Models (LLMs) to analyze handwriting characteristics from uploaded images. By extracting features such as slant, pressure, and spacing, the system utilizes the Mistral AI API to generate a comprehensive psychological report (Graphology) in PDF format.

## NOTE : This system can analyse only Latin-alphabet writing systems, such as English, Spanish, French, etc.

## Technical Architecture
- **Backend**: Flask (Python)
- **AI Engine**: Mistral AI (mistral-large-latest)
- **PDF Generation**: FPDF
- **Image Processing**: Custom processing logic (referenced in utils)

## Core Features
- **Automated Feature Extraction**: Processes uploaded images to identify nine specific handwriting traits including loop variety, garland styles, and baseline angles.
- **LLM-Powered Analysis**: Uses a structured prompt engineering approach via Mistral AI to interpret raw data into natural language psychological profiles.
- **Conflict Resolution**: The AI agent is instructed to identify and explain contradictory traits (e.g., high coordination vs. temporary emotional instability).
- **Personalized Reporting**: Generates a professional PDF report tailored to the subject's name provided during submission.

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Mistral AI API Key

### Installation
1. Clone the repository to your local machine.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration
Create a `.env` file in the root directory or update `constants.py` with your API credentials. Ensure the `uploads/` and `app/reports/` directories exist before running the application.

## Usage
1. Start the Flask server:
   ```bash
   python run.py
   ```
2. Navigate to `http://127.0.0.1:5000` in your web browser.
3. Enter the subject's name and upload a clear image of handwriting.
4. Download the generated PDF psychological report.

## Disclaimer
Graphology is a pseudoscientific study of handwriting. This tool is intended for educational and entertainment purposes and should not be used as a substitute for professional psychological evaluation.
```