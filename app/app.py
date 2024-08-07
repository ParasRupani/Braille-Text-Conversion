from flask import Flask, request, jsonify, send_file, render_template, url_for
from braille_model import load_model, text_to_braille_image, process_image, transform
import io
import torch
import os, logging
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get environment variables
log_file_path = os.getenv('LOG_FILE_PATH')
model_path = os.getenv('MODEL_PATH')
braille_image_folder = os.getenv('BRAILLE_IMAGE_FOLDER')
uploads_folder = os.getenv('UPLOADS_FOLDER')
num_classes = int(os.getenv('NUM_CLASSES', 27))  # Default to 27 if not set
max_label_length = int(os.getenv('MAX_LABEL_LENGTH', 20))  # Default to 20 if not set
device = torch.device(os.getenv('DEVICE', 'cpu'))  # Default to 'cpu' if not set

# Ensure the directory exists
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Create the file if it doesn't exist
if not os.path.exists(log_file_path):
    with open(log_file_path, 'w') as file:
        file.write('')

# Configure logging to output to console and file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler()
                    ])

app = Flask(__name__, static_folder='static', static_url_path='/static')
model = load_model(model_path, num_classes, max_label_length).to(device)

def decode_predictions(predictions):
    """Decode the model predictions into readable text."""
    decoded = ''.join([chr(p + 97) if p != -1 else '' for p in predictions])
    decoded = decoded.replace('{', '')  # Remove any '{' characters that may appear
    return decoded.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze_text_structure', methods=['POST'])
def analyze_text_structure():
    data = request.json
    text = data.get('text', '')
    language = data.get('language', 'English')
    paragraphs = text.count('\n') + 1
    sentences = text.count('.') + text.count('!') + text.count('?')
    words = len(text.split())
    response = {
        "status": "success",
        "text_structure": {
            "paragraphs": paragraphs,
            "sentences": sentences,
            "words": words,
            "language": language
        }
    }
    return jsonify(response)

@app.route('/api/convert_to_braille', methods=['POST'])
def convert_to_braille():
    data = request.json
    text = data.get('text', '')
    logging.info(f"Received text to convert to braille: {text}")
    try:
        braille_image = text_to_braille_image(text, braille_image_folder)
        img_byte_arr = io.BytesIO()
        braille_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        logging.info("Braille image created successfully")
        return send_file(img_byte_arr, mimetype='image/png')
    except Exception as e:
        logging.error(f"Error converting text to braille: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/convert_braille_to_text', methods=['POST'])
def convert_braille_to_text():
    try:
        file = request.files['file']
        filename = os.path.join(uploads_folder, file.filename)
        file.save(filename)
        logging.info(f"File saved to {filename}")

        new_filename = os.path.join(uploads_folder, 'braille_new.png')
        process_image(filename, new_filename, braille_image_folder)
        logging.info(f"Processed image saved to {new_filename}")

        image = transform(Image.open(new_filename).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 2)
            predicted = predicted.squeeze(0).tolist()
            converted_text = decode_predictions(predicted)
        
        logging.info(f"Predicted text: {converted_text}")

        response = {
            "status": "success",
            "text": converted_text
        }
    except Exception as e:
        logging.error(f"Error converting braille to text: {e}")
        response = {
            "status": "error",
            "error": str(e)
        }
    return jsonify(response)

@app.route('/api/provide_feedback', methods=['POST'])
def provide_feedback():
    data = request.json
    braille_text = data.get('braille_text', '')
    corrections = data.get('corrections', [])
    response = {
        "status": "success",
        "message": "Feedback submitted"
    }
    return jsonify(response)

@app.route('/api/conversion_history', methods=['GET'])
def conversion_history():
    user_id = request.args.get('user_id', '')
    history = [
        {
            "text": "Example text",
            "braille_text": "⠠⠃⠗⠁⠊⠇",
            "timestamp": "2024-06-19T12:34:56"
        }
    ]
    response = {
        "status": "success",
        "history": history
    }
    return jsonify(response)

@app.route('/api/supported_languages', methods=['GET'])
def supported_languages():
    languages = ["English", "Spanish", "French", "German"]
    response = {
        "status": "success",
        "languages": languages
    }
    return jsonify(response)

@app.route('/api/conversion_status/<conversion_id>', methods=['GET'])
def conversion_status(conversion_id):
    response = {
        "status": "success",
        "conversion_status": "completed",
        "braille_text": "⠠⠃⠗⠁⠊⠇ ⠞⠑⠭⠞"
    }
    return jsonify(response)

if __name__ == '__main__':
    os.makedirs(uploads_folder, exist_ok=True)
    app.run(debug=True, port=os.getenv('PORT'))
