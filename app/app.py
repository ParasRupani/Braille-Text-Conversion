from flask import Flask, request, jsonify, send_file, render_template, url_for
from braille_model import load_model, text_to_braille_image, process_image, transform, create_pipeline, BrailleDataset, collate_fn
import io
import torch
import os, logging
from PIL import Image
from dotenv import load_dotenv
from torch.utils.data import DataLoader

# Load environment variables from .env file
load_dotenv()

# Get environment variables
log_file_path = os.getenv('LOG_FILE_PATH')
model_path = os.getenv('MODEL_PATH')
braille_image_folder = os.getenv('BRAILLE_IMAGE_FOLDER')
uploads_folder = os.getenv('UPLOADS_FOLDER')
num_classes = int(os.getenv('NUM_CLASSES', 27))  # Default to 27 if not set
max_label_length = int(os.getenv('MAX_LABEL_LENGTH', 20))  # Default to 20 if not set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

logging.info("Starting the application.")

app = Flask(__name__, static_folder='static', static_url_path='/static')
model = load_model(model_path, num_classes, max_label_length).to(device)
pipeline = create_pipeline(model)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze_text_structure', methods=['POST'])
def analyze_text_structure():
    logging.info("API call: /api/analyze_text_structure started.")
    data = request.json
    text = data.get('text', '')
    language = data.get('language', 'English')

    # Analyze text structure
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
    logging.info("API call: /api/analyze_text_structure completed.")
    return jsonify(response)

@app.route('/api/convert_to_braille', methods=['POST'])
def convert_to_braille():
    logging.info("API call: /api/convert_to_braille started.")
    try:
        file = request.files['file']
        filename = os.path.join(uploads_folder, file.filename)
        file.save(filename)
        logging.info(f"File saved to {filename}")

        new_filename = os.path.join(uploads_folder, 'english_braille_new.png')
        process_image(filename, new_filename, braille_image_folder)  # Assuming similar preprocessing as `convert_braille_to_text`
        logging.info(f"Processed image saved to {new_filename}")

        # Load the English CNN model instead of Braille CNN model
        english_model_path = os.path.join(os.path.dirname(__file__), 'english_cnn.pth')
        english_model = load_model(english_model_path, num_classes, max_label_length).to(device)
        english_pipeline = create_pipeline(english_model)

        braille_dataset = BrailleDataset(img_paths=[new_filename], transform=transform)
        data_loader = DataLoader(braille_dataset, batch_size=1, collate_fn=lambda x: x)

        decoded_predictions = english_pipeline.fit_transform(data_loader)

        logging.info(f"Predicted text: {decoded_predictions[0]}")

        response = {
            "status": "success",
            "text": decoded_predictions[0]
        }
    except Exception as e:
        logging.error(f"Error converting English text to braille: {e}")
        response = {
            "status": "error",
            "error": str(e)
        }
    logging.info("API call: /api/convert_to_braille completed.")
    return jsonify(response)


@app.route('/api/convert_braille_to_text', methods=['POST'])
def convert_braille_to_text():
    logging.info("API call: /api/convert_braille_to_text started.")
    try:
        file = request.files['file']
        filename = os.path.join(uploads_folder, file.filename)
        file.save(filename)
        logging.info(f"File saved to {filename}")

        new_filename = os.path.join(uploads_folder, 'braille_new.png')
        process_image(filename, new_filename, braille_image_folder)
        logging.info(f"Processed image saved to {new_filename}")

        braille_dataset = BrailleDataset(img_paths=[new_filename], transform=transform)
        data_loader = DataLoader(braille_dataset, batch_size=1, collate_fn=lambda x: x)

        decoded_predictions = pipeline.fit_transform(data_loader)
        
        logging.info(f"Predicted text: {decoded_predictions[0]}")

        response = {
            "status": "success",
            "text": decoded_predictions[0]
        }
    except Exception as e:
        logging.error(f"Error converting braille to text: {e}")
        response = {
            "status": "error",
            "error": str(e)
        }
    logging.info("API call: /api/convert_braille_to_text completed.")
    return jsonify(response)

@app.route('/api/provide_feedback', methods=['POST'])
def provide_feedback():
    logging.info("API call: /api/provide_feedback started.")
    data = request.json
    braille_text = data.get('braille_text', '')
    corrections = data.get('corrections', [])

    # Process feedback
    response = {
        "status": "success",
        "message": "Feedback submitted"
    }
    logging.info("API call: /api/provide_feedback completed.")
    return jsonify(response)

@app.route('/api/conversion_history', methods=['GET'])
def conversion_history():
    logging.info("API call: /api/conversion_history started.")
    user_id = request.args.get('user_id', '')
    
    # Retrieve conversion history (example response)
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
    logging.info("API call: /api/conversion_history completed.")
    return jsonify(response)

@app.route('/api/supported_languages', methods=['GET'])
def supported_languages():
    logging.info("API call: /api/supported_languages started.")
    languages = ["English", "Spanish", "French", "German"]

    response = {
        "status": "success",
        "languages": languages
    }
    logging.info("API call: /api/supported_languages completed.")
    return jsonify(response)

@app.route('/api/upload_braille_image', methods=['POST'])
def upload_braille_image():
    logging.info("API call: /api/upload_braille_image started.")
    try:
        file = request.files['file']
        filename = os.path.join(uploads_folder, file.filename)
        file.save(filename)
        logging.info(f"Braille image uploaded to {filename}")

        response = {
            "status": "success",
            "message": "Image uploaded successfully."
        }
    except Exception as e:
        logging.error(f"Error uploading braille image: {e}")
        response = {
            "status": "error",
            "message": str(e)
        }
    logging.info("API call: /api/upload_braille_image completed.")
    return jsonify(response)

@app.route('/api/delete_conversion_history', methods=['DELETE'])
def delete_conversion_history():
    logging.info("API call: /api/delete_conversion_history started.")
    user_id = request.json.get('user_id', '')

    # Process deletion (example response)
    response = {
        "status": "success",
        "message": "Conversion history deleted."
    }
    logging.info("API call: /api/delete_conversion_history completed.")
    return jsonify(response)

@app.route('/api/get_braille_image', methods=['GET'])
def get_braille_image():
    logging.info("API call: /api/get_braille_image started.")
    image_id = request.args.get('image_id', '')

    try:
        # Example to send back a specific image based on image_id
        img_path = os.path.join(uploads_folder, f'{image_id}.png')
        return send_file(img_path, mimetype='image/png')
    except Exception as e:
        logging.error(f"Error retrieving braille image: {e}")
        response = {
            "status": "error",
            "message": str(e)
        }
    logging.info("API call: /api/get_braille_image completed.")
    return jsonify(response)

@app.route('/api/get_braille_text_by_image', methods=['POST'])
def get_braille_text_by_image():
    logging.info("API call: /api/get_braille_text_by_image started.")
    try:
        file = request.files['file']
        filename = os.path.join(uploads_folder, file.filename)
        file.save(filename)
        logging.info(f"File saved to {filename}")

        # Process and return standard text (example response)
        response = {
            "status": "success",
            "standard_text": "Example text from Braille image."
        }
    except Exception as e:
        logging.error(f"Error converting braille image to text: {e}")
        response = {
            "status": "error",
            "message": str(e)
        }
    logging.info("API call: /api/get_braille_text_by_image completed.")
    return jsonify(response)

if __name__ == '__main__':
    os.makedirs(uploads_folder, exist_ok=True)
    logging.info("Starting Flask server.")
    app.run(debug=True, port=os.getenv('PORT'))
    logging.info("Flask server stopped.")
