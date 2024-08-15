from flask import Flask, request, jsonify, render_template
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os
from dotenv import load_dotenv
import logging
from PIL import Image, ImageDraw, ImageFont
import io

app = Flask(__name__)
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for the email server
EMAIL_HOST = os.getenv('EMAIL_HOST')
EMAIL_PORT = os.getenv('EMAIL_PORT')
EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD')
EMAIL_HISTORY_FILE = "email.txt"

@app.before_request
def initialize():
    if not hasattr(app, 'has_run'):
        app.has_run = True
        logger.info("Application has started")

@app.route('/')
def index():
    logger.info("Rendering the Braille Email Composer HTML page")
    return render_template('braille_email_composer.html')

@app.route('/api/compose_email_braille', methods=['POST'])
def compose_email_braille():
    logger.info("API call /api/compose_email_braille started")
    data = request.json
    braille_text = data.get('braille_text')
    recipient = data.get('recipient')
    subject = data.get('subject')

    if not braille_text or not recipient or not subject:
        logger.error("Required fields are missing in /api/compose_email_braille")
        return jsonify({"status": "error", "error": "All fields are required."}), 400

    # Convert Braille text to an image
    braille_image = create_braille_image(braille_text)

    # Save the email details to a text file
    save_email_to_file(recipient, subject, braille_text)

    logger.info("API call /api/compose_email_braille completed")
    return jsonify({
        "status": "success",
        "message": "Email composed and converted to Braille image."
    })

@app.route('/api/send_braille_email', methods=['POST'])
def send_braille_email():
    logger.info("API call /api/send_braille_email started")
    data = request.json
    braille_text = data.get('braille_text')
    recipient = data.get('recipient')
    subject = data.get('subject')

    if not braille_text or not recipient or not subject:
        logger.error("Required fields are missing in /api/send_braille_email")
        return jsonify({"status": "error", "error": "All fields are required."}), 400

    # Convert Braille text to an image
    braille_image = create_braille_image(braille_text)

    # Send the email with the Braille image as an attachment
    response = send_email_with_image(recipient, subject, braille_image)
    
    # Save the email details to a text file
    save_email_to_file(recipient, subject, braille_text)

    logger.info("API call /api/send_braille_email completed")
    return response

def create_braille_image(braille_text):
    # Create an image with Braille text
    img = Image.new('RGB', (800, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    d.text((10, 10), braille_text, font=font, fill=(0, 0, 0))

    # Convert the image to binary data
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return img_byte_arr

def send_email_with_image(recipient, subject, braille_image):
    try:
        # Create the email message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_HOST_USER
        msg['To'] = recipient
        msg['Subject'] = subject

        # Attach the Braille image
        image = MIMEImage(braille_image, name="braille.png")
        msg.attach(image)

        # Send the email
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)
        server.sendmail(EMAIL_HOST_USER, recipient, msg.as_string())
        server.quit()

        logger.info("Email sent successfully to %s", recipient)
        return jsonify({"status": "success", "message": "Email sent successfully."})
    except Exception as e:
        logger.error("Failed to send email: %s", str(e))
        return jsonify({"status": "error", "error": str(e)}), 500

def save_email_to_file(recipient, subject, body):
    try:
        # Open the file in append mode ('a')
        with open(EMAIL_HISTORY_FILE, 'a') as f:
            f.write(f"{recipient}|{subject}|{body}\n")
        logger.info("Email saved to file %s", EMAIL_HISTORY_FILE)
    except Exception as e:
        logger.error("Failed to save email to file: %s", str(e))

@app.route('/api/list_emails', methods=['GET'])
def list_emails():
    try:
        if not os.path.exists(EMAIL_HISTORY_FILE):
            return jsonify({"status": "success", "emails": []})
        
        with open(EMAIL_HISTORY_FILE, 'r') as file:
            emails = [line.strip() for line in file.readlines()]
        return jsonify({"status": "success", "emails": emails})
    except Exception as e:
        logger.error("Failed to read email list: %s", str(e))
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/api/get_email_details', methods=['GET'])
def get_email_details():
    logger.info("API call /api/get_email_details started")
    email_id = request.args.get('email_id')
    user_id = request.args.get('user_id')

    if not email_id or not user_id:
        logger.error("Email ID or User ID is missing in /api/get_email_details")
        return jsonify({"status": "error", "error": "Email ID and User ID are required."}), 400

    email = get_email_details_by_id(email_id, user_id)

    logger.info("API call /api/get_email_details completed")
    return jsonify({"status": "success", "email": email})

@app.route('/api/mark_email_read', methods=['POST'])
def mark_email_read():
    logger.info("API call /api/mark_email_read started")
    data = request.json
    email_id = data.get('email_id')
    user_id = data.get('user_id')

    if not email_id or not user_id:
        logger.error("Email ID or User ID is missing in /api/mark_email_read")
        return jsonify({"status": "error", "error": "Email ID and User ID are required."}), 400

    mark_email_as_read(email_id, user_id)

    logger.info("API call /api/mark_email_read completed")
    return jsonify({"status": "success", "message": "Email marked as read."})

@app.route('/api/delete_email', methods=['DELETE'])
def delete_email():
    logger.info("API call /api/delete_email started")
    data = request.json
    email_id = data.get('email_id')
    user_id = data.get('user_id')

    if not email_id or not user_id:
        logger.error("Email ID or User ID is missing in /api/delete_email")
        return jsonify({"status": "error", "error": "Email ID and User ID are required."}), 400

    delete_email_by_id(email_id, user_id)

    logger.info("API call /api/delete_email completed")
    return jsonify({"status": "success", "message": "Email deleted successfully."})

if __name__ == '__main__':
    logger.info("Starting the Flask application")
    app.run(debug=True, port=os.getenv('EMAIL_APP_PORT'))
