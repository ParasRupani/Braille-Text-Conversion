from flask import Flask, request, jsonify, render_template
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Configuration for the email server
EMAIL_HOST = os.getenv('EMAIL_HOST')
EMAIL_PORT = os.getenv('EMAIL_PORT')
EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD')
EMAIL_HISTORY_FILE = os.getenv('EMAIL_HISTORY_FILE')

@app.route('/')
def index():
    return render_template('braille_email_composer.html')

@app.route('/api/send_email', methods=['POST'])
def send_email():
    data = request.json
    recipient = data.get('recipient')
    subject = data.get('subject')
    body = data.get('body')

    if not recipient or not subject or not body:
        return jsonify({"status": "error", "error": "All fields are required."})

    # Save email to history
    save_email_to_history(recipient, subject, body)

    try:
        # Create the email message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_HOST_USER
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Send the email
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)
        server.sendmail(EMAIL_HOST_USER, recipient, msg.as_string())
        server.quit()

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})

@app.route('/api/email_history', methods=['GET'])
def email_history():
    if not os.path.exists(EMAIL_HISTORY_FILE):
        return jsonify({"status": "success", "emails": []})
    with open(EMAIL_HISTORY_FILE, 'r') as file:
        emails = file.readlines()
    email_list = [email.strip().split('|') for email in emails]
    return jsonify({"status": "success", "emails": email_list})

def save_email_to_history(recipient, subject, body):
    with open(EMAIL_HISTORY_FILE, 'a') as file:
        file.write(f"{recipient}|{subject}|{body}\n")

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv('EMAIL_APP_PORT'))