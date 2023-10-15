from flask import Flask, request, jsonify
from PIL import Image
from fastai.vision.all import *

import os


def load_image(file):
    img = Image.open(file)
    return img

app = Flask(__name__)

UPLOAD_FOLDER = '/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the uploaded file
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        learn = load_learner('fastai_flask_app/model/model.pkl')
        img = load_image(file)
        
        prediction, index, probabilidade = learn.predict(img)
        index_probabilidade = probabilidade[index]

        result = str(prediction)
        result_proba = index_probabilidade.tolist()

        response_data = {'prediction': result, 'certeza' : result_proba}
        print('foi?')
        return jsonify(response_data), 200
    
    except requests.exceptions.RequestException as e:
        # Log the error for debugging
        app.logger.error(f"Error connecting to external API: {e}")
        return jsonify({"error": "Failed to connect to external API"}), 500
    
    except Exception as e:
        # Log the error for debugging
        app.logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500
    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)