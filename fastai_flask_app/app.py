from flask import Flask, request, jsonify
from fastai.data.all import *
from fastai.vision.all import *
from pathlib import *
from flask import Flask, render_template, request, jsonify
import numpy



app = Flask(__name__)

# Load FastAI model
model_path = '/home/pc/Documents/fastai_flask_app/model/model.pkl'
learn = load_learner(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Process the image using the FastAI model
    img = load_image(file)
    
    prediction, index, probabilidade = learn.predict(img)
    index_probabilidade = probabilidade[index]

    result = str(prediction)
    result_proba = index_probabilidade.tolist()
    # Create JSON response
    response_data = {'prediction': result, 'certeza' : result_proba}
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
