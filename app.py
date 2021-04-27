import os

from werkzeug.utils import secure_filename
from flask import Flask, render_template, Response, url_for, request
from model.model_factory import ModelFactory

app = Flask(__name__)
model_factory = ModelFactory()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', "GET"])
def prediction():
    model = model_factory.get_resnet34()

    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        pred_class = model.predict(file_path)
        return pred_class

    return None


if __name__ == '__main__':
    app.run(host='localhost', debug=True)
