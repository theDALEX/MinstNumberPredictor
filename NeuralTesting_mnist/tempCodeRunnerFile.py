from flask import Flask, request, render_template
from tensorflow import keras
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model('mnist_model.h5')

def prepare_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img = np.array(img)
    img = img / 255.0  # Normalize
    img = img.reshape(1, 28 * 28)  # Flatten
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(file)
            img = prepare_image(img)
            prediction = model.predict(img)
            digit = np.argmax(prediction)
            
            # Extract the weights and biases
            weights = model.get_weights()
            weights_str = "<br>".join([f"Layer {i//2+1} {'weights' if i % 2 == 0 else 'biases'}:<br>{layer}" for i, layer in enumerate(weights)])
            
            return f'''
            <!doctype html>
            <title>Prediction Result</title>
            <h1>The predicted digit is: {digit}</h1>
            <h2>Model Weights and Biases:</h2>
            <pre>{weights_str}</pre>
            '''
    return '''
    <!doctype html>
    <title>Upload an Image</title>
    <h1>Upload a Handwritten Digit Image</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''

if __name__ == "__main__":
    app.run()
