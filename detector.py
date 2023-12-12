import argparse
import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json
import os

def load_model(model_path):
    with open(os.path.join(model_path, 'model.json'), 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(os.path.join(model_path, 'model.h5'))
    return model

def get_prediction(model, image_path):
    open_image = Image.open(image_path)
    resized_image = open_image.resize((256, 256))
    np_image = np.array(resized_image)
    reshaped = np.expand_dims(np_image, axis=0)

    predicted_prob = model.predict(reshaped)[0][0]

    if predicted_prob >= 0.5:
        return f"Real, Confidence: {str(predicted_prob)[:4]}"
    else:
        return f"Fake, Confidence: {str(1 - predicted_prob)[:4]}"

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detector')
    parser.add_argument('--image_path', type=str, help='Path to the image for prediction')
    parser.add_argument('--model_path', type=str, help='Path to the model files directory')
    args = parser.parse_args()

    if args.image_path and args.model_path:
        classifier = load_model(args.model_path)
        prediction = get_prediction(classifier, args.image_path)
        print(f"Prediction: {prediction}")
    else:
        print("Please provide both --image_path and --model_path arguments.")

if __name__ == "__main__":
    main()