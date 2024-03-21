import argparse
import logging
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def predict_result(model, image_path: str) -> dict:
    """
    Predicts the class and score for an image using the provided model.

    Parameters:
        model (tf.keras.Model): The trained model.
        image_path (str): The path to the image file.

    Returns:
        dict: A dictionary containing the predicted class and score.
    """
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x /= 255.
    x = np.expand_dims(x, axis=0)

    score = model.predict(x).ravel()

    # Determine class and score
    result = {
        'score': f"{score[0]:.2f}",
        'class': 'Negative' if score[0] < 0.5 else 'Positive'
    }

    return result


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s | %(message)s')

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="model/weights.h5", help="model path")
    parser.add_argument("-i", "--image", default="test/00007.jpg", help="image path")
    args = parser.parse_args()

    try:
        # Load model and make prediction
        model = load_model(args.model)
        result = predict_result(model, args.image)

        # Log the result
        logging.info("Prediction Result: %s", result)
    except Exception as e:
        logging.error("An error occurred: %s", e)
