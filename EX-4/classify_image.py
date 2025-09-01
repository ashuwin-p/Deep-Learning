import argparse

# importing pretrained models
from tensorflow.keras.applications import (
    VGG16,
    VGG19,
    InceptionV3,
    Xception,
    ResNet50,
    imagenet_utils,
)
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


# Parsing the commandline arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument(
    "-model",
    "--model",
    type=str,
    default="vgg16",
    help="name of pre-trained network to use",
)
args = vars(ap.parse_args())

MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50,
}

if args["model"] not in MODELS:
    print("Available Models : ", list(MODELS))
    raise AssertionError(
        "The --model command line argument should be a key in MODELS dictionary"
    )

if args["model"] in ("inception", "xception"):
    inputshape = (299, 299)
    preprocess = preprocess_input
else:
    inputshape = (224, 224)
    preprocess = imagenet_utils.preprocess_input

print(f"[INFO] Loading {args['model']} ...")
Network = MODELS[args["model"]]
model = Network(weights="imagenet")

print("[INFO] Loading and Preprocessing the image ...")
image = load_img(args["image"], target_size=inputshape)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess(image)

print(f"[INFO] Classifying Image with {args['model']} ...")
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)

for i, (imagenetID, label, prob) in enumerate(P[0]):
    print(f"{i+1} \t {label} \t {prob*100:.2f}%")

# Sample Image Path
# sample_images\sam1.jpeg
# sample_images\sam2.jpeg

# Sample Run
# python classify_image.py --image sample_images\sam1.jpeg --model vgg16
