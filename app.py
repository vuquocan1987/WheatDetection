from flask import Flask
from PIL import Image, ImageDraw
from flask import render_template,url_for
from flask import request, redirect
from werkzeug.utils import secure_filename
from datetime import datetime
from keras.models import Sequential
from tensorflow.keras.models import load_model
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.measure import label, regionprops
from keras import backend as K
import numpy as np
import tensorflow as tf
import os
import re

app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = os.path.join('static','upload')
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG","JPG","PNG","GIF"]
app.config["MAX_CONTENT_LENGTH"] = 50*1024*1024
IMG_CHANNELS = 3
MODEL_NAME = 'wheat_detection.h5'
IMAGE_SIZE = 512
def castF(x):
    return K.cast(x, K.floatx())

def castB(x):
    return K.cast(x, bool)

def iou_loss_core(true,pred):  #this can be used as a loss if you make it negative
    intersection = true * pred
    notTrue = 1 - true
    union = true + (notTrue * pred)

    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())

def competitionMetric2(true, pred): #any shape can go - can't be a loss function

    tresholds = [0.55 + (i * 0.05)  for i in range(5)]

    #flattened images (batch, pixels)
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = castF(K.greater(pred, 0.5))

    #total white pixels - (batch,)
    trueSum = K.sum(true, axis=-1)
    predSum = K.sum(pred, axis=-1)

    #has mask or not per image - (batch,)
    true1 = castF(K.greater(trueSum, 1))    
    pred1 = castF(K.greater(predSum, 1))

    #to get images that have mask in both true and pred
    truePositiveMask = castB(true1 * pred1)

    #separating only the possible true positives to check iou
    testTrue = tf.boolean_mask(true, truePositiveMask)
    testPred = tf.boolean_mask(pred, truePositiveMask)

    #getting iou and threshold comparisons
    iou = iou_loss_core(testTrue,testPred) 
    truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]

    #mean of thressholds for true positives and total sum
    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)
    truePositives = K.sum(truePositives)

    #to get images that don't have mask in both true and pred
    trueNegatives = (1-true1) * (1 - pred1) # = 1 -true1 - pred1 + true1*pred1
    trueNegatives = K.sum(trueNegatives) 

    return (truePositives + trueNegatives) / castF(K.shape(true)[0])

model = load_model(MODEL_NAME,custom_objects =   {'competitionMetric2':competitionMetric2})
model.summary()
def allowed_image(filename):
    if "." not in filename:
        return False
    ext = filename.rsplit(".",1)[1]
    return ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]
def allowed_image_filesize(filesize):
    return int(filesize) <= app.config["MAX_CONTENT_LENGTH"] 
@app.route("/")
def home():
    return render_template("public/upload_image.html")

@app.route("/hello/<name>")
def hello_there(name):
    now = datetime.now()
    formatted_now = now.strftime("%A, %d %B, %Y at %X")

    # Filter the name argument to letters only using regular expressions. URL arguments
    # can contain arbitrary text, so we restrict to safe characters only.
    match_object = re.match("[a-zA-Z]+", name)

    if match_object:
        clean_name = match_object.group(0)
    else:
        clean_name = "Friend"

    content = "Hello there, " + clean_name + "! It's " + formatted_now
    return content
def load_and_preprocess_image(path):
    img = imread(path)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMAGE_SIZE, IMAGE_SIZE), mode='constant', preserve_range=True)
    X_train = []
    X_train.append(img)
    return X_train

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    
    return image

def detect_wheat(path):
    processed_img = load_and_preprocess_image(path)
    result = model.predict([processed_img])
    print(result)
    alpha = result[0]
    alpha = resize(alpha,(1024,1024))
    alpha = np.squeeze(alpha)
    return alpha
def draw_alpha(path,alpha):
    alpha = np.where(alpha>0.43,.5,1.0)
    alpha_mask_img = Image.fromarray((alpha*255).astype(np.uint8))
    image = Image.open(path)
    white_img = Image.new('RGB',image.size,(255,255,255))
    compsited_img = Image.composite(image,white_img,alpha_mask_img)
    
    dir_file = os.path.split(path)
    alpha_image_path = os.path.join(dir_file[0],"a"+dir_file[1])
    print(alpha_image_path)
    compsited_img.save(alpha_image_path)
    
    return alpha_image_path
def test_model():
    model = load_model(MODEL_NAME,custom_objects =   {'competitionMetric2':competitionMetric2})
    model.summary()


@app.route("/files",methods=["GET"])
def get_image():
    names = os.listdir(app.config["IMAGE_UPLOADS"])
    image_paths = [os.path.join(app.config["IMAGE_UPLOADS"],name) for name in names]

    return render_template('public/all_image.html', image_paths=image_paths)

@app.route("/classification",methods = ["GET","POST"])
def upload_image():
#   test_model()
#    model.summary()
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            if image.filename == "":
                print("No filename")
                return redirect(request.url)
            if "filesize" in request.cookies:
                if not allowed_image_filesize(request.cookies["filesize"]):
                    print("Filesize exceeded maximum limit")
                    return redirect(request.url)
            if allowed_image(image.filename):
                filename = secure_filename(image.filename)
                path = os.path.join(app.config["IMAGE_UPLOADS"],filename)
                image.save(os.path.join(app.config["IMAGE_UPLOADS"],filename))
                alpha = detect_wheat(path)
                image_path = draw_alpha(path,alpha)
                
                print("Image saved")
            
            return render_template("public/show_result.html",image_path = image_path)
    return render_template("public/upload_image.html")

