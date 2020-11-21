import flask
from flask import render_template, Flask, request, Response, jsonify
import numpy as np
from copy import deepcopy
from PIL import Image
from flask_cors import CORS, cross_origin

import cv2
import base64

#####

from PIL import Image
import argparse
import os , io , sys
import colorsys

from flask import Flask, request, Response

#####

simType = 0

#ArgParse Function
def parse_args():
	parser = argparse.ArgumentParser(
		description = 'Simulate and Correct Images for Color Blindness')
	parser.add_argument(
		'-input', type=str, help = 'Path to input image')
	parser.add_argument(
		'-output', type=str, help = 'Path for output image')
	parser.add_argument('-sp', action='store_true', help = 'Simulate Protanopia (Common Red-Green  Blindness)')
	parser.add_argument('-sd', action='store_true', help = 'Simulate Deutranopia (Rare Red-Green Blindness)')
	parser.add_argument('-st', action='store_true', help = 'Simulate Tritanopia (Blue-Yellow Color Blindness)')
	parser.add_argument('-correct', action='store_true', help = 'Correct Image for Protanopia')
	args = parser.parse_args()
	return args

#Restructuring laterally inverted image
def normalise(editablePhoto,sizeX,sizeY):
	NormalPhoto =  np.zeros((sizeX,sizeY,3),'float')
	x=sizeX-1
	y=sizeY
	for i in range(0,sizeX):
		for j in range(0,sizeY):
			for k in range(0,3):
				NormalPhoto[x,j,k]=editablePhoto[i,j,k]
		x=x-1

	return NormalPhoto

#Matrix Multiplication Block (Common for all operations, just varying matrix)
def getImageArray(respectiveArray, editablePhoto, sizeX, sizeY):
	for i in range(0,sizeX):
		for j in range(0,sizeY):
			currMatrix = np.array((0,0,0),dtype=float)
			for k in range(0,3):
				currMatrix[k]=editablePhoto[i,j,k]
			lmsImage = np.dot(respectiveArray,currMatrix)
			for k in range(0,3):
				editablePhoto[i,j,k]=lmsImage[k]
	return editablePhoto

#Converting RGB to LMS
def convertToLMS(im,sizeX,sizeY):
	photo = im.load()
	editablePhoto = np.zeros((sizeX,sizeY,3),'float')
	for i in range(0,sizeX):
		for j in range(0,sizeY):
			for k in range(0,3):
				editablePhoto[i,j,k] = photo[i,j][k]
				editablePhoto[i,j,k] = ((editablePhoto[i,j,k])/255)

	lmsConvert = np.array([[17.8824,43.5161,4.11935],[3.45565,27.1554,3.86714],[0.0299566,0.184309,1.46709]])
	editablePhoto = getImageArray(lmsConvert, editablePhoto, sizeX, sizeY)

	NormalPhoto =  normalise(editablePhoto,sizeX,sizeY)
	return NormalPhoto

#Simulating Protanopia
def ConvertToProtanopes(editablePhoto,sizeX,sizeY):
	protanopeConvert = np.array([[0,2.02344,-2.52581],[0,1,0],[0,0,1]])
	editablePhoto = getImageArray(protanopeConvert, editablePhoto, sizeX, sizeY)
	NormalPhoto = normalise(editablePhoto, sizeX, sizeY)
	return NormalPhoto

#Simulating Deutranopia
def ConvertToDeuteranopes(editablePhoto,sizeX,sizeY):
	DeuteranopesConvert = np.array([[1,0,0],[0.494207,0,1.24827],[0,0,1]])
	editablePhoto = getImageArray(DeuteranopesConvert, editablePhoto, sizeX, sizeY)
	NormalPhoto = normalise(editablePhoto, sizeX, sizeY)
	return NormalPhoto

#Simulating Tritanopia
def ConvertToTritanope(editablePhoto,sizeX,sizeY):
	TritanopeConvert = np.array([[1,0,0],[0,1,0],[-0.395913,0.801109,0]])
	editablePhoto = getImageArray(TritanopeConvert, editablePhoto, sizeX, sizeY)
	NormalPhoto = normalise(editablePhoto, sizeX, sizeY)
	return NormalPhoto

#Converting LMS to RGB
def convertToRGB(editablePhoto,sizeX,sizeY):
	rgb2lms = np.array([[17.8824,43.5161,4.11935],[3.45565,27.1554,3.86714],[0.0299566,0.184309,1.46709]])
	RGBConvert = np.linalg.inv(rgb2lms)
	editablePhoto = getImageArray(RGBConvert, editablePhoto, sizeX, sizeY)
	for i in range(0,sizeX):
		for j in range(0,sizeY):
			for k in range(0,3):
				editablePhoto[i,j,k]=((editablePhoto[i,j,k]))*255

	NormalPhoto = normalise(editablePhoto, sizeX, sizeY)
	return NormalPhoto

#Converting Processed Array to Image
def arrayToImage(editablePhoto,sizeX,sizeY,saveAs):
	rgbArray = np.zeros((sizeX,sizeY,3),'uint8')
	for i in range(0,sizeX):
		for j in range(0,sizeY):
			for k in range(0,3):
				rgbArray[i,j,k] = editablePhoto[i,j,k]
	img = Image.fromarray(rgbArray)
	img.save(saveAs)

#Correcting the image using HSV Shifting Algorithm
def correct(inputIm, sizeX,sizeY,saveAs):

	photo = inputIm.load()
	editablePhoto = np.zeros((sizeX,sizeY,3),'float')
	hsvArray=np.zeros((sizeX,sizeY,3),'float')
				
	for i in range(0,sizeX):
		for j in range(0,sizeY):
			for k in range(0,3):
				editablePhoto[i,j,k] = photo[i,j][k]
				editablePhoto[i,j,k] = ((editablePhoto[i,j,k])/255)
			rNew=editablePhoto[i,j,0]
			gNew=editablePhoto[i,j,1]
			bNew=editablePhoto[i,j,2]

			tempArray=np.zeros((3),'float')

			for k in range(0,3):
				hsvArray[i,j,k]=colorsys.rgb_to_hsv(editablePhoto[i,j,0],editablePhoto[i,j,1],editablePhoto[i,j,2])[k]

			
			greenRatio = (hsvArray[i,j,0] - (60/360))/gNew
			blueRange = greenRatio*bNew
			hsvArray[i,j,0] = 0.5 + blueRange

			tempArray=np.zeros((3),'float')
			for k in range(0,3):
				tempArray[k]=hsvArray[i,j,k]
			tempArray.tolist()
			tempArray = (colorsys.hsv_to_rgb(tempArray[0],tempArray[1],tempArray[2]))

			for k in range(0,3):
				editablePhoto[i,j, k] = tempArray[k]*255

	NormalPhoto = normalise(editablePhoto, sizeX, sizeY)
	arrayToImage(NormalPhoto,sizeX,sizeY,saveAs)

#Main function decides which function to call based on user choice in ArgParse
def main(input, output, simType):
	inputIm = Image.open(input)
	sizeX = inputIm.size[0]
	sizeY = inputIm.size[1]

	if simType != 4:
		lmsPhoto = convertToLMS(inputIm,sizeX,sizeY)
		if simType == 1:
			simPhoto = ConvertToProtanopes(lmsPhoto,sizeX,sizeY)
		elif simType == 2:
			simPhoto = ConvertToDeuteranopes(lmsPhoto,sizeX,sizeY)
		elif simType == 3:
			simPhoto = ConvertToTritanope(lmsPhoto,sizeX,sizeY)

		rgbPhoto = convertToRGB(simPhoto,sizeX,sizeY)
		arrayToImage(rgbPhoto,sizeX,sizeY,output)
	elif simType == 4:
		correct(inputIm,sizeX,sizeY,output)

#Fetching ArgParse Data

# Initialize the app
app = flask.Flask(__name__)
# CORS(app)
# CORS(app, support_credentials=True)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/")
@cross_origin(origin="*")
def home_page():
    return 'home page !'

@app.route("/conversion/<conversion_type>", methods=["POST"])
@cross_origin(origin="*")
def process_image(conversion_type):
    if request.method == 'POST':
        
        file = request.data
        npimg = np.fromstring(file, np.uint8)
        img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
        print('Received Image')

        cv2.imwrite('upload.jpg', img) 
        print('Saved Image')

        ### Do Image processing
        
        if conversion_type == 'Protanope':
            main('upload.jpg', 'download.jpg', 1)
        elif conversion_type == 'Deuteranope':
            main('upload.jpg', 'download.jpg', 2)
        elif conversion_type == 'Tritanope':
            main('upload.jpg', 'download.jpg', 3)
        else:
            return jsonify({'Invalid Type of conversion'})

        #cv2.imwrite('download.jpg', img) 
        ###

        img = Image.open('download.jpg')
        img = img.rotate(-90	)
        
        print('Opened Download image in PIL')
        
        rawBytes = io.BytesIO()
        img.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())
        return str(img_base64)
        # return jsonify({'file':str(img_base64)})
        # return img_base64
        return jsonify("data:image/png;base64,"+img_base64)


#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0', port=8000, debug=True)