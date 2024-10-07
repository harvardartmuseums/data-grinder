import os
import json
import argparse
import datetime
import time
import requests
import imagehash
from flask import Flask, request
from dotenv import  load_dotenv
from PIL import Image
from parsers import azureoai, clarifai, vision, imagga, iiif, mcsvision, colors, aws, awsanthropic

load_dotenv()

temp_folder = os.path.dirname(os.path.realpath(__file__)) + "/temp"
if not os.path.exists(temp_folder): 
	os.mkdir(temp_folder)

app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
	return {"status": "ok"}

@app.route("/extract", methods=['GET'])
def extract():
	response = {"status": "missing parameters url, services"}

	url = request.args.get('url')
	services = request.args.get('services')
	if services is not None:
		services = services.split(',')
	
	if url and services: 
		response = process_image(url, services)

	return response

def main(url, services):
	image_info = process_image(url, services)
	print(json.dumps(image_info))

## HELPER FUNCTIONS ##
def get_image_id(URL):
	r = requests.get(URL, timeout=21)
	if (r.status_code == 200) and (r.headers["Content-Type"] == 'image/jpeg'):
		status = "ok"
		id = r.url[37:]
		
	else:
		status = "bad"
		id = ""

	return (status, id)

def download_image(URL):
	r = requests.get(URL, timeout=21)
	if r.status_code == 200:
		status = "ok"
		# path = config.TEMPORARY_FILE_DIR + "/temp.jpg"
		path = os.path.dirname(os.path.realpath(__file__)) + "/temp/temp.jpg"
		
		with open(path, 'wb') as out:
			for chunk in r.iter_content(chunk_size=128):
				out.write(chunk)
	else:
		status = "bad"
		path = ""

	return (status, path)

def process_image(URL, services):
	image = {
		"url": URL,
		"lastupdated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		"runtime": 0
	}

	start = time.time()

	# get IDS ID
	(status, id) = get_image_id(URL)

	image["drsstatus"] = status

	if status == "ok": 
		image_url = iiif.IIIFImage.get_full_image_url(id)

		image["idsid"] = id
		image["iiifbaseuri"] = iiif.IIIFImage.get_base_uri(id)

		# Download the image
		(status, image_local_path) = download_image(image_url)

		# Gather and store image metadata
		im=Image.open(image_local_path)
		size = im.size # (width,height) tuple
		image["width"] = size[0]
		image["height"] = size[1]

		iiif_image_info = iiif.IIIFImage().fetch(image["iiifbaseuri"])
		image["widthFull"] = iiif_image_info["width"]
		image["heightFull"] = iiif_image_info["height"]

		imageScaleFactor = iiif_image_info["width"]/image["width"]

		annotationFragmentFullImage = "xywh=0,0," + str(int(image["width"])) + "," + str(int(image["height"]))


		# scalefactor is useful when converting annotation coordinates between different image sizes
		image["scalefactor"] = imageScaleFactor

		# Run through image hashing algorithms
		if "hash" in services: 
			hashes = {}
			i = Image.open(image_local_path)
			
			hash = imagehash.average_hash(i)
			hashes["average"] = str(hash)

			hash = imagehash.colorhash(i)
			hashes["color"] = str(hash)
			
			hash = imagehash.phash(i)
			hashes["perceptual"] = str(hash)
			
			hash = imagehash.dhash(i)
			hashes["difference"] = str(hash)

			hash = imagehash.whash(i)
			hashes["wavelet"] = str(hash)

			image["hashes"] = hashes

		# Run through HAM color service
		if "color" in services: 
			result = colors.Colors().fetch_colors(image_url)
			image["colors"] = result["colors"]
		
		# Run through Clarifai
		if "clarifai" in services:
			result = clarifai.Clarifai().fetch(image_url, id)
			
			# Process concepts
			if "data" in result["outputs"][0]:
				for concept in result["outputs"][0]["data"]["concepts"]:
					concept["annotationFragment"] = annotationFragmentFullImage

			image["clarifai"] = result

		# Run through Microsoft Cognitive Services
		if "mcs" in services: 
			image["microsoftvision"] = {}

			result = mcsvision.MCSVision().fetch_description(image_local_path)
			# Process description->captions
			if "description" in result:
				for caption in result["description"]["captions"]:
					caption["annotationFragment"] = annotationFragmentFullImage
			image["microsoftvision"]["describe"] = result

			result = mcsvision.MCSVision().fetch_analyze(image_local_path)
			# Process description->captions
			if "description" in result:
				for caption in result["description"]["captions"]:
					caption["annotationFragment"] = annotationFragmentFullImage
			
			# Process categories
			if "categories" in result:
				for category in result["categories"]:
					category["annotationFragment"] = annotationFragmentFullImage

			# Process tags
			if "tags" in result:
				for tag in result["tags"]:
					tag["annotationFragment"] = annotationFragmentFullImage

			# convert faces to IIIF image API URLs
			if "faces" in result:
				for index in range(len(result["faces"])):
					face = result["faces"][index]

					xOffset = face["faceRectangle"]["left"]*imageScaleFactor
					yOffset = face["faceRectangle"]["top"]*imageScaleFactor
					width = face["faceRectangle"]["width"]*imageScaleFactor
					height = face["faceRectangle"]["height"]*imageScaleFactor

					iiifFaceImageURL = image["iiifbaseuri"] + "/" + str(int(xOffset)) + "," + str(int(yOffset)) + "," + str(int(width)) + "," + str(int(height)) + "/full/0/native.jpg"
					face["iiifFaceImageURL"] = iiifFaceImageURL
					face["annotationFragment"] = "xywh=" + str(int(xOffset)) + "," + str(int(yOffset)) + "," + str(int(width)) + "," + str(int(height))

					result["faces"][index] = face

			image["microsoftvision"]["analyze"] = result

		# Run through Google Vision
		if "gv" in services: 
			result = vision.Vision().fetch(image_local_path)

			# Process labels/tags
			if "labelAnnotations" in result["responses"][0]:
				for label in result["responses"][0]["labelAnnotations"]:
					label["annotationFragment"] = annotationFragmentFullImage

			# convert bounding polys for face annotations to IIIF image API URL to fetch the specific region
			if "faceAnnotations" in result["responses"][0]:
				for index in range(len(result["responses"][0]["faceAnnotations"])):
					face = result["responses"][0]["faceAnnotations"][index]
					bounding = face["boundingPoly"]["vertices"]

					# sometimes the boundingPoly is missing X or Y values if it's too close to the edge of the image
					# fill in the missing coordinate using fdBoundingPoly
					fdBounding = face["fdBoundingPoly"]["vertices"]

					if "x" not in fdBounding[0]: fdBounding[0]["x"] = 0
					if "y" not in fdBounding[0]: fdBounding[0]["y"] = 0

					if "x" not in fdBounding[1]: fdBounding[1]["x"] = image["width"]
					if "y" not in fdBounding[1]: fdBounding[1]["y"] = 0
						
					if "x" not in fdBounding[2]: fdBounding[2]["x"] = image["width"]
					if "y" not in fdBounding[2]: fdBounding[2]["y"] = image["height"]

					if "x" not in fdBounding[3]: fdBounding[3]["x"] = 0
					if "y" not in fdBounding[3]: fdBounding[3]["y"] = image["height"]

					for i in range(len(bounding)):
						if "x" not in bounding[i]:
							bounding[i]["x"] = fdBounding[i]["x"]

						if "y" not in bounding[i]:
							bounding[i]["y"] = fdBounding[i]["y"]

					xOffset = bounding[0]["x"]*imageScaleFactor
					yOffset = bounding[0]["y"]*imageScaleFactor
					width = (bounding[1]["x"] - bounding[0]["x"])*imageScaleFactor
					height = (bounding[2]["y"] - bounding[0]["y"])*imageScaleFactor

					iiifFaceImageURL = image["iiifbaseuri"] + "/" + str(int(xOffset)) + "," + str(int(yOffset)) + "," + str(int(width)) + "," + str(int(height)) + "/full/0/native.jpg"
					face["iiifFaceImageURL"] = iiifFaceImageURL
					face["annotationFragment"] = "xywh=" + str(int(xOffset)) + "," + str(int(yOffset)) + "," + str(int(width)) + "," + str(int(height))

					result["responses"][0]["faceAnnotations"][index] = face

			# convert bounding polys for text annotations to IIIF image API URL to fetch the specific region
			if "textAnnotations" in result["responses"][0]:
				regionPadding = 5

				for index in range(len(result["responses"][0]["textAnnotations"])):
					text = result["responses"][0]["textAnnotations"][index]
					bounding = text["boundingPoly"]["vertices"]

					# sometimes the boundingPoly is missing X or Y values if it's too close to the edge of the image
					# fill in the missing coordinate using the bounds of the actual image
					if "x" not in bounding[0]: bounding[0]["x"] = 0
					if "y" not in bounding[0]: bounding[0]["y"] = 0

					if "x" not in bounding[1]: bounding[1]["x"] = image["width"]
					if "y" not in bounding[1]: bounding[1]["y"] = 0
						
					if "x" not in bounding[2]: bounding[2]["x"] = image["width"]
					if "y" not in bounding[2]: bounding[2]["y"] = image["height"]

					if "x" not in bounding[3]: bounding[3]["x"] = 0
					if "y" not in bounding[3]: bounding[3]["y"] = image["height"]

					seqX = [x["x"] for x in bounding]
					seqY = [x["y"] for x in bounding]

					xOffset = min(seqX)*imageScaleFactor
					yOffset = min(seqY)*imageScaleFactor
					width = (max(seqX) - min(seqX))*imageScaleFactor+regionPadding
					height = (max(seqY) - min(seqY))*imageScaleFactor+regionPadding

					iiifTextImageURL = image["iiifbaseuri"] + "/" + str(int(xOffset)) + "," + str(int(yOffset)) + "," + str(int(width)) + "," + str(int(height)) + "/full/0/native.jpg"
					text["iiifTextImageURL"] = iiifTextImageURL
					text["annotationFragment"] = "xywh=" + str(int(xOffset)) + "," + str(int(yOffset)) + "," + str(int(width)) + "," + str(int(height))

					result["responses"][0]["textAnnotations"][index] = text		

			image["googlevision"] = result

		# Run through Imagga
		if "imagga" in services: 
			image["imagga"] = {}

			# Process tags
			result = imagga.Imagga().fetch(image_url)
			if "tags" in result["result"]:
				for tag in result["result"]["tags"]:
					tag["annotationFragment"] = annotationFragmentFullImage

			image["imagga"]["tags"] = result

			# Process categories
			result = imagga.Imagga().fetch_categories(image_url)
			if "categories" in result["result"]:
				for category in result["result"]["categories"]:
					category["annotationFragment"] = annotationFragmentFullImage

			image["imagga"]["categories"] = result

		# Run through AWS Rekognition
		if "aws" in services:
			image["aws"] = {}

			# Process labels
			result = aws.AWS().fetch_labels(image_local_path)
			if "Labels" in result:
				for label in result["Labels"]:
					label["annotationFragment"] = annotationFragmentFullImage

					for instance in label["Instances"]:
						if "BoundingBox" in instance:

							xOffset = (image["width"]*instance["BoundingBox"]["Left"])*imageScaleFactor
							yOffset = (image["height"]*instance["BoundingBox"]["Top"])*imageScaleFactor
							width = (image["width"]*instance["BoundingBox"]["Width"])*imageScaleFactor
							height = (image["height"]*instance["BoundingBox"]["Height"])*imageScaleFactor

							iiifImageURL = image["iiifbaseuri"] + "/" + str(int(xOffset)) + "," + str(int(yOffset)) + "," + str(int(width)) + "," + str(int(height)) + "/full/0/native.jpg"
							instance["iiifLabelImageURL"] = iiifImageURL
							instance["annotationFragment"] = "xywh=" + str(int(xOffset)) + "," + str(int(yOffset)) + "," + str(int(width)) + "," + str(int(height))
			
			image["aws"]["labels"] = result

			# Process faces
			result = aws.AWS().fetch_faces(image_local_path)
			if "FaceDetails" in result:
				for face in result["FaceDetails"]:
					if "BoundingBox" in face:
						xOffset = (image["width"]*face["BoundingBox"]["Left"])*imageScaleFactor
						yOffset = (image["height"]*face["BoundingBox"]["Top"])*imageScaleFactor
						width = (image["width"]*face["BoundingBox"]["Width"])*imageScaleFactor
						height = (image["height"]*face["BoundingBox"]["Height"])*imageScaleFactor
						
						iiifImageURL = image["iiifbaseuri"] + "/" + str(int(xOffset)) + "," + str(int(yOffset)) + "," + str(int(width)) + "," + str(int(height)) + "/full/0/native.jpg"
						face["iiifFaceImageURL"] = iiifImageURL
						face["annotationFragment"] = "xywh=" + str(int(xOffset)) + "," + str(int(yOffset)) + "," + str(int(width)) + "," + str(int(height))
			
			image["aws"]["faces"] = result

			# Proces text
			result = aws.AWS().fetch_text(image_local_path)
			if "TextDetections" in result:
				for text in result["TextDetections"]:
					if "Geometry" in text:
						boundingBox = text["Geometry"]["BoundingBox"]

						xOffset = (image["width"]*boundingBox["Left"])*imageScaleFactor
						yOffset = (image["height"]*boundingBox["Top"])*imageScaleFactor

						# Sometimes width and height are reported as negative values. AWS documentation doesn't say why this happens.
						# I'm using ABS as a hack to make values that work in IIIF fragments
						width = (image["width"]*abs(boundingBox["Width"]))*imageScaleFactor
						height = (image["height"]*abs(boundingBox["Height"]))*imageScaleFactor

						iiifImageURL = image["iiifbaseuri"] + "/" + str(int(xOffset)) + "," + str(int(yOffset)) + "," + str(int(width)) + "," + str(int(height)) + "/full/0/native.jpg"
						text["iiifTextImageURL"] = iiifImageURL
						text["annotationFragment"] = "xywh=" + str(int(xOffset)) + "," + str(int(yOffset)) + "," + str(int(width)) + "," + str(int(height))

			image["aws"]["text"] = result

		# Run through OpenAI
		if "openai" in services: 
			image["openai"] = {}

			result = azureoai.AzureOAI().fetch(image_url)
			image["openai"] = result

		if "gpt-4" in services:
			image["gpt-4"] = {}

			result = azureoai.AzureOAI().fetch(image_url, "gpt-4")
			image["gpt-4"] = result

		if "gpt-4o" in services:
			image["gpt-4o"] = {}

			result = azureoai.AzureOAI().fetch(image_url, "gpt-4o")
			image["gpt-4o"] = result

		# Run through Claude on AWS Bedrock
		if "claude" in services: 
			image["claude"] = {}

			result = awsanthropic.AWSAnthropic().fetch(image_local_path)
			image["claude"] = result

		if "claude-3-haiku" in services: 
			image["claude-3-haiku"] = {}

			result = awsanthropic.AWSAnthropic().fetch(image_local_path)
			image["claude-3-haiku"] = result

		if "claude-3-5-sonnet" in services:
			result = awsanthropic.AWSAnthropic().fetch(image_local_path, "sonnet")
			image["claude-3-5-sonnet"] = result


	end = time.time()
	image["runtime"] = end - start

	return image


# [START run_application]
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-url', nargs='?', default=None, required=True)
	parser.add_argument('-services', nargs='+', choices=['imagga', 'gv', 'mcs', 'clarifai', 'color', 'aws', 'hash', 'openai', 'claude'], default=['imagga', 'gv', 'mcs', 'clarifai', 'color', 'aws', 'hash', 'openai', 'claude'])
	args = parser.parse_args()
	main(args.url, args.services)
# [END run_application]