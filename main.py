import json
import argparse
import datetime
import config
import requests
from PIL import Image
from parsers import clarifai, vision, imagga, iiif, mcsvision, colors


def main(url):
	image_info = process_image(url)
	print(json.dumps(image_info))


## HELPER FUNCTIONS ##
def get_image(URL):
	r = requests.get(URL + "?width=2400&height=2400")
	if r.status_code == 200:
		status = "ok"
		id = r.url[37:r.url.find("?")]
		path = config.TEMPORARY_FILE_DIR + "/%s.jpg" % (id)
		
		with open(path, 'wb') as out:
			for chunk in r.iter_content(chunk_size=128):
				out.write(chunk)
	else:
		status = "bad"
		id = ""
		path = ""

	return (status, path, id)

def process_image(URL):
	image = {
		"url": URL,
		"lastupdated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	}

	image_url = URL
	(status, image_local_path, id) = get_image(image_url)

	image["drsstatus"] = status

	if status == "ok": 
		image["idsid"] = id
		image["iiifbaseuri"] = iiif.IIIFImage.get_base_uri(id)

		# Gather and store image metadata
		im=Image.open(image_local_path)
		size = im.size # (width,height) tuple
		image["width"] = size[0]
		image["height"] = size[1]

		iiif_image_info = iiif.IIIFImage().fetch(image["iiifbaseuri"])
		image["widthFull"] = iiif_image_info["width"]
		image["heightFull"] = iiif_image_info["height"]

		imageScaleFactor = iiif_image_info["width"]/image["width"]

		# scalefactor is useful when converting annotation coordinates between different image sizes
		image["scalefactor"] = imageScaleFactor

		# Run through HAM color service
		result = colors.Colors().fetch_colors(image_url)
		image["colors"] = result["colors"]
		
		# Run through Clarifai
		result = clarifai.Clarifai().fetch(image_url, id)
		image["clarifai"] = result

		# Run through Microsoft Cognitive Services
		image["microsoftvision"] = {}

		result = mcsvision.MCSVision().fetch_description(image_local_path)
		image["microsoftvision"]["describe"] = result

		result = mcsvision.MCSVision().fetch_analyze(image_local_path)

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
		result = vision.Vision().fetch(image_local_path)

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
		image["imagga"] = {}
		result = imagga.Imagga().fetch(image_url)
		image["imagga"]["tags"] = result

		result = imagga.Imagga().fetch_categories(image_url)
		image["imagga"]["categories"] = result

		# result = imagga.Imagga().fetch_colors(image_url)
		# image["imagga"]["colors"] = result


	return image


# [START run_application]
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-url', nargs='?', default=None)
	args = parser.parse_args()
	main(args.url)
# [END run_application]