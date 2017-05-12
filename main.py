import json
import argparse
import datetime
from PIL import Image
from parsers import clarifai, vision, imagga, models, ham, iiif, mcsvision


def main(page_count, person_id, technique_id, object_id, keyword):
	type = "object"

	for page_num in range(1, int(page_count)+1):
		print("Fetching page %s of %s" % (page_num, page_count))
		(success, ids) = ham.get_ham_object_id_list(page=page_num, person=person_id, technique=technique_id, object=object_id, keyword=keyword)
		if success:
			records = []

			for id in ids:	
				print("Working on record %s" % id)

				try:	
					manifest_exists = models.manifest_exists(id, type)
					if not manifest_exists:
						(success, source) = ham.get_ham_object(id)
						if success:
							# Process the record
							record = process_object(source)

							# Save it in Elasticsearch
							models.add_or_update_manifest(id, record, type)
					else:
						print("Already processed")
				except:
					print("Error processing record %s" % id)


## HELPER FUNCTIONS ##
def process_object(source):
	ham_json = json.loads(source)
	id = ham_json["id"]

	record = {
		"id": id,
		"lastupdated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	}

	images = []
	for image in ham_json["images"]:
		image_url = image["baseimageurl"]
		image_local_path = ham.get_ham_image(image_url, image["idsid"])

		# Gather and store image metadata
		im=Image.open(image_local_path)
		size = im.size # (width,height) tuple
		image["width"] = size[0]
		image["height"] = size[1]

		iiif_image_info = iiif.IIIFImage().fetch(image["iiifbaseuri"])
		imageScaleFactor = iiif_image_info["width"]/image["width"]

		# scalefactor is useful when converting annotation coordinates between different image sizes
		image["scalefactor"] = imageScaleFactor

		# # Run through Clarifai
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
				for i in range(len(bounding)):
					if "x" not in bounding[i]:
						bounding[i]["x"] = face["fdBoundingPoly"]["vertices"][i]["x"]

					if "y" not in bounding[i]:
						bounding[i]["y"] = face["fdBoundingPoly"]["vertices"][i]["y"]

				xOffset = bounding[0]["x"]*imageScaleFactor
				yOffset = bounding[0]["y"]*imageScaleFactor
				width = (bounding[1]["x"] - bounding[0]["x"])*imageScaleFactor
				height = (bounding[2]["y"] - bounding[0]["y"])*imageScaleFactor

				iiifFaceImageURL = image["iiifbaseuri"] + "/" + str(int(xOffset)) + "," + str(int(yOffset)) + "," + str(int(width)) + "," + str(int(height)) + "/full/0/native.jpg"
				face["iiifFaceImageURL"] = iiifFaceImageURL

				result["responses"][0]["faceAnnotations"][index] = face

		# convert bounding polys for text annotations to IIIF image API URL to fetch the specific region
		if "textAnnotations" in result["responses"][0]:
			regionPadding = 5

			for index in range(len(result["responses"][0]["textAnnotations"])):
				text = result["responses"][0]["textAnnotations"][index]
				bounding = text["boundingPoly"]["vertices"]

				seqX = [x["x"] for x in bounding]
				seqY = [x["y"] for x in bounding]

				xOffset = min(seqX)*imageScaleFactor
				yOffset = min(seqY)*imageScaleFactor
				width = (max(seqX) - min(seqX))*imageScaleFactor+regionPadding
				height = (max(seqY) - min(seqY))*imageScaleFactor+regionPadding

				iiifTextImageURL = image["iiifbaseuri"] + "/" + str(int(xOffset)) + "," + str(int(yOffset)) + "," + str(int(width)) + "," + str(int(height)) + "/full/0/native.jpg"
				text["iiifTextImageURL"] = iiifTextImageURL

				result["responses"][0]["textAnnotations"][index] = text		

		image["googlevision"] = result

		# Run through Imagga
		image["imagga"] = {}
		result = imagga.Imagga().fetch(image_url)
		image["imagga"]["tags"] = result

		result = imagga.Imagga().fetch_categories(image_url)
		image["imagga"]["categories"] = result

		result = imagga.Imagga().fetch_colors(image_url)
		image["imagga"]["colors"] = result

		# Run through Betaface
		# TODO

		# Run txt through Open Calias
		# TODO

		images.append(image)

	record["images"] = images

	return json.dumps(record)


# [START run_application]
if __name__ == '__main__':
	# parser = argparse.ArgumentParser()
	# parser.add_argument('image_file', help='The image you\'d like to label.')
	# args = parser.parse_args()
	# main(args.image_file)

	parser = argparse.ArgumentParser()
	parser.add_argument('-pages', nargs='?', default=1)
	parser.add_argument('-keyword', nargs='?', default=1)
	parser.add_argument('-person', nargs='?', default=None)
	parser.add_argument('-technique', nargs='?', default=None)
	parser.add_argument('-object', nargs='?', default=None)
	args = parser.parse_args()
	main(args.pages, args.person, args.technique, args.object, args.keyword)
# [END run_application]