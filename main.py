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
from parsers import (
	azureoai, 
	clarifai, 
	vision, 
	imagga, 
	iiif, 
	mcsvision, 
	colors, 
	aws, 
	awsanthropic,
	awsmeta,
	awsnova,
	googlegemini,
	awsmistral,
	qwen,
	salesforce
)

load_dotenv()

USER_AGENT = os.getenv("USER_AGENT", "data-grinder/1.0")

temp_folder = os.path.dirname(os.path.realpath(__file__)) + "/temp"
if not os.path.exists(temp_folder): 
	os.mkdir(temp_folder)

# number of days to keep a downloaded image before refreshing it; configurable via
# environment variable to ease testing
CACHE_DAYS = int(os.getenv("IMAGE_CACHE_DAYS", "30"))

# Registry of LLM-style models that follow the simple pattern:
#   result = ModelClass().fetch(image_path, model_enum)
#   result["annotationFragment"] = annotationFragmentFullImage
# Each entry is (model_enum, model_class, image_size)
# image_size is "full" for full-resolution or "scaled" for the 1110px version
GENERIC_MODELS = [
	# Azure / OpenAI
	(azureoai.OpenAIModel.OPENAI,               azureoai.AzureOAI,           "full"),
	(azureoai.OpenAIModel.GPT_4,                azureoai.AzureOAI,           "full"),
	(azureoai.OpenAIModel.GPT_4O,               azureoai.AzureOAI,           "full"),
	(azureoai.OpenAIModel.GPT_4_1_MINI,         azureoai.AzureOAI,           "full"),
	# Anthropic / Claude on AWS Bedrock
	(awsanthropic.AnthropicModel.CLAUDE,            awsanthropic.AWSAnthropic, "full"),
	(awsanthropic.AnthropicModel.CLAUDE_3_HAIKU,    awsanthropic.AWSAnthropic, "full"),
	(awsanthropic.AnthropicModel.CLAUDE_4_5_HAIKU,  awsanthropic.AWSAnthropic, "full"),
	(awsanthropic.AnthropicModel.CLAUDE_3_OPUS,     awsanthropic.AWSAnthropic, "full"),
	(awsanthropic.AnthropicModel.CLAUDE_4_1_OPUS,   awsanthropic.AWSAnthropic, "full"),
	(awsanthropic.AnthropicModel.CLAUDE_4_5_OPUS,   awsanthropic.AWSAnthropic, "full"),
	(awsanthropic.AnthropicModel.CLAUDE_3_5_SONNET,   awsanthropic.AWSAnthropic, "full"),
	(awsanthropic.AnthropicModel.CLAUDE_3_5_SONNET_2, awsanthropic.AWSAnthropic, "full"),
	(awsanthropic.AnthropicModel.CLAUDE_3_7_SONNET,   awsanthropic.AWSAnthropic, "full"),
	(awsanthropic.AnthropicModel.CLAUDE_4_SONNET,     awsanthropic.AWSAnthropic, "full"),
	(awsanthropic.AnthropicModel.CLAUDE_4_5_SONNET,   awsanthropic.AWSAnthropic, "full"),
	# Meta / Llama on AWS Bedrock (scaled image)
	(awsmeta.MetaModel.LLAMA_3_2_11B,       awsmeta.AWSMeta,   "scaled"),
	(awsmeta.MetaModel.LLAMA_3_2_90B,       awsmeta.AWSMeta,   "scaled"),
	(awsmeta.MetaModel.LLAMA_4_MAVERICK_17B, awsmeta.AWSMeta,  "scaled"),
	(awsmeta.MetaModel.LLAMA_4_SCOUT_17B,    awsmeta.AWSMeta,  "scaled"),
	# Nova on AWS Bedrock (scaled image)
	(awsnova.NovaModel.NOVA_LITE_1_0, awsnova.AWSNova, "scaled"),
	(awsnova.NovaModel.NOVA_PRO_1_0,  awsnova.AWSNova, "scaled"),
	(awsnova.NovaModel.NOVA_LITE_2_0, awsnova.AWSNova, "scaled"),
	# Google Gemini (scaled image)
	(googlegemini.GoogleGeminiModel.FLASH_2_0,      googlegemini.GoogleGemini, "scaled"),
	(googlegemini.GoogleGeminiModel.FLASH_2_5,      googlegemini.GoogleGemini, "scaled"),
	(googlegemini.GoogleGeminiModel.FLASH_LITE_2_0, googlegemini.GoogleGemini, "scaled"),
	(googlegemini.GoogleGeminiModel.FLASH_LITE_2_5, googlegemini.GoogleGemini, "scaled"),
	# Mistral on AWS Bedrock (scaled image)
	(awsmistral.MistralModel.PIXTRAL_LARGE_2502,   awsmistral.AWSMistral, "scaled"),
	(awsmistral.MistralModel.MAGISTRAL_SMALL_2509, awsmistral.AWSMistral, "scaled"),
	(awsmistral.MistralModel.MINISTRAL_3_3B,       awsmistral.AWSMistral, "scaled"),
	(awsmistral.MistralModel.MINISTRAL_3_8B,       awsmistral.AWSMistral, "scaled"),
	(awsmistral.MistralModel.MINISTRAL_3_14B,      awsmistral.AWSMistral, "scaled"),
	(awsmistral.MistralModel.MISTRAL_LARGE_3_675B, awsmistral.AWSMistral, "scaled"),
	# Qwen on Hyperbolic
	(qwen.QwenModel.QWEN_2_5_VL_7B,  qwen.Qwen, "full"),
	(qwen.QwenModel.QWEN_2_5_VL_72B, qwen.Qwen, "full"),
]

app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
	return {"status": "ok"}

@app.route("/list/services", methods=['GET'])
def list_services():
	return {"services": azureoai.OpenAIModel.list_models() + \
			awsanthropic.AnthropicModel.list_models() + \
			awsmeta.MetaModel.list_models() + \
			awsnova.NovaModel.list_models() + \
			googlegemini.GoogleGeminiModel.list_models() + \
			awsmistral.MistralModel.list_models() + \
			qwen.QwenModel.list_models() + \
			aws.AWSModel.list_models() + \
			clarifai.ClarifaiModel.list_models() + \
			imagga.ImaggaModel.list_models() + \
			mcsvision.MCSVisionModel.list_models() + \
			vision.GVisionModel.list_models() + \
			salesforce.SalesForceModel.list_models() + \
			[{"name":"hash", "model_id":""}] +\
			[{"name":"color", "model_id":""}]}

@app.route("/extract", methods=['GET'])
def extract():
	response = {"status": "missing parameters url, services"}

	url = request.args.get('url')
	services = request.args.get('services')
	if services is not None:
		services = parse_service_features(services)

	if url and services: 
		response = process_image(url, services)

	return response

def main(url, services):
	image_info = process_image(url, services)
	print(json.dumps(image_info))

## HELPER FUNCTIONS ##
def parse_service_features(query: str, default_value: str = "all"):
	result = {}

	for part in query.split(","):
		part = part.strip()
		if not part:
			continue

		# If no colon, just assume key with default value
		if ":" not in part:
			key = part.strip()
			result[key] = [default_value]
			continue

		key, features_str = part.split(":", 1)
		key = key.strip()

		# If feature list is missing or empty/whitespace-only
		if not features_str.strip():
			features = [default_value]
		else:
			features = [f.strip() for f in features_str.split("|") if f.strip()]
			if not features:
				features = [default_value]

		result[key] = features

	return result

def download_image(URL,filename="temp.jpg"):
	# create a subfolder using the first four characters of the filename 
	# to avoid having too many files in a single folder which can cause performance issues on some file systems
	# this also makes it easier to manage and clean up cached images if needed
	subfolder_name = filename[:4]
	subfolder_path = os.path.join(temp_folder, subfolder_name)
	
	# Create subfolder if it doesn't exist
	if not os.path.exists(subfolder_path):
		os.makedirs(subfolder_path)
	
	path = os.path.join(subfolder_path, filename)

	# try to reuse a recently downloaded copy in the temp folder 
	# to avoid unnecessary downloads and speed up processing
	# check file modified time to ensure it's not too old
	if os.path.exists(path):
		age = time.time() - os.path.getmtime(path)
		# check against configurable cache lifetime
		if age < CACHE_DAYS * 24 * 3600:
			return ("ok", path)
		# otherwise fall through and re-download a fresh copy

	r = requests.get(URL, headers={"User-Agent": USER_AGENT}, timeout=21)
	
	if r.status_code == 200:
		status = "ok"
		with open(path, 'wb') as out:
			for chunk in r.iter_content(chunk_size=128):
				out.write(chunk)
	else:
		status = "bad"
		path = ""

	return (status, path)

# ── Bounding-box helpers ──────────────────────────────────────────────────────

def _make_annotation_fragment(x, y, w, h):
	"""Return an IIIF xywh= annotation fragment string from integer pixel coords."""
	return f"xywh={int(x)},{int(y)},{int(w)},{int(h)}"

def _scale_bbox(x, y, w, h, scale):
	"""Apply a uniform scale factor to a bounding box."""
	return x * scale, y * scale, w * scale, h * scale

# ── Per-service handlers ──────────────────────────────────────────────────────

def _run_hash(image, image_local_path):
	"""Compute multiple image hashes and store them in image["hashes"]."""
	i = Image.open(image_local_path)
	image["hashes"] = {
		"average":    str(imagehash.average_hash(i)),
		"color":      str(imagehash.colorhash(i)),
		"perceptual": str(imagehash.phash(i)),
		"difference": str(imagehash.dhash(i)),
		"wavelet":    str(imagehash.whash(i)),
	}

def _run_color(image, image_local_path):
	"""Run the HAM color service and store results in image["colors"]."""
	image["colors"] = colors.Colors().fetch(image_local_path)

def _run_clarifai(image, image_local_path, features, annotation_fragment, image_width, image_height, image_scale):
	"""Run Clarifai classification, object detection, and/or color analysis."""
	image["clarifai"] = {}

	if any(v in ["all", "classification"] for v in features):
		result = clarifai.Clarifai().fetch(image_local_path)
		if "outputs" in result and result["outputs"] and "data" in result["outputs"][0]:
			for concept in result["outputs"][0]["data"]["concepts"]:
				concept["annotationFragment"] = annotation_fragment
		image["clarifai"]["classification"] = result

	if any(v in ["all", "objects"] for v in features):
		result = clarifai.Clarifai().fetch_objects(image_local_path)
		if "outputs" in result and result["outputs"] and "data" in result["outputs"][0]:
			if "regions" in result["outputs"][0]["data"]:
				for region in result["outputs"][0]["data"]["regions"]:
					bb = region["region_info"]["bounding_box"]
					left   = int((bb["left_col"]  * image_width)  * image_scale)
					top    = int((bb["top_row"]   * image_height) * image_scale)
					right  = int((bb["right_col"] * image_width)  * image_scale)
					bottom = int((bb["bottom_row"]* image_height) * image_scale)
					region["annotationFragment"] = _make_annotation_fragment(left, top, right - left, bottom - top)
		image["clarifai"]["objects"] = result

	if any(v in ["all", "colors"] for v in features):
		image["clarifai"]["colors"] = clarifai.Clarifai().fetch_colors(image_local_path)

def _run_microsoftvision(image, image_local_path, features, annotation_fragment, image_scale, iiif_image):
	"""Run Microsoft Cognitive Services describe and/or analyze."""
	image["microsoftvision"] = {}

	if any(v in ["all", "describe"] for v in features):
		result = mcsvision.MCSVision().fetch_description(image_local_path)
		if "description" in result:
			for caption in result["description"]["captions"]:
				caption["annotationFragment"] = annotation_fragment
		image["microsoftvision"]["describe"] = result

	if any(v in ["all", "analyze"] for v in features):
		result = mcsvision.MCSVision().fetch_analyze(image_local_path)

		if "description" in result:
			for caption in result["description"]["captions"]:
				caption["annotationFragment"] = annotation_fragment

		if "categories" in result:
			for category in result["categories"]:
				category["annotationFragment"] = annotation_fragment

		if "tags" in result:
			for tag in result["tags"]:
				tag["annotationFragment"] = annotation_fragment

		if "faces" in result:
			for i, face in enumerate(result["faces"]):
				if "faceRectangle" in face:
					x, y, w, h = _scale_bbox(
						face["faceRectangle"]["left"],
						face["faceRectangle"]["top"],
						face["faceRectangle"]["width"],
						face["faceRectangle"]["height"],
						image_scale
					)
					face["iiifFaceImageURL"] = iiif_image.get_fragment_image_url(int(x), int(y), int(w), int(h))
					face["annotationFragment"] = _make_annotation_fragment(x, y, w, h)
					result["faces"][i] = face

		if "objects" in result:
			for i, obj in enumerate(result["objects"]):
				x, y, w, h = _scale_bbox(
					obj["rectangle"]["x"],
					obj["rectangle"]["y"],
					obj["rectangle"]["w"],
					obj["rectangle"]["h"],
					image_scale
				)
				obj["iiifFaceImageURL"] = iiif_image.get_fragment_image_url(int(x), int(y), int(w), int(h))
				obj["annotationFragment"] = _make_annotation_fragment(x, y, w, h)
				result["objects"][i] = obj

		image["microsoftvision"]["analyze"] = result

def _run_googlevision(image, image_local_path, image_width, image_height, image_scale, iiif_image):
	"""Run Google Vision and annotate labels, faces, and text regions."""
	result = vision.Vision().fetch(image_local_path)
	resp = result["responses"][0]

	if "labelAnnotations" in resp:
		annotation_fragment = _make_annotation_fragment(0, 0, image_width, image_height)
		for label in resp["labelAnnotations"]:
			label["annotationFragment"] = annotation_fragment

	if "faceAnnotations" in resp:
		for i, face in enumerate(resp["faceAnnotations"]):
			bounding   = face["boundingPoly"]["vertices"]
			fd_bounding = face["fdBoundingPoly"]["vertices"]

			# Fill in missing coordinates using fdBoundingPoly fallbacks
			fd_defaults = [(0, 0), (image_width, 0), (image_width, image_height), (0, image_height)]
			for j, (fx, fy) in enumerate(fd_defaults):
				if "x" not in fd_bounding[j]: fd_bounding[j]["x"] = fx
				if "y" not in fd_bounding[j]: fd_bounding[j]["y"] = fy

			for j in range(len(bounding)):
				if "x" not in bounding[j]: bounding[j]["x"] = fd_bounding[j]["x"]
				if "y" not in bounding[j]: bounding[j]["y"] = fd_bounding[j]["y"]

			x, y, w, h = _scale_bbox(
				bounding[0]["x"],
				bounding[0]["y"],
				bounding[1]["x"] - bounding[0]["x"],
				bounding[2]["y"] - bounding[0]["y"],
				image_scale
			)
			face["iiifFaceImageURL"] = iiif_image.get_fragment_image_url(int(x), int(y), int(w), int(h))
			face["annotationFragment"] = _make_annotation_fragment(x, y, w, h)
			resp["faceAnnotations"][i] = face

	if "textAnnotations" in resp:
		region_padding = 5
		for i, text in enumerate(resp["textAnnotations"]):
			bounding = text["boundingPoly"]["vertices"]

			# Fill in missing coordinates using image bounds as fallback
			edge_defaults = [(0, 0), (image_width, 0), (image_width, image_height), (0, image_height)]
			for j, (ex, ey) in enumerate(edge_defaults):
				if "x" not in bounding[j]: bounding[j]["x"] = ex
				if "y" not in bounding[j]: bounding[j]["y"] = ey

			xs = [v["x"] for v in bounding]
			ys = [v["y"] for v in bounding]
			x, y, w, h = _scale_bbox(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), image_scale)
			w += region_padding
			h += region_padding

			text["iiifTextImageURL"] = iiif_image.get_fragment_image_url(int(x), int(y), int(w), int(h))
			text["annotationFragment"] = _make_annotation_fragment(x, y, w, h)
			resp["textAnnotations"][i] = text

	image["googlevision"] = result

def _run_imagga(image, image_local_path, features, annotation_fragment, image_width, image_height, image_scale, iiif_image):
	"""Run Imagga tags, categories, faces, and/or colors."""
	image["imagga"] = {}

	if any(v in ["all", "tags"] for v in features):
		result = imagga.Imagga().fetch(image_local_path)
		if "tags" in result.get("result", {}):
			for tag in result["result"]["tags"]:
				tag["annotationFragment"] = annotation_fragment
		image["imagga"]["tags"] = result

	if any(v in ["all", "categories"] for v in features):
		result = imagga.Imagga().fetch_categories(image_local_path)
		if "categories" in result.get("result", {}):
			for category in result["result"]["categories"]:
				category["annotationFragment"] = annotation_fragment
		image["imagga"]["categories"] = result

	if any(v in ["all", "faces"] for v in features):
		result = imagga.Imagga().fetch_faces(image_local_path)
		if "faces" in result.get("result", {}):
			for i, face in enumerate(result["result"]["faces"]):
				x, y, w, h = _scale_bbox(
					face["coordinates"]["xmin"],
					face["coordinates"]["ymin"],
					face["coordinates"]["width"],
					face["coordinates"]["height"],
					image_scale
				)
				face["iiifFaceImageURL"] = iiif_image.get_fragment_image_url(int(x), int(y), int(w), int(h))
				face["annotationFragment"] = _make_annotation_fragment(x, y, w, h)
				result["result"]["faces"][i] = face
		image["imagga"]["faces"] = result

	if any(v in ["all", "colors"] for v in features):
		image["imagga"]["colors"] = imagga.Imagga().fetch_colors(image_local_path)

def _run_aws_rekognition(image, image_local_path, features, image_width, image_height, image_scale, iiif_image, annotation_fragment):
	"""Run AWS Rekognition labels, faces, and/or text detection."""
	image["aws"] = {}

	if any(v in ["all", "labels"] for v in features):
		result = aws.AWS().fetch_labels(image_local_path)
		if "Labels" in result:
			for label in result["Labels"]:
				label["annotationFragment"] = annotation_fragment
				for instance in label["Instances"]:
					if "BoundingBox" in instance:
						x, y, w, h = _scale_bbox(
							image_width  * instance["BoundingBox"]["Left"],
							image_height * instance["BoundingBox"]["Top"],
							image_width  * instance["BoundingBox"]["Width"],
							image_height * instance["BoundingBox"]["Height"],
							image_scale
						)
						instance["iiifLabelImageURL"] = iiif_image.get_fragment_image_url(int(x), int(y), int(w), int(h))
						instance["annotationFragment"] = _make_annotation_fragment(x, y, w, h)
		image["aws"]["labels"] = result

	if any(v in ["all", "faces"] for v in features):
		result = aws.AWS().fetch_faces(image_local_path)
		if "FaceDetails" in result:
			for face in result["FaceDetails"]:
				if "BoundingBox" in face:
					x, y, w, h = _scale_bbox(
						image_width  * face["BoundingBox"]["Left"],
						image_height * face["BoundingBox"]["Top"],
						image_width  * face["BoundingBox"]["Width"],
						image_height * face["BoundingBox"]["Height"],
						image_scale
					)
					face["iiifFaceImageURL"] = iiif_image.get_fragment_image_url(int(x), int(y), int(w), int(h))
					face["annotationFragment"] = _make_annotation_fragment(x, y, w, h)
		image["aws"]["faces"] = result

	if any(v in ["all", "text"] for v in features):
		result = aws.AWS().fetch_text(image_local_path)
		if "TextDetections" in result:
			for text in result["TextDetections"]:
				if "Geometry" in text:
					bb = text["Geometry"]["BoundingBox"]
					# AWS can return negative width/height; use abs() as a safeguard
					x, y, w, h = _scale_bbox(
						image_width  * bb["Left"],
						image_height * bb["Top"],
						image_width  * abs(bb["Width"]),
						image_height * abs(bb["Height"]),
						image_scale
					)
					text["iiifTextImageURL"] = iiif_image.get_fragment_image_url(int(x), int(y), int(w), int(h))
					text["annotationFragment"] = _make_annotation_fragment(x, y, w, h)
		image["aws"]["text"] = result

def _run_salesforce(image, image_local_path, services, annotation_fragment):
	"""Run Salesforce/BLIP models via Clarifai."""
	blip_models = [
		salesforce.SalesForceModel.BLIP,
		salesforce.SalesForceModel.BLIP_2,
		salesforce.SalesForceModel.BLIP_2_6_7B,
	]
	for model in blip_models:
		if model.name in services:
			result = salesforce.SalesForce().fetch(image_local_path, model)
			if "data" in result["outputs"][0]:
				result["outputs"][0]["data"]["annotationFragment"] = annotation_fragment
			image[model.name] = result

# ── Main orchestrator ─────────────────────────────────────────────────────────

def process_image(URL, services):
	image = {
		"url": URL,
		"lastupdated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		"runtime": 0
	}

	start = time.time()

	iiif_image = iiif.IIIFImage(URL)
	image["drsstatus"] = iiif_image.status

	if iiif_image.is_valid():
		image_url = iiif_image.get_full_image_url()

		image["idsid"]         = int(iiif_image.id)
		image["iiifbaseuri"]   = iiif_image.get_base_uri()
		image["iiifFullImageURL"] = image_url

		# Download full and scaled copies
		_, image_local_path        = download_image(image_url, f"{iiif_image.id}.jpg")
		_, image_local_path_scaled = download_image(iiif_image.get_scaled_image_url("!1110,1110"), f"{iiif_image.id}_1110.jpg")

		# Image metadata
		im = Image.open(image_local_path)
		image["width"], image["height"] = im.size
		image["widthFull"]  = iiif_image.info["width"]
		image["heightFull"] = iiif_image.info["height"]

		image_scale      = iiif_image.info["width"] / image["width"]
		image["scalefactor"] = image_scale

		annotation_fragment_full = _make_annotation_fragment(0, 0, image["width"], image["height"])

		# ── Simple / hash / color services ──────────────────────────────────
		if "hash" in services:
			_run_hash(image, image_local_path)

		if "color" in services:
			_run_color(image, image_local_path)

		# ── Structured vision services ───────────────────────────────────────
		if clarifai.ClarifaiModel.BASE.name in services:
			_run_clarifai(image, image_local_path,
						  services[clarifai.ClarifaiModel.BASE.name],
						  annotation_fragment_full,
						  image["width"], image["height"], image_scale)

		if mcsvision.MCSVisionModel.BASE.name in services:
			_run_microsoftvision(image, image_local_path,
								 services[mcsvision.MCSVisionModel.BASE.name],
								 annotation_fragment_full,
								 image_scale, iiif_image)

		if vision.GVisionModel.BASE.name in services:
			_run_googlevision(image, image_local_path,
							  image["width"], image["height"],
							  image_scale, iiif_image)

		if imagga.ImaggaModel.BASE.name in services:
			_run_imagga(image, image_local_path,
						services[imagga.ImaggaModel.BASE.name],
						annotation_fragment_full,
						image["width"], image["height"],
						image_scale, iiif_image)

		if aws.AWSModel.BASE.name in services:
			_run_aws_rekognition(image, image_local_path,
								 services[aws.AWSModel.BASE.name],
								 image["width"], image["height"],
								 image_scale, iiif_image,
								 annotation_fragment_full)

		# ── Generic LLM / vision model dispatch ─────────────────────────────
		path_map = {"full": image_local_path, "scaled": image_local_path_scaled}
		for model, cls, img_size in GENERIC_MODELS:
			if model.name in services:
				result = cls().fetch(path_map[img_size], model)
				result["annotationFragment"] = annotation_fragment_full
				image[model.name] = result

		# ── Salesforce / BLIP ────────────────────────────────────────────────
		_run_salesforce(image, image_local_path, services, annotation_fragment_full)

	image["runtime"] = time.time() - start
	return image


# [START run_application]
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-url', nargs='?', default=None, required=True)
	parser.add_argument('-services', nargs='+', choices=['imagga', 'gv', 'mcs', 'clarifai', 'color', 'aws', 'hash', 'openai', 'claude'], default=['imagga', 'gv', 'mcs', 'clarifai', 'color', 'aws', 'hash'])
	args = parser.parse_args()
	main(args.url, args.services)
# [END run_application]
