import gc
import os
import json
import logging
import argparse
import datetime
import time
import unicodedata
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import imagehash
from PIL import Image
from flask import Flask, request
from dotenv import  load_dotenv
import log
import cache
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
	salesforce,
	awsqwen,
	awsmoonshot,
	awswriter,
	ollama
)

log.configure()
logger = logging.getLogger(__name__)

load_dotenv()

USER_AGENT = os.getenv("USER_AGENT", "data-grinder/1.0")
MAX_PROMPT_LEN = int(os.getenv("MAX_PROMPT_LEN", "500"))
LLM_CONNECT_TIMEOUT = int(os.getenv("LLM_CONNECT_TIMEOUT", "10"))
LLM_READ_TIMEOUT = int(os.getenv("LLM_READ_TIMEOUT", "60"))
REQUEST_BUDGET = int(os.getenv("REQUEST_BUDGET", "90"))

_executor = ThreadPoolExecutor(max_workers=int(os.getenv("LLM_WORKERS", "10")))

# Registry of LLM-style models that follow the simple pattern:
#   result = ModelClass().fetch(image_path, model_enum)
#   result["annotationFragment"] = annotationFragmentFullImage
# Each entry is (model_enum, model_class, image_size)
# image_size is "full" for full-resolution or "1110" for the 1110px version or "512" for the 512px version
GENERIC_MODELS = [
	# Azure / OpenAI
	(azureoai.OpenAIModel.GPT_4,                azureoai.AzureOAI,           "full"),
	(azureoai.OpenAIModel.GPT_4O,               azureoai.AzureOAI,           "full"),
	(azureoai.OpenAIModel.GPT_4_1_MINI,         azureoai.AzureOAI,           "full"),
	(azureoai.OpenAIModel.GPT_5_NANO,          	azureoai.AzureOAI,           "full"),
	# Anthropic / Claude on AWS Bedrock
	(awsanthropic.AnthropicModel.CLAUDE_3_HAIKU,    awsanthropic.AWSAnthropic, "full"),
	(awsanthropic.AnthropicModel.CLAUDE_4_5_HAIKU,  awsanthropic.AWSAnthropic, "full"),
	(awsanthropic.AnthropicModel.CLAUDE_3_OPUS,     awsanthropic.AWSAnthropic, "full"),
	(awsanthropic.AnthropicModel.CLAUDE_4_5_OPUS,   awsanthropic.AWSAnthropic, "full"),
	(awsanthropic.AnthropicModel.CLAUDE_3_5_SONNET,   awsanthropic.AWSAnthropic, "full"),
	(awsanthropic.AnthropicModel.CLAUDE_3_5_SONNET_2, awsanthropic.AWSAnthropic, "full"),
	(awsanthropic.AnthropicModel.CLAUDE_3_7_SONNET,   awsanthropic.AWSAnthropic, "full"),
	(awsanthropic.AnthropicModel.CLAUDE_4_SONNET,     awsanthropic.AWSAnthropic, "full"),
	(awsanthropic.AnthropicModel.CLAUDE_4_5_SONNET,   awsanthropic.AWSAnthropic, "full"),
	# Meta / Llama on AWS Bedrock (scaled image)
	(awsmeta.MetaModel.LLAMA_3_2_11B,       awsmeta.AWSMeta,   "1110"),
	(awsmeta.MetaModel.LLAMA_3_2_90B,       awsmeta.AWSMeta,   "1110"),
	(awsmeta.MetaModel.LLAMA_4_MAVERICK_17B, awsmeta.AWSMeta,  "1110"),
	(awsmeta.MetaModel.LLAMA_4_SCOUT_17B,    awsmeta.AWSMeta,  "1110"),
	# Nova on AWS Bedrock (scaled image)
	(awsnova.NovaModel.NOVA_LITE_1_0, awsnova.AWSNova, "1110"),
	(awsnova.NovaModel.NOVA_PRO_1_0,  awsnova.AWSNova, "1110"),
	(awsnova.NovaModel.NOVA_LITE_2_0, awsnova.AWSNova, "1110"),
	# Google Gemini (scaled image)
	(googlegemini.GoogleGeminiModel.FLASH_2_0,      googlegemini.GoogleGemini, "1110"),
	(googlegemini.GoogleGeminiModel.FLASH_2_5,      googlegemini.GoogleGemini, "1110"),
	(googlegemini.GoogleGeminiModel.FLASH_LITE_2_0, googlegemini.GoogleGemini, "1110"),
	(googlegemini.GoogleGeminiModel.FLASH_LITE_2_5, googlegemini.GoogleGemini, "1110"),
	(googlegemini.GoogleGeminiModel.FLASH_LITE_3_1, googlegemini.GoogleGemini, "1110"),
	# Mistral on AWS Bedrock (scaled image)
	(awsmistral.MistralModel.PIXTRAL_LARGE_2502,   awsmistral.AWSMistral, "1110"),
	(awsmistral.MistralModel.MAGISTRAL_SMALL_2509, awsmistral.AWSMistral, "1110"),
	(awsmistral.MistralModel.MINISTRAL_3_3B,       awsmistral.AWSMistral, "1110"),
	(awsmistral.MistralModel.MINISTRAL_3_8B,       awsmistral.AWSMistral, "1110"),
	(awsmistral.MistralModel.MINISTRAL_3_14B,      awsmistral.AWSMistral, "1110"),
	(awsmistral.MistralModel.MISTRAL_LARGE_3_675B, awsmistral.AWSMistral, "1110"),
	# Qwen on Hyperbolic
	(qwen.QwenModel.QWEN_2_5_VL_7B,  qwen.Qwen, "full"),
	(qwen.QwenModel.QWEN_2_5_VL_72B, qwen.Qwen, "full"),
	# Qwen on AWS Bedrock
	(awsqwen.QwenModel.QWEN_3_VL_235B, awsqwen.AWSQwen, "1110"),
	# Moonshot on AWS Bedrock 
	(awsmoonshot.MoonshotModel.KIMI_K_2_5, awsmoonshot.AWSMoonshot, "1110"),
	# Writer on AWS Bedrock
	(awswriter.WriterModel.PALMYRA_VISION_7B, awswriter.AWSWriter, "1110"),
	# Salesforce BLIP on Clarifai
	(salesforce.SalesForceModel.BLIP,           salesforce.SalesForce, "full"),
	(salesforce.SalesForceModel.BLIP_2,         salesforce.SalesForce, "full"),
	(salesforce.SalesForceModel.BLIP_2_6_7B,   	salesforce.SalesForce, "full"),
	# Gemma on local Ollama (scaled image)
	(ollama.OllamaModel.GEMMA_4,     ollama.Ollama, "1110"),
	(ollama.OllamaModel.GEMMA_4_26B, ollama.Ollama, "1110"),
]

app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
	return {"status": "ok"}

@app.route("/list/services", methods=['GET'])
def list_services():
	return {"services": 
		{
			"computer vision": 
				aws.AWSModel.list_models() + \
				clarifai.ClarifaiModel.list_models() + \
				imagga.ImaggaModel.list_models() + \
				mcsvision.MCSVisionModel.list_models() + \
				vision.GVisionModel.list_models(), 
			"large language models": 
				azureoai.OpenAIModel.list_models() + \
				awsanthropic.AnthropicModel.list_models() + \
				awsmeta.MetaModel.list_models() + \
				awsnova.NovaModel.list_models() + \
				googlegemini.GoogleGeminiModel.list_models() + \
				awsmistral.MistralModel.list_models() + \
				qwen.QwenModel.list_models() + \
				awsqwen.QwenModel.list_models() + \
				salesforce.SalesForceModel.list_models() + \
				awsmoonshot.MoonshotModel.list_models() + \
				awswriter.WriterModel.list_models() + \
				ollama.OllamaModel.list_models(),
			"other": [
				{"name":"hash", "model_id":"", "eol_date": None}, 
				{"name":"color", "model_id":"", "eol_date": None}
			]
		}
	}

@app.after_request
def _force_gc(response):
	gc.collect()
	return response

@app.route("/extract", methods=['GET'])
def extract():
	response = {"status": "missing parameters url, services"}

	url = request.args.get('url')
	services = request.args.get('services')
	prompt = request.args.get('prompt')

	if prompt is not None:
		prompt, err = _validate_prompt(prompt)
		if err:
			return {"status": err}, 400

	if services is not None:
		services = parse_service_features(services)

	if url and services:
		logger.info("request", extra={"url": url, "services": list(services.keys()), "prompt": prompt})
		response = process_image(url, services, prompt=prompt)

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

def _validate_prompt(raw: str):
	if len(raw) > MAX_PROMPT_LEN:
		return "", f"prompt exceeds {MAX_PROMPT_LEN} character limit"
	
	sanitized = "".join(
		c for c in raw
		if not unicodedata.category(c).startswith("C") or c in " \t\n\r"
	)

	return sanitized, None

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
	with Image.open(image_local_path) as i:
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

	if any(v in ["all", "structured-tags"] for v in features):
		result = imagga.Imagga().fetch_structured_tags(image_local_path)
		if "caption" in result.get("result", {}):
			image["imagga"]["caption"] = result["result"]["caption"]

		if "tags" in result.get("result", {}):
			result["result"]["tags"]["annotationFragment"] = annotation_fragment
		image["imagga"]["structuredTags"] = result

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

def _fetch_model(model, cls, img_size, cached, prompt, annotation_fragment):
	t0 = time.time()
	result = cls().fetch(cached[img_size]["path"], model, prompt=prompt,
						 connect_timeout=LLM_CONNECT_TIMEOUT,
						 read_timeout=LLM_READ_TIMEOUT)
	result["runtime"] = time.time() - t0
	result["annotationFragment"] = annotation_fragment
	logger.info("model_result", extra={"model": model.name, "status": result["status"], "runtime_s": round(result["runtime"], 3)})
	return model.name, result

# ── Main orchestrator ─────────────────────────────────────────────────────────

def process_image(URL, services, prompt=None):
	image = {
		"url": URL,
		"lastupdated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		"runtime": 0
	}

	start = time.time()

	iiif_image = iiif.IIIFImage(URL)
	image["status"] = iiif_image.status

	if not iiif_image.is_valid():
		time.sleep(1)
	else:
		image_url = iiif_image.get_full_image_url()

		image["iiifbaseuri"]   = iiif_image.get_base_uri()
		image["iiifFullImageURL"] = image_url

		cached = cache.get_image(image_url, URL, cache.CACHE_DAYS)
		logger.info("cache", extra={"status": cached["status"], "url": URL})
		if cached["status"] != "ok":
			logger.warning("cache_failed", extra={"url": URL})
			image["status"] = "bad"
			image["runtime"] = time.time() - start
			return image

		image["width"]   = cached["full"]["width"]
		image["height"]  = cached["full"]["height"]

		image["widthFull"]  = iiif_image.info["width"]
		image["heightFull"] = iiif_image.info["height"]

		image_scale          = iiif_image.info["width"] / image["width"]
		image["scalefactor"] = image_scale

		annotation_fragment_full = _make_annotation_fragment(0, 0, image["width"], image["height"])

		# ── Simple / hash / color services ──────────────────────────────────
		if "hash" in services:
			_t = time.time()
			try:
				_run_hash(image, cached["full"]["path"])
				logger.info("service_result", extra={"service": "hash", "runtime_s": round(time.time() - _t, 3)})
			except Exception:
				logger.error("service_failed", extra={"service": "hash"}, exc_info=True)

		if "color" in services:
			_t = time.time()
			try:
				_run_color(image, cached["full"]["path"])
				logger.info("service_result", extra={"service": "color", "runtime_s": round(time.time() - _t, 3)})
			except Exception:
				logger.error("service_failed", extra={"service": "color"}, exc_info=True)

		# ── Structured vision services ───────────────────────────────────────
		if clarifai.ClarifaiModel.BASE.name in services:
			_t = time.time()
			try:
				_run_clarifai(image, cached["full"]["path"],
							  services[clarifai.ClarifaiModel.BASE.name],
							  annotation_fragment_full,
							  image["width"], image["height"], image_scale)
				logger.info("service_result", extra={"service": "clarifai", "runtime_s": round(time.time() - _t, 3)})
			except Exception:
				logger.error("service_failed", extra={"service": "clarifai"}, exc_info=True)

		if mcsvision.MCSVisionModel.BASE.name in services:
			_t = time.time()
			try:
				_run_microsoftvision(image, cached["full"]["path"],
									 services[mcsvision.MCSVisionModel.BASE.name],
									 annotation_fragment_full,
									 image_scale, iiif_image)
				logger.info("service_result", extra={"service": "microsoftvision", "runtime_s": round(time.time() - _t, 3)})
			except Exception:
				logger.error("service_failed", extra={"service": "microsoftvision"}, exc_info=True)

		if vision.GVisionModel.BASE.name in services:
			_t = time.time()
			try:
				_run_googlevision(image, cached["full"]["path"],
								  image["width"], image["height"],
								  image_scale, iiif_image)
				logger.info("service_result", extra={"service": "googlevision", "runtime_s": round(time.time() - _t, 3)})
			except Exception:
				logger.error("service_failed", extra={"service": "googlevision"}, exc_info=True)

		if imagga.ImaggaModel.BASE.name in services:
			_t = time.time()
			try:
				_run_imagga(image, cached["full"]["path"],
							services[imagga.ImaggaModel.BASE.name],
							annotation_fragment_full,
							image["width"], image["height"],
							image_scale, iiif_image)
				logger.info("service_result", extra={"service": "imagga", "runtime_s": round(time.time() - _t, 3)})
			except Exception:
				logger.error("service_failed", extra={"service": "imagga"}, exc_info=True)

		if aws.AWSModel.BASE.name in services:
			_t = time.time()
			try:
				_run_aws_rekognition(image, cached["full"]["path"],
									 services[aws.AWSModel.BASE.name],
									 image["width"], image["height"],
									 image_scale, iiif_image,
									 annotation_fragment_full)
				logger.info("service_result", extra={"service": "aws", "runtime_s": round(time.time() - _t, 3)})
			except Exception:
				logger.error("service_failed", extra={"service": "aws"}, exc_info=True)

		# ── Generic LLM / vision model dispatch (parallel) ──────────────────
		active_models = [(m, c, s) for m, c, s in GENERIC_MODELS if m.name in services]
		futures = {
			_executor.submit(
				_fetch_model, m, c, s, cached, prompt, annotation_fragment_full
			): m
			for m, c, s in active_models
		}
		remaining = max(0, REQUEST_BUDGET - (time.time() - start))
		done_iter = as_completed(futures, timeout=remaining)
		try:
			for future in done_iter:
				try:
					name, result = future.result()
					image[name] = result
				except Exception:
					model = futures[future]
					logger.error("model_failed", extra={"model": model.name}, exc_info=True)
		except TimeoutError:
			timed_out = [m.name for f, m in futures.items() if not f.done()]
			logger.warning("models_timed_out", extra={"models": timed_out, "budget_s": REQUEST_BUDGET})
			for f in futures:
				f.cancel()

	image["runtime"] = time.time() - start
	return image


# [START run_application]
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-url', nargs='?', default=None, required=True)
	parser.add_argument('-services', nargs='+', choices=['imagga', 'gv', 'mcs', 'clarifai', 'color', 'aws', 'hash'], default=['imagga', 'gv', 'mcs', 'clarifai', 'color', 'aws', 'hash'])
	args = parser.parse_args()
	main(args.url, args.services)
# [END run_application]
