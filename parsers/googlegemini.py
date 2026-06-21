import os
import base64
import requests
from enum import Enum

class GoogleGeminiModel(Enum):
	FLASH_LITE_2_0 = (
		"gemini-2-0-flash-lite",
		"gemini-2.0-flash-lite",
		{"maxTokens": 2000, "temperature": 1.0, "topP": 0.5},
		"2026-06-01"
	)
	FLASH_LITE_2_5 = (
		"gemini-2-5-flash-lite",
		"gemini-2.5-flash-lite",
		{"maxTokens": 2000, "temperature": 1.0, "topP": 0.5},
		None
	)
	FLASH_LITE_3_1 = (
		"gemini-3-1-flash-lite",
		"gemini-3.1-flash-lite",
		{"maxTokens": 2000, "temperature": 1.0, "topP": 0.5},
		None
	)	
	FLASH_2_0 = (
		"gemini-2-0-flash",
		"gemini-2.0-flash",
		{"maxTokens": 2000, "temperature": 1.0, "topP": 0.5},
		"2026-06-01"
	)
	FLASH_2_5 = (
		"gemini-2-5-flash",
		"gemini-2.5-flash",
		{"maxTokens": 2000, "temperature": 1.0, "topP": 0.5},
		None
	)

	def __init__(self, name: str, model_id: str, inference_config: dict, eol_date: str):
		self._model_id = model_id
		self._name = name
		self._inference_config = inference_config
		self._eol_date = eol_date

	def list_models():
		return [
			{
				"name": model.name,
				"model_id": model.model_id,
				"eol_date": model.eol_date,
				"provider": model.provider
			}
			for model in GoogleGeminiModel
		]

	@property
	def model_id(self):
		return self._model_id    
	
	@property
	def name(self):
		return self._name    

	@property
	def inference_config(self):
		return self._inference_config

	@property
	def eol_date(self):
		return self._eol_date

	@property
	def provider(self):
		return "Google Gemini"

class GoogleGemini(object):

	def __init__(self):
		self.api_key = os.getenv("GOOGLE_API_KEY")
			  			  
	def fetch(self, photo_file, model: GoogleGeminiModel = GoogleGeminiModel.FLASH_LITE_2_0, prompt=None, connect_timeout=10, read_timeout=60):
		# Fetch the image data from the local copy
		with open(photo_file, 'rb') as image:
			image_content = base64.b64encode(image.read()).decode('utf-8')

		# Determine which model to call
		model_id = model.model_id

		try:
			url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={self.api_key}"

			params = {
				"contents": [
					{
						"parts": [
							{
								"text": prompt or "Describe this image"
							},
							{
								"inline_data": {
									"mime_type": "image/jpeg",
									"data": image_content
								}
							}
						]
					}
				]
			}
			headers = {
				'Content-Type': 'application/json'
			}
			response = requests.post(url, headers=headers, json=params, timeout=(connect_timeout, read_timeout))

			if response.status_code != 200:
				return {
					"body": None,
					"model": model_id,
					"provider": model.provider,
					"status": response.status_code,
					"description": response.text,
					"full": None
				}

			result = response.json()

			# Prompt blocked before generation (content policy violation)
			prompt_feedback = result.get("promptFeedback", {})
			if prompt_feedback.get("blockReason"):
				return {
					"body": None,
					"model": model_id,
					"provider": model.provider,
					"status": 200,
					"full": result,
					"content_policy_violation": True
				}

			candidates = result.get("candidates", [])
			if not candidates:
				return {
					"body": None,
					"model": model_id,
					"provider": model.provider,
					"status": 200,
					"description": "No candidates returned",
					"full": result
				}

			finish_reason = candidates[0].get("finishReason")

			# Response generated but withheld by safety filter
			if finish_reason == "SAFETY":
				return {
					"body": None,
					"model": model_id,
					"provider": model.provider,
					"status": 200,
					"full": result,
					"filtered": True
				}

			body = candidates[0]["content"]["parts"][0]["text"]
			out = {
				"body": body,
				"model": model_id,
				"provider": model.provider,
				"status": 200,
				"full": result
			}
			if finish_reason == "MAX_TOKENS":
				out["truncated"] = True
			return out

		except Exception as e:
			return {
				"body": None,
				"model": model.model_id,
				"provider": model.provider,
				"status": 500,
				"description": str(e),
				"full": None
			}