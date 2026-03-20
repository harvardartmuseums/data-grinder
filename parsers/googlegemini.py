import os
import base64
import requests
from enum import Enum

class GoogleGeminiModel(Enum):
	FLASH_LITE_2_0 = (
		"gemini-2-0-flash-lite",
		"gemini-2.0-flash-lite",
		{"maxTokens": 2000, "temperature": 1.0, "topP": 0.5},
		None
	)
	FLASH_LITE_2_5 = (
		"gemini-2-5-flash-lite",
		"gemini-2.5-flash-lite",
		{"maxTokens": 2000, "temperature": 1.0, "topP": 0.5},
		None
	)
	FLASH_2_0 = (
		"gemini-2-0-flash",
		"gemini-2.0-flash",
		{"maxTokens": 2000, "temperature": 1.0, "topP": 0.5},
		None
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
				"eol_date": model.eol_date
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
	
class GoogleGemini(object):

	def __init__(self):
		self.api_key = os.getenv("GOOGLE_API_KEY")
			  			  
	def fetch(self, photo_file, model: GoogleGeminiModel = GoogleGeminiModel.FLASH_LITE_2_0):
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
								"text": "Describe this image"
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
			response = requests.post(url, headers=headers, json=params)

			result = response.json()
			response = result
			response['model'] = model_id
			response['status'] = 200			
			return response

		except Exception as e:
			response = {
				"model": model.model_id,
				"status": 500,
				"description": str(e)
			}
			return response