import os
import base64
import requests
from enum import Enum

class GoogleGeminiModel(Enum):
	FLASH_LITE_2_0 = (
		"gemini-2-0-flash-lite",
		"gemini-2.0-flash-lite"
	)
	FLASH_2_0 = (
		"gemini-2-0-flash",
		"gemini-2.0-flash"
	)

	def __init__(self, name: str, model_id: str):
		self._model_id = model_id
		self._name = name

	def list_models():
		return [
			{
				"name": model.name,
				"model_id": model.model_id
			}
			for model in GoogleGeminiModel
		]

	@property
	def model_id(self):
		return self._model_id    
	
	@property
	def name(self):
		return self._name    
	
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
			response = {"status": 500}
			return response