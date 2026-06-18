import requests
import os
import base64
import json
from enum import Enum
from requests.exceptions import Timeout, ConnectionError, RequestException

class ClarifaiModel(Enum):
	BASE = (
		"clarifai",
		"",
		["classification", "colors", "objects"],
		"2026-07-16"
	)

	def __init__(self, name: str, model_id: str, functions: list, eol_date: str):
		self._model_id = model_id
		self._name = name
		self._functions = functions
		self._eol_date = eol_date

	def list_models():
		return [
			{
				"name": model.name,
				"model_id": model.model_id,
				"functions": model.functions,
				"provider": model.provider,
				"eol_date": model.eol_date
			}
			for model in ClarifaiModel
		]

	@property
	def model_id(self):
		return self._model_id    
	
	@property
	def name(self):
		return self._name   
	
	@property
	def functions(self):
		return self._functions

	@property
	def provider(self):
		return "Clarifai"

	@property
	def eol_date(self):
		return self._eol_date

class Clarifai(object):

	def __init__(self):
		self.pat = os.getenv("CLARIFAI_PAT")
				
		self.base_url = "https://api.clarifai.com"

	def __make_params(self, photo_file):
		with open(photo_file, 'rb') as image:
			image_content = base64.b64encode(image.read()).decode('utf-8')

		return {
				"inputs": [
					{
						"data": {
							"image": {
								"base64": image_content
							}
						}
					}
				]
			}
	
	def __make_headers(self):
		return {
				'Authorization': 'Key ' + self.pat,
				'Content-Type': 'application/json'
			}

	def fetch(self, photo_file):
		try:
			url = f"{self.base_url}/v2/users/clarifai/apps/main/models/general-image-recognition/outputs"
			response = requests.post(url, 
							headers=self.__make_headers(), 
							data=json.dumps(self.__make_params(photo_file)),
							timeout=30)
			result = response.json()
			result['provider'] = ClarifaiModel.BASE.provider
			return result

		except Timeout:
			return {"status": 504, "error": "Request to Clarifai API timed out", "provider": ClarifaiModel.BASE.provider}

		except ConnectionError as e:
			return {"status": 503, "error": f"Connection error: {str(e)}", "provider": ClarifaiModel.BASE.provider}

		except RequestException as e:
			return {"status": 500, "error": f"Request error: {str(e)}", "provider": ClarifaiModel.BASE.provider}

		except Exception as e:
			return {"status": 500, "error": f"Unexpected error: {str(e)}", "provider": ClarifaiModel.BASE.provider}

	def fetch_objects(self, photo_file):
		try:
			url = f"{self.base_url}/v2/users/clarifai/apps/main/models/general-image-detection/outputs"
			response = requests.post(url, 
							headers=self.__make_headers(), 
							data=json.dumps(self.__make_params(photo_file)),
							timeout=30)
			result = response.json()
			result['provider'] = ClarifaiModel.BASE.provider
			return result

		except Timeout:
			return {"status": 504, "error": "Request to Clarifai API timed out", "provider": ClarifaiModel.BASE.provider}

		except ConnectionError as e:
			return {"status": 503, "error": f"Connection error: {str(e)}", "provider": ClarifaiModel.BASE.provider}

		except RequestException as e:
			return {"status": 500, "error": f"Request error: {str(e)}", "provider": ClarifaiModel.BASE.provider}

		except Exception as e:
			return {"status": 500, "error": f"Unexpected error: {str(e)}", "provider": ClarifaiModel.BASE.provider}

	def fetch_colors(self, photo_file):
		try:
			url = f"{self.base_url}/v2/users/clarifai/apps/main/models/color-recognition/outputs"
			response = requests.post(url, 
							headers=self.__make_headers(), 
							data=json.dumps(self.__make_params(photo_file)),
							timeout=30)
			result = response.json()
			result['provider'] = ClarifaiModel.BASE.provider
			return result

		except Timeout:
			return {"status": 504, "error": "Request to Clarifai API timed out", "provider": ClarifaiModel.BASE.provider}

		except ConnectionError as e:
			return {"status": 503, "error": f"Connection error: {str(e)}", "provider": ClarifaiModel.BASE.provider}

		except RequestException as e:
			return {"status": 500, "error": f"Request error: {str(e)}", "provider": ClarifaiModel.BASE.provider}

		except Exception as e:
			return {"status": 500, "error": f"Unexpected error: {str(e)}", "provider": ClarifaiModel.BASE.provider}
