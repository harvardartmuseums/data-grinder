import requests
import os
from enum import Enum

class MCSVisionModel(Enum):
	BASE = (
		"mcs",
		"",
		["analyze", "describe"]
	)

	def __init__(self, name: str, model_id: str, functions: list):
		self._model_id = model_id
		self._name = name
		self._functions = functions

	def list_models():
		return [
			{
				"name": model.name,
				"model_id": model.model_id,
				"functions": model.functions
			}
			for model in MCSVisionModel
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
	
class MCSVision(object):

	def __init__(self):
		self.api_key = os.getenv("MICROSOFT_CS_KEY")
		self.endpoint = os.getenv("MICROSOFT_CS_ENDPOINT")
		self.api_version = os.getenv("MICROSOFT_CS_API_VERSION")

		self.base_url = self.endpoint + 'vision/' + self.api_version + '/'

	def fetch_analyze(self, photo_file):
		url = self.base_url + 'analyze'

		params = {'visualFeatures': 'Categories,Tags,Description,Faces,Color,Objects'}
		headers = {
			'Ocp-Apim-Subscription-Key': self.api_key,
			'Content-Type': 'application/octet-stream'
		}

		with open(photo_file, 'rb') as image:
			image_content = image.read()
			response = requests.post(url, data=image_content, params=params, headers=headers)
			return response.json()				

	def fetch_description(self, photo_file):
		url = self.base_url + 'describe'

		params = {'maxCandidates': 3}
		headers = {
			'Ocp-Apim-Subscription-Key': self.api_key,
			'Content-Type': 'application/octet-stream'
		}

		with open(photo_file, 'rb') as image:
			image_content = image.read()
			response = requests.post(url, data=image_content, params=params, headers=headers)
			return response.json()				