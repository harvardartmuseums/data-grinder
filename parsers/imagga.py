import requests
import os
from enum import Enum

class ImaggaModel(Enum):
	BASE = (
		"imagga",
		"",
		["colors", "categories", "faces", "tags"]
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
			for model in ImaggaModel
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

class Imagga(object):

	def __init__(self):
		self.api_key = os.getenv("IMAGGA_KEY")
		self.api_secret = os.getenv("IMAGGA_SECRET")

	def fetch(self, photo_file):
		response = requests.get('https://api.imagga.com/v2/tags?image_url=%s' % photo_file, auth=(self.api_key, self.api_secret))
		response = response.json()

		response['model'] = 'unknown'

		return response

	def fetch_categories(self, photo_file):
		response = requests.get('https://api.imagga.com/v2/categories/personal_photos?image_url=%s' % photo_file, auth=(self.api_key, self.api_secret))
		response = response.json()
		
		response['model'] = 'personal_photos'

		return response

	def fetch_colors(self, photo_file):
		response = requests.get('https://api.imagga.com/v2/colors?image_url=%s' % photo_file, auth=(self.api_key, self.api_secret))
		response = response.json()
		
		response['model'] = 'unknown'

		return response
	
	def fetch_faces(self, photo_file):
		response = requests.get('https://api.imagga.com/v2/faces/detections?image_url=%s' % photo_file, auth=(self.api_key, self.api_secret))
		response = response.json()
		
		response['model'] = 'unknown'
		
		return response	