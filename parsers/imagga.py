import requests
import os
from enum import Enum

class ImaggaModel(Enum):
	BASE = (
		"imagga",
		"",
		["colors", "categories", "faces", "tags", "structured-tags"],
		None
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

	@property
	def provider(self):
		return "Imagga"
	
	@property
	def eol_date(self):
		return self._eol_date

class Imagga(object):

	def __init__(self):
		self.api_key = os.getenv("IMAGGA_KEY")
		self.api_secret = os.getenv("IMAGGA_SECRET")

	def __make_files(self, photo_file):
		with open(photo_file, 'rb') as image:
			image_content = image.read()

		return {'image': image_content}

	def fetch(self, photo_file):
		response = requests.post('https://api.imagga.com/v2/tags', files=self.__make_files(photo_file), auth=(self.api_key, self.api_secret))
		response = response.json()

		response['model'] = 'unknown'
		response['provider'] = ImaggaModel.BASE.provider

		return response
	
	def fetch_structured_tags(self, photo_file):
		params = {
			"model": "light",
			"include_caption": True
		}
		response = requests.post('https://api.imagga.com/v3/tags', params=params, files=self.__make_files(photo_file), auth=(self.api_key, self.api_secret))
		response = response.json()
		
		response['model'] = 'pro'
		response['provider'] = ImaggaModel.BASE.provider

		return response

	def fetch_categories(self, photo_file):
		response = requests.post('https://api.imagga.com/v2/categories/personal_photos', files=self.__make_files(photo_file), auth=(self.api_key, self.api_secret))
		response = response.json()
		
		response['model'] = 'personal_photos'
		response['provider'] = ImaggaModel.BASE.provider

		return response

	def fetch_colors(self, photo_file):
		response = requests.post('https://api.imagga.com/v2/colors', files=self.__make_files(photo_file), auth=(self.api_key, self.api_secret))
		response = response.json()
		
		response['model'] = 'unknown'
		response['provider'] = ImaggaModel.BASE.provider

		return response
	
	def fetch_faces(self, photo_file):
		response = requests.post('https://api.imagga.com/v2/faces/detections', files=self.__make_files(photo_file), auth=(self.api_key, self.api_secret))
		response = response.json()
		
		response['model'] = 'unknown'
		response['provider'] = ImaggaModel.BASE.provider
		
		return response	