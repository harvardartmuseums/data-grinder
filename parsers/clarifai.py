import os
import json
from clarifai.rest import ClarifaiApp

class Clarifai(object):

	def fetch(self, photo_file, id):
		try:
			clarifai_api = ClarifaiApp(api_key = os.getenv('CLARIFAI_API_KEY'))
			result = clarifai_api.models.get('general-v1.3').predict_by_url(photo_file)
		except Exception as e:
			print(e)
			
		return result