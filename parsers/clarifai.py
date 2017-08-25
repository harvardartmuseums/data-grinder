import config
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage

class Clarifai(object):

	def fetch(self, photo_file, id):

		clarifai_api = ClarifaiApp(api_key = config.CLARIFAI_API_KEY)
		result = clarifai_api.models.get('general-v1.3').predict_by_url(photo_file)

		return result