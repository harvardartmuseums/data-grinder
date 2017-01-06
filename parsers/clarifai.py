import config
from clarifai.rest import ClarifaiApp

class Clarifai(object):

	def fetch(self, photo_file, id):

		clarifai_api = ClarifaiApp(config.CLARIFAI_APP_ID, config.CLARIFAI_APP_SECRET)
		result = clarifai_api.models.get('general-v1.3').predict_by_url(photo_file)

		return result