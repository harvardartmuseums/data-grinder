import base64
import os
from googleapiclient import discovery
from enum import Enum

class GVisionModel(Enum):
	BASE = (
		"gv",
		"",
		[]
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
			for model in GVisionModel
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

class Vision(object):

	def __init__(self):
		self.api_key = os.getenv("GOOGLE_API_KEY")

	def fetch(self, photo_file):
		# The url template to retrieve the discovery document for trusted testers.
		DISCOVERY_URL='https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'

		MAX_RESULTS = 20

		"""Run a label request on a single image"""

		service = discovery.build('vision', 'v1', cache_discovery=False, 
								  developerKey=self.api_key,
								  discoveryServiceUrl=DISCOVERY_URL)
		# [END authenticate]

		# [START construct_request]
		with open(photo_file, 'rb') as image:
			image_content = base64.b64encode(image.read())
			service_request = service.images().annotate(body={
				'requests': [{
					'image': {
						'content': image_content.decode('UTF-8')
					},
					'features': [{
							'type': 'LABEL_DETECTION',
							'maxResults': MAX_RESULTS
						},
						{
							'type': 'FACE_DETECTION',
							'maxResults': MAX_RESULTS
						},
						{
							'type': 'TEXT_DETECTION',
							'maxResults': MAX_RESULTS
						},
						{
							'type': 'LANDMARK_DETECTION',
							'maxResults': MAX_RESULTS
						}
					]
				}]
			})
			# [END construct_request]
			# [START parse_response]
			response = service_request.execute()
			response["model"] = "unknown"
			return response