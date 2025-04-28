import requests
import os
import json

class Clarifai(object):

	def __init__(self):
		self.pat = os.getenv("CLARIFAI_PAT")
		
		self.user_id_classification = os.getenv("CLARIFAI_USER_ID_CLASSIFICATION")
		self.app_id_classification = os.getenv("CLARIFAI_APP_ID_CLASSIFICATION")
		self.model_id_classification = os.getenv("CLARIFAI_MODEL_ID_CLASSIFICATION")

		self.user_id_detection = os.getenv("CLARIFAI_USER_ID_DETECTION")
		self.app_id_detection = os.getenv("CLARIFAI_APP_ID_DETECTION")
		self.model_id_detection = os.getenv("CLARIFAI_MODEL_ID_DETECTION")
		
		self.user_id_caption = os.getenv("CLARIFAI_USER_ID_CAPTION")
		self.app_id_caption = os.getenv("CLARIFAI_APP_ID_CAPTION")
		self.model_id_caption = os.getenv("CLARIFAI_MODEL_ID_CAPTION")

		self.base_url = "https://api.clarifai.com"

	def __make_params(self, photo_file):
		return {
				"inputs": [
					{
					"data": {
						"image": {
						"url": photo_file
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
			url = f"{self.base_url}/v2/users/{self.user_id_classification}/apps/{self.app_id_classification}/models/{self.model_id_classification}/outputs"
			response = requests.post(url, 
							headers=self.__make_headers(), 
							data=json.dumps(self.__make_params(photo_file)))
			return response.json()

		except Exception as e:
			error =  json.loads(e.response.content)
			return {"status": 500, "error": response.json()}

	def fetch_objects(self, photo_file):
		try:
			url = f"{self.base_url}/v2/users/{self.user_id_detection}/apps/{self.app_id_detection}/models/{self.model_id_detection}/outputs"
			response = requests.post(url, 
							headers=self.__make_headers(), 
							data=json.dumps(self.__make_params(photo_file)))
			return response.json()

		except Exception as e:
			error =  json.loads(e.response.content)
			return {"status": 500, "error": response.json()}
	
	def fetch_caption(self, photo_file):
		try:
			url = f"{self.base_url}/v2/users/{self.user_id_caption}/apps/{self.app_id_caption}/models/{self.model_id_caption}/outputs"
			response = requests.post(url, 
							headers=self.__make_headers(), 
							data=json.dumps(self.__make_params(photo_file)))
			return response.json()

		except Exception as e:
			error =  json.loads(e.response.content)
			return {"status": 500, "error": response.json()}
			