import requests
import os
import json

class Clarifai(object):

	def __init__(self):
		self.user_id = os.getenv("CLARIFAI_USER_ID")
		self.pat = os.getenv("CLARIFAI_PAT")
		self.app_id = os.getenv("CLARIFAI_APP_ID")
		self.model_id = os.getenv("CLARIFAI_MODEL_ID")
		self.base_url = "https://api.clarifai.com"

	def fetch(self, photo_file, id):
		try:
			url = self.base_url + "/v2/users/" + self.user_id + "/apps/" + self.app_id + "/models/" + self.model_id + "/outputs"

			params = json.dumps({
				"inputs": [
					{
					"data": {
						"image": {
						"url": photo_file
						}
					}
					}
				]
			})
			headers = {
				'Authorization': 'Key ' + self.pat,
				'Content-Type': 'application/json'
			}
			response = requests.post(url, headers=headers, data=params)

		except Exception as e:
			error =  json.loads(e.response.content)
			print(error)
			
		return response.json()