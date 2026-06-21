import requests
import os
import base64
import json
from enum import Enum

class SalesForceModel(Enum):
	BLIP = (
		"blip",
		"general-english-image-caption-blip",
		{},
		"2025-12-31"
	)
	BLIP_2 = (
		"blip-2",
		"general-english-image-caption-blip-2",
		{},
		"2026-07-16"
	)
	BLIP_2_6_7B = (
		"blip-2-6-7B",
		"general-english-image-caption-blip-2-6_7B",
		{},
		"2025-12-31"
	)

	def __init__(self, name: str, model_id: str, inference_config: dict, eol_date: str):
		self._model_id = model_id
		self._name = name
		self._inference_config = inference_config
		self._eol_date = eol_date

	def list_models():
		return [
			{
				"name": model.name,
				"model_id": model.model_id,
				"eol_date": model.eol_date,
				"provider": model.provider
			}
			for model in SalesForceModel
		]
	
	@property
	def model_id(self):
		return self._model_id

	@property
	def name(self):
		return self._name

	@property
	def inference_config(self):
		return self._inference_config

	@property
	def eol_date(self):
		return self._eol_date

	@property
	def provider(self):
		return "Salesforce"

class SalesForce(object):

	def __init__(self):
		self.pat = os.getenv("CLARIFAI_PAT")
		
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
	
	def fetch(self, photo_file, model: SalesForceModel = SalesForceModel.BLIP_2, prompt=None, connect_timeout=10, read_timeout=60):
		try:
			url = f"https://api.clarifai.com/v2/users/salesforce/apps/blip/models/{model.model_id}/outputs"
			response = requests.post(url,
							headers=self.__make_headers(),
							data=json.dumps(self.__make_params(photo_file)),
							timeout=(connect_timeout, read_timeout))
			result = response.json()

			return {
				"body": result["outputs"][0]["data"]["text"]["raw"],
				"model": result["outputs"][0]["model"]["id"],
				"provider": model.provider,
				"status": 200,
				"full": result
			}

		except Exception as e:
			response = {
				"body": None,
				"model": model.model_id,
				"provider": model.provider,
				"status": 500,
				"description": str(e),
				"full": None
			}			
			return response