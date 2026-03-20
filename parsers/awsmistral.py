import os
import boto3
from enum import Enum

class MistralModel(Enum):
	PIXTRAL_LARGE_2502 = (
		"pixtral-large-2502",
		"us.mistral.pixtral-large-2502-v1:0",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		None
	)
	MAGISTRAL_SMALL_2509 = (
		"magistral-small-2509",
		"mistral.magistral-small-2509",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		None
	)
	MINISTRAL_3_3B = (
		"ministral-3-3b",
		"mistral.ministral-3-3b-instruct",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		None
	)
	MINISTRAL_3_8B = (
		"ministral-3-8b",
		"mistral.ministral-3-8b-instruct",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		None
	)
	MINISTRAL_3_14B = (
		"ministral-3-14b",
		"mistral.ministral-3-14b-instruct",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		None
	)
	MISTRAL_LARGE_3_675B = (
		"mistral-large-3-675b",
		"mistral.mistral-large-3-675b-instruct",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		None
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
				"eol_date": model.eol_date
			}
			for model in MistralModel
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
	
class AWSMistral(object):

	def __init__(self):
		self.aws_key = os.getenv("AWS_ACCESS_KEY")
		self.aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
		self.aws_region = os.getenv("AWS_REGION")
	
	def get_client(self):
		return boto3.client('bedrock-runtime', 
							region_name=self.aws_region, 
							aws_access_key_id=self.aws_key, 
							aws_secret_access_key=self.aws_secret)

	def fetch(self, photo_file, model: MistralModel = MistralModel.PIXTRAL_LARGE_2502):
		response = ""
		client = self.get_client()

		with open(photo_file, 'rb') as image:
			image_content = image.read()

		# Invoke the model with the prompt and the encoded image
		messages = [
			{
				"role": "user",
				"content": [
					{
						"text": "Describe this image:",
					},
					{
						"image": {
							"format": "jpeg",
							"source": {
								"bytes": image_content,
							}
						},
					},
				],
			}
		]
		
		try: 
			awsresponse = client.converse(
				modelId=model.model_id,
				messages=messages,
				inferenceConfig=model.inference_config
			)

			# Process and print the response
			response = awsresponse
			response['model'] = model.model_id
			response['status'] = 200
			return response
		
		except ( client.exceptions.AccessDeniedException, client.exceptions.ResourceNotFoundException, client.exceptions.ThrottlingException, client.exceptions.ModelTimeoutException, client.exceptions.InternalServerException, client.exceptions.ValidationException, client.exceptions.ModelNotReadyException, client.exceptions.ServiceQuotaExceededException) as e:
			response = {
				"model": model.model_id,
				"status": 400,
				"description": str(e)
			}
			return response
			
		except client.exceptions.ModelErrorException as e:
			response = {
				"model": model.model_id,
				"status": 500,
				"description": str(e)
			}
			return response


