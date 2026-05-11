import os
import boto3
from botocore.config import Config
import botocore.exceptions
from enum import Enum

class NovaModel(Enum):
	NOVA_LITE_1_0 = (
		"nova-lite-1-0",
		"amazon.nova-lite-v1:0",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		None
	)
	NOVA_PRO_1_0 = (
		"nova-pro-1-0",
		"amazon.nova-pro-v1:0",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		None
	)
	NOVA_LITE_2_0 = (
		"nova-lite-2-0",
		"us.amazon.nova-2-lite-v1:0",
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
				"eol_date": model.eol_date,
				"provider": model.provider
			}
			for model in NovaModel
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
		return "Amazon"

class AWSNova(object):

	def __init__(self):
		self.aws_key = os.getenv("AWS_ACCESS_KEY")
		self.aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
		self.aws_region = os.getenv("AWS_REGION")
	
	def get_client(self, connect_timeout=10, read_timeout=60):
		return boto3.client('bedrock-runtime',
							region_name=self.aws_region,
							aws_access_key_id=self.aws_key,
							aws_secret_access_key=self.aws_secret,
							config=Config(connect_timeout=connect_timeout, read_timeout=read_timeout))

	def fetch(self, photo_file, model: NovaModel = NovaModel.NOVA_LITE_1_0, prompt=None, connect_timeout=10, read_timeout=60):
		response = ""
		client = self.get_client(connect_timeout, read_timeout)

		with open(photo_file, 'rb') as image:
			image_content = image.read()

		# Invoke the model with the prompt and the encoded image
		messages = [
			{
				"role": "user",
				"content": [
					{
						"text": prompt or "Describe this image:",
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

			return {
				"body": awsresponse["output"]["message"]["content"][0]["text"],
				"model": model.model_id,
				"provider": model.provider,
				"status": 200,
				"full": awsresponse
			}
		
		except ( client.exceptions.AccessDeniedException, client.exceptions.ResourceNotFoundException, client.exceptions.ThrottlingException, client.exceptions.ModelTimeoutException, client.exceptions.InternalServerException, client.exceptions.ValidationException, client.exceptions.ModelNotReadyException, client.exceptions.ServiceQuotaExceededException) as e:
			response = {
				"body": None,
				"model": model.model_id,
				"provider": model.provider,
				"status": 400,
				"description": str(e),
				"full": None
			}
			return response
			
		except client.exceptions.ModelErrorException as e:
			response = {
				"body": None,
				"model": model.model_id,
				"provider": model.provider,
				"status": 500,
				"description": str(e),
				"full": None
			}
			return response

		except botocore.exceptions.BotoCoreError as e:
			return {
				"body": None,
				"model": model.model_id,
				"provider": model.provider,
				"status": 500,
				"description": str(e),
				"full": None
			}


