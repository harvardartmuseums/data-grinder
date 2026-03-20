import os
import boto3
from enum import Enum

class AnthropicModel(Enum):
	CLAUDE = (
		"claude",
		"anthropic.claude-3-haiku-20240307-v1:0",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		"2026-09-10"
	)    
	CLAUDE_3_HAIKU = (
		"claude-3-haiku",
		"anthropic.claude-3-haiku-20240307-v1:0",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		"2026-09-10"
	)
	CLAUDE_4_5_HAIKU = (
		"claude-4-5-haiku",
		"us.anthropic.claude-haiku-4-5-20251001-v1:0",
		{"maxTokens": 2048, "temperature": 0.5},
		None
	)
	CLAUDE_3_OPUS = (
		"claude-3-opus",
		"us.anthropic.claude-3-opus-20240229-v1:0",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		"2026-01-30"
	)
	CLAUDE_4_1_OPUS = (
		"claude-4-1-opus",
		"us.anthropic.claude-opus-4-1-20250805-v1:0",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		None
	)
	CLAUDE_4_5_OPUS = (
		"claude-4-5-opus",
		"us.anthropic.claude-opus-4-5-20251101-v1:0",
		{"maxTokens": 2048, "temperature": 0.5},
		None
	)
	CLAUDE_3_5_SONNET = (
		"claude-3-5-sonnet",
		"anthropic.claude-3-5-sonnet-20240620-v1:0",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		"2026-03-01"
	)
	CLAUDE_3_5_SONNET_2 = (
		"claude-3-5-sonnet-v-2", 
		"us.anthropic.claude-3-5-sonnet-20241022-v2:0",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		"2026-03-01"
	)
	CLAUDE_3_7_SONNET = (
		"claude-3-7-sonnet",
		"us.anthropic.claude-3-7-sonnet-20250219-v1:0",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		"2026-04-28"
	)
	CLAUDE_4_SONNET = (
		"claude-4-sonnet",
		"global.anthropic.claude-sonnet-4-20250514-v1:0",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		None
	)
	CLAUDE_4_5_SONNET = (
		"claude-4-5-sonnet",
		"global.anthropic.claude-sonnet-4-5-20250929-v1:0",
		{"maxTokens": 2048, "temperature": 0.5},
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
			for model in AnthropicModel
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

class AWSAnthropic(object):

	def __init__(self):
		self.aws_key = os.getenv("AWS_ACCESS_KEY")
		self.aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
		self.aws_region = os.getenv("AWS_REGION")
	
	def get_client(self):
		return boto3.client('bedrock-runtime', 
							region_name=self.aws_region, 
							aws_access_key_id=self.aws_key, 
							aws_secret_access_key=self.aws_secret)

	def fetch(self, photo_file, model: AnthropicModel = AnthropicModel.CLAUDE_3_HAIKU):
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


