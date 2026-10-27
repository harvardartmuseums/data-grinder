import os
import boto3
from botocore.config import Config
import botocore.exceptions
from enum import Enum

class MoonshotModel(Enum):
	KIMI_K_2_5 = (
		"kimi-k2-5",
		"moonshotai.kimi-k2.5",
		{"maxTokens": 2000, "temperature": 1.0, "topP": 1.0},
		"2027-04-25"
	)
	KIMI_K_3 = (
		"kimi-k3",
		"us.moonshotai.kimi-k3",
		{"maxTokens": 2000},
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
			for model in MoonshotModel
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
		return "Moonshot AI"

_client = None

class AWSMoonshot(object):

	def __init__(self):
		self.aws_key = os.getenv("AWS_ACCESS_KEY")
		self.aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
		self.aws_region = os.getenv("AWS_REGION")

	def get_client(self, connect_timeout=10, read_timeout=60):
		global _client
		if _client is None:
			_client = boto3.client('bedrock-runtime',
								region_name=self.aws_region,
								aws_access_key_id=self.aws_key,
								aws_secret_access_key=self.aws_secret,
								config=Config(connect_timeout=connect_timeout, read_timeout=read_timeout))
		return _client

	def _extract_text(self, content):
		"""Join the text blocks of a Converse response, in order.

		Models that reason return a reasoningContent block ahead of the text block
		(boto3 1.35 surfaces it as SDK_UNKNOWN_MEMBER), so the text is not always
		content[0]. Returns None when the response carries no text block at all.
		"""
		text = "\n\n".join(block["text"] for block in content if "text" in block)
		return text or None

	def fetch(self, photo_file, model: MoonshotModel = MoonshotModel.KIMI_K_2_5, prompt=None, connect_timeout=10, read_timeout=60):
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

			stop_reason = awsresponse.get("stopReason")

			if stop_reason == "content_filtered":
				return {
					"body": None,
					"model": model.model_id,
					"provider": model.provider,
					"status": 200,
					"full": awsresponse,
					"filtered": True
				}

			body = self._extract_text(awsresponse["output"]["message"]["content"])

			result = {
				"body": body,
				"model": model.model_id,
				"provider": model.provider,
				"status": 200,
				"full": awsresponse
			}
			if body is None:
				result["description"] = "response contained no text content block"
			if stop_reason == "max_tokens":
				result["truncated"] = True
			return result
		
		except ( client.exceptions.AccessDeniedException, client.exceptions.ResourceNotFoundException, client.exceptions.ThrottlingException, client.exceptions.ModelTimeoutException, client.exceptions.InternalServerException, client.exceptions.ValidationException, client.exceptions.ModelNotReadyException, client.exceptions.ServiceQuotaExceededException, client.exceptions.ServiceUnavailableException) as e:
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


