import os
import io
import boto3
from botocore.config import Config
import botocore.exceptions
from enum import Enum
from PIL import Image

class AnthropicModel(Enum):
	CLAUDE_3_HAIKU = (
		"claude-3-haiku",
		"anthropic.claude-3-haiku-20240307-v1:0",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		"2026-09-10",
		3800000
	)
	CLAUDE_4_5_HAIKU = (
		"claude-4-5-haiku",
		"us.anthropic.claude-haiku-4-5-20251001-v1:0",
		{"maxTokens": 2048, "temperature": 0.5},
		None,
		3800000
	)
	CLAUDE_3_OPUS = (
		"claude-3-opus",
		"us.anthropic.claude-3-opus-20240229-v1:0",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		"2026-01-30",
		None
	)
	CLAUDE_4_5_OPUS = (
		"claude-4-5-opus",
		"us.anthropic.claude-opus-4-5-20251101-v1:0",
		{"maxTokens": 2048, "temperature": 0.5},
		None,
		3800000
	)
	CLAUDE_3_5_SONNET = (
		"claude-3-5-sonnet",
		"anthropic.claude-3-5-sonnet-20240620-v1:0",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		"2026-03-01",
		None
	)
	CLAUDE_3_5_SONNET_2 = (
		"claude-3-5-sonnet-v-2", 
		"us.anthropic.claude-3-5-sonnet-20241022-v2:0",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		"2026-03-01",
		None
	)
	CLAUDE_3_7_SONNET = (
		"claude-3-7-sonnet",
		"us.anthropic.claude-3-7-sonnet-20250219-v1:0",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		"2026-04-28",
		3800000
	)
	CLAUDE_4_SONNET = (
		"claude-4-sonnet",
		"global.anthropic.claude-sonnet-4-20250514-v1:0",
		{"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
		"2026-10-14",
		3800000
	)
	CLAUDE_4_5_SONNET = (
		"claude-4-5-sonnet",
		"global.anthropic.claude-sonnet-4-5-20250929-v1:0",
		{"maxTokens": 2048, "temperature": 0.5},
		None,
		3800000
	)

	def __init__(self, name: str, model_id: str, inference_config: dict, eol_date: str, image_size_limit: int):
		self._model_id = model_id
		self._name = name
		self._inference_config = inference_config
		self._eol_date = eol_date
		self._image_size_limit = image_size_limit

	def list_models():
		return [
			{
				"name": model.name,
				"model_id": model.model_id,
				"eol_date": model.eol_date,
				"image_size_limit": model.image_size_limit,
				"provider": model.provider
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

	@property
	def image_size_limit(self):
		return self._image_size_limit

	@property
	def provider(self):
		return "Anthropic"

_client = None

class AWSAnthropic(object):

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

	def _read_image_bytes(self, photo_file, model: AnthropicModel):
		"""Read image file, resampling if necessary to fit within the model's image size limit.

		If the model has no image size limit or the raw bytes are already within the limit,
		returns raw bytes unchanged. Otherwise, iteratively reduces JPEG quality and then
		pixel dimensions until the encoded size is strictly under the limit.

		Raises:
			ValueError: If unable to resample image below the model's limit.
		"""
		limit = model.image_size_limit

		with open(photo_file, 'rb') as image:
			raw_bytes = image.read()

		if limit is None or len(raw_bytes) < limit:
			return raw_bytes

		# Raw bytes exceed the model's image size limit — resample
		img = Image.open(io.BytesIO(raw_bytes))
		quality = 95

		while True:
			buf = io.BytesIO()
			img.save(buf, format="JPEG", quality=quality, optimize=True)
			data = buf.getvalue()

			if len(data) < limit:
				return data

			# Reduce quality first
			if quality > 20:
				quality -= 5
				continue

			# Quality floor reached — shrink pixel dimensions by 10%
			width, height = img.size
			new_width = int(width * 0.9)
			new_height = int(height * 0.9)

			if new_width < 10 or new_height < 10:
				raise ValueError(
					f"Unable to resample image '{photo_file}' below {limit} bytes. "
					f"Original size: {len(raw_bytes)} bytes."
				)

			img = img.resize((new_width, new_height), Image.LANCZOS)

	def fetch(self, photo_file, model: AnthropicModel = AnthropicModel.CLAUDE_3_HAIKU, prompt=None, connect_timeout=10, read_timeout=60):
		response = ""
		client = self.get_client(connect_timeout, read_timeout)

		image_content = self._read_image_bytes(photo_file, model)

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

			body = "Content filtered. Blocked by Bedrock Guardrails" if awsresponse.get("stopReason") == "content_filtered" else awsresponse["output"]["message"]["content"][0]["text"]
			return {
				"body": body,
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
