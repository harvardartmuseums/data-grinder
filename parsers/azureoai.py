from openai import AzureOpenAI, BadRequestError, APIConnectionError, RateLimitError, APIStatusError, ContentFilterFinishReasonError
import os
import base64
import httpx
from enum import Enum

class OpenAIModel(Enum):
	GPT_4 = (
		"gpt-4",
		"gpt-4-turbo-2024-04-09",
		{"maxTokens": 4096, "temperature": 1.0, "topP": 1.0},
		"2025-11-14"
	)
	GPT_4O = (
		"gpt-4o",
		"gpt-4o-2024-11-20",
		{"maxTokens": 4096, "temperature": 1.0, "topP": 1.0},
		None
	)
	GPT_4_1_MINI = (
		"gpt-4-1-mini",
		"gpt-4.1-mini-2025-04-14",
		{"maxTokens": 4096, "temperature": 1.0, "topP": 1.0},
		None
	)
	GPT_5_NANO = (
		"gpt-5-nano",
		"gpt-5-nano-2025-08-07",
		{"maxCompletionTokens": 4096, "temperature": 1.0, "topP": 1.0},
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
			for model in OpenAIModel
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
		return "Azure OpenAI Service"

_client = None

class AzureOAI(object):

	def __init__(self):
		self.api_key = os.getenv("AZURE_OPENAI_KEY")
		self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
		self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")

	def get_client(self, connect_timeout=10, read_timeout=60):
		global _client
		if _client is None:
			_client = AzureOpenAI(
				api_version=self.api_version,
				api_key=self.api_key,
				azure_endpoint=self.endpoint,
				timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
			)
		return _client

	def fetch(self, photo_file, model: OpenAIModel = OpenAIModel.GPT_4, prompt=None, connect_timeout=10, read_timeout=60):
		client = self.get_client(connect_timeout, read_timeout)

		with open(photo_file, 'rb') as image:
			image_content = base64.b64encode(image.read()).decode('utf-8')

		prompt =  [ 
				{ "role": "system", "content": "You are a helpful assistant." }, 
				{ "role": "user", "content": [  
					{ 
						"type": "text", 
						"text": prompt or "Describe this image:"
					},
					{ 
						"type": "image_url",
						"image_url": {
							"url": f"data:image/png;base64,{image_content}"
						}
					}
				] } 
			]
		try: 
			if ("maxTokens" in model.inference_config):
				response = client.chat.completions.create(
					model = model.model_id,
					messages = prompt,
					max_tokens = model.inference_config["maxTokens"]
				)
			elif ("maxCompletionTokens" in model.inference_config):
				response = client.chat.completions.create(
					model = model.model_id,
					messages = prompt,
					max_completion_tokens = model.inference_config["maxCompletionTokens"]
				)

			result = {
				"body": None,
				"model": response.model,
				"provider": model.provider,
				"status": 200,
				"full": response.model_dump()
			}

			finish_reason = response.choices[0].finish_reason
			if finish_reason == "content_filter":
				cfr = response.choices[0].content_filter_results
				result["body"] = f"<p>The response was filtered due to the prompt triggering Azure OpenAI's content management policy. The content filter results follow.</p><p>" + \
						f"hate: {cfr['hate']['severity']}<br/>" + \
						f"self-harm: {cfr['self_harm']['severity']}<br/>" + \
						f"sexual: {cfr['sexual']['severity']}<br/>" + \
						f"violence: {cfr['violence']['severity']}<br/></p>"
				result["filtered"] = True

			else:
				result["body"] = response.choices[0].message.content

			if finish_reason == "length":
				result["truncated"] = True
				
			return result

		except APIConnectionError as e:
			return {
				"body": None,
				"model": model.model_id,
				"provider": model.provider,
				"status": e.status_code,
				"description": str(e),
				"full": None
			}	

		except RateLimitError as e:
			# A 429 status code was received; we should back off a bit
			return {
				"body": None,
				"model": model.model_id,
				"provider": model.provider,
				"status": e.status_code,
				"description": str(e),
				"full": None
			}					

		except ContentFilterFinishReasonError as e:
			return {
				"body": None,
				"model": model.model_id,
				"provider": model.provider,
				"status": e.status_code,
				"description": "Content filter triggered: " + str(e),
				"full": None,
				"filtered": True
			}

		except BadRequestError as e:
			if "content_filter" in str(e) or "ResponsibleAIPolicyViolation" in str(e) or "content_policy_violation" in str(e):
				description = {
					"choices": [
						{
							"finish_reason": "content_filter",
							"message": {
								"content": e.body["message"]
							},
							**e.body.get("inner_error", {})
						}
					],
					"model": "unknown",
					"error": e.body
				}
				return {
					"body": None,
					"model": model.model_id,
					"provider": model.provider,
					"status": e.status_code,
					"description": description,
					"full": None,
					"content_policy_violation": True
				}
			else:
				return {
					"body": None,
					"model": model.model_id,
					"provider": model.provider,
					"status": e.status_code,
					"description": str(e),
					"full": None
				}

		except APIStatusError as e:
			return {
				"body": None,
				"model": model.model_id,
				"provider": model.provider,
				"status": e.status_code,
				"description": str(e),
				"full": None
			}
		
		except Exception as e:
			return {
				"body": None,
				"model": model.model_id,
				"provider": model.provider,
				"status": 500,
				"description": str(e),
				"full": None
			}
