from openai import AzureOpenAI, BadRequestError, APIConnectionError, RateLimitError, APIStatusError, ContentFilterFinishReasonError
import os
import base64
from enum import Enum

class OpenAIModel(Enum):
	OPENAI = (
		"openai",
		"HAM-GPT-4-V-D1",
		{"maxTokens": 2000, "temperature": 1.0, "topP": 1.0},
		None
	)        
	GPT_4 = (
		"gpt-4",
		"HAM-GPT-4-V-D1",
		{"maxTokens": 2000, "temperature": 1.0, "topP": 1.0},
		None
	)    
	GPT_4O = (
		"gpt-4o",
		"HAM-GPT-4o-V-D1",
		{"maxTokens": 2000, "temperature": 1.0, "topP": 1.0},
		None
	)
	GPT_4_1_MINI = (
		"gpt-4-1-mini",
		"HAM-GPT-4-1-MINI-D1",
		{"maxTokens": 2000, "temperature": 1.0, "topP": 1.0},
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

class AzureOAI(object):

	def __init__(self):
		self.api_key = os.getenv("AZURE_OPENAI_KEY")
		self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
		self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")

	def fetch(self, photo_file, model: OpenAIModel = OpenAIModel.GPT_4):
		client = AzureOpenAI(
			api_version = self.api_version,
			api_key = self.api_key,
			azure_endpoint = self.endpoint
		)

		with open(photo_file, 'rb') as image:
			image_content = base64.b64encode(image.read()).decode('utf-8')

		prompt =  [ 
				{ "role": "system", "content": "You are a helpful assistant." }, 
				{ "role": "user", "content": [  
					{ 
						"type": "text", 
						"text": "Describe this image:" 
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
			response = client.chat.completions.create(
				model = model.model_id,
				messages = prompt,
				max_tokens = model.inference_config["maxTokens"]
			)
			
			result = {
				"description": response.model_dump(),
				"prompt": "",
				"status": 200
			}

			return result

		except APIConnectionError as e:
			return {
				"model": model.model_id,
				"status": e.status_code,
				"description": str(e),
				"prompt": prompt
			}	

		except RateLimitError as e:
			# A 429 status code was received; we should back off a bit
			return {
				"model": model.model_id,
				"status": e.status_code,
				"description": str(e),
				"prompt": prompt
			}					

		except ContentFilterFinishReasonError as e:
			response = {
				"model": model.model_id,
				"status": e.status_code,
				"description": "Content filter triggered: " + str(e),
				"prompt": prompt
			}
			return response

		except BadRequestError as e:
			if "content_filter" in str(e) or "ResponsibleAIPolicyViolation" in str(e):
				description = {
					"choices": [
						{
							"finish_reason": "content_filter",
							"message": {
								"content": e.body["message"]
							},
							**e.body["inner_error"]
						}
					],
					"model": "unknown",
					"error": e.body
				}
			else:
				description = str(e)

			return {
				"status": e.status_code,
				"description": description,
				"prompt": prompt
			}		

		except APIStatusError as e:
			result = {
				"status": e.status_code,
				"description": str(e),
				"prompt": prompt
			}
			return result
		
		except Exception as e:
			response = {
				"status": 500,
				"description": str(e)
			}
			return response
