import os
import boto3
from enum import Enum

class MetaModel(Enum):
	LLAMA_3_2_11B = (
		"llama-3-2-11b",
		"us.meta.llama3-2-11b-instruct-v1:0"
	)
	LLAMA_3_2_90B = (
		"llama-3-2-90b",
		"us.meta.llama3-2-90b-instruct-v1:0"
	)

	def __init__(self, name: str, model_id: str):
		self._model_id = model_id
		self._name = name

	def list_models():
		return [
			{
				"name": model.name,
				"model_id": model.model_id
			}
			for model in MetaModel
		]

	@property
	def model_id(self):
		return self._model_id    
	
	@property
	def name(self):
		return self._name    
	
class AWSMeta(object):

	def __init__(self):
		self.aws_key = os.getenv("AWS_ACCESS_KEY")
		self.aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
		self.aws_region = os.getenv("AWS_REGION")
	
	def get_client(self):
		return boto3.client('bedrock-runtime', 
							region_name=self.aws_region, 
							aws_access_key_id=self.aws_key, 
							aws_secret_access_key=self.aws_secret)

	def fetch(self, photo_file, model: MetaModel = MetaModel.LLAMA_3_2_11B):
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
				inferenceConfig={"maxTokens": 2048, "temperature": 0.5, "topP": 0.9}
			)

			# Process and print the response
			response = awsresponse
			response['model'] = model.model_id
			response['status'] = 200
			return response
		
		except ( client.exceptions.AccessDeniedException, client.exceptions.ResourceNotFoundException, client.exceptions.ThrottlingException, client.exceptions.ModelTimeoutException, client.exceptions.InternalServerException, client.exceptions.ValidationException, client.exceptions.ModelNotReadyException, client.exceptions.ServiceQuotaExceededException) as e:
			print(e)
			response = {"status": 400}
			return response
			
		except client.exceptions.ModelErrorException as e:
			response = {"status": 500}
			return response


