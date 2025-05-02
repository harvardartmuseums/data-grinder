import os
import boto3
from enum import Enum

class AWSModel(Enum):
	BASE = (
		"aws",
		"",
		["faces", "labels", "text"]
	)

	def __init__(self, name: str, model_id: str, functions: list):
		self._model_id = model_id
		self._name = name
		self._functions = functions

	def list_models():
		return [
			{
				"name": model.name,
				"model_id": model.model_id,
				"functions": model.functions
			}
			for model in AWSModel
		]

	@property
	def model_id(self):
		return self._model_id    
	
	@property
	def name(self):
		return self._name   
	
	@property
	def functions(self):
		return self._functions

class AWS(object):

    def __init__(self):
        self.aws_key = os.getenv("AWS_ACCESS_KEY")
        self.aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION")
    
    def get_client(self):
        return boto3.client('rekognition', region_name=self.aws_region, aws_access_key_id=self.aws_key, aws_secret_access_key=self.aws_secret)

    def fetch_labels(self, photo_file):
        client = self.get_client()

        with open(photo_file, 'rb') as image:
            response = client.detect_labels(Image={'Bytes': image.read()})

        return response

    def fetch_faces(self, photo_file):
        client = self.get_client()

        with open(photo_file, 'rb') as image:
            response = client.detect_faces(Image={'Bytes': image.read()}, Attributes=['ALL'])

        return response

    def fetch_text(self, photo_file):
        client = self.get_client()

        with open(photo_file, 'rb') as image:
            response = client.detect_text(Image={'Bytes': image.read()})

        return response