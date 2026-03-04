import os
import io
import boto3
from enum import Enum
from botocore.exceptions import ClientError
from PIL import Image

# AWS Rekognition has a 5 MB limit for inline image bytes
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5242880 bytes

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

    def _read_image_bytes(self, photo_file):
        """Read image file, resampling if necessary to fit within AWS limit.
        
        If file is under the limit, returns raw bytes.
        If file exceeds limit, iteratively reduces quality and resolution until it fits.
        
        Raises:
			ValueError: If unable to resample image below the limit.
        """
        size = os.path.getsize(photo_file)
        if size <= MAX_IMAGE_BYTES:
            # File is already within limits, return as-is
            with open(photo_file, 'rb') as image:
                return image.read()
        
        # File is too large, need to resample
        # Open image and begin resampling
        img = Image.open(photo_file)
        quality = 95
        
        while True:
            # Try saving at current quality level
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            data = buf.getvalue()
            
            if len(data) <= MAX_IMAGE_BYTES:
                return data
            
            # Quality too high, reduce it
            if quality > 20:
                quality -= 5
                continue
            
            # Quality is low, reduce image dimensions
            width, height = img.size
            new_width = int(width * 0.9)
            new_height = int(height * 0.9)
            
            # Stop if image becomes too small
            if new_width < 10 or new_height < 10:
                raise ValueError(
                    f"Unable to resample image '{photo_file}' below {MAX_IMAGE_BYTES} bytes. "
                    f"Original size: {size} bytes."
                )
            
            img = img.resize((new_width, new_height), Image.LANCZOS)

    def fetch_labels(self, photo_file):
        client = self.get_client()
        image_bytes = self._read_image_bytes(photo_file)

        try:
            response = client.detect_labels(Image={'Bytes': image_bytes})
        except ClientError as e:
            if "length less than or equal to" in str(e):
                raise ValueError(
                    f"AWS Rekognition rejected image '{photo_file}' - "
                    "file exceeds 5MB inline limit"
                ) from e
            raise

        return response

    def fetch_faces(self, photo_file):
        client = self.get_client()
        image_bytes = self._read_image_bytes(photo_file)

        try:
            response = client.detect_faces(Image={'Bytes': image_bytes}, Attributes=['ALL'])
        except ClientError as e:
            if "length less than or equal to" in str(e):
                raise ValueError(
                    f"AWS Rekognition rejected image '{photo_file}' - "
                    "file exceeds 5MB inline limit"
                ) from e
            raise

        return response

    def fetch_text(self, photo_file):
        client = self.get_client()
        image_bytes = self._read_image_bytes(photo_file)

        try:
            response = client.detect_text(Image={'Bytes': image_bytes})
        except ClientError as e:
            if "length less than or equal to" in str(e):
                raise ValueError(
                    f"AWS Rekognition rejected image '{photo_file}' - "
                    "file exceeds 5MB inline limit"
                ) from e
            raise

        return response