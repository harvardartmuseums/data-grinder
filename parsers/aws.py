import config
import boto3

class AWS(object):

    def __init__(self):
        self.aws_key = config.AWS_ACCESS_KEY
        self.aws_secret = config.AWS_SECRET_ACCESS_KEY
        self.aws_region = 'us-east-1'
    
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