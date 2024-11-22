import base64
import os
import boto3
import json

class AWSAnthropic(object):

    def __init__(self):
        self.aws_key = os.getenv("AWS_ACCESS_KEY")
        self.aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION")
    
    def get_client(self):
        return boto3.client('bedrock-runtime', region_name=self.aws_region, aws_access_key_id=self.aws_key, aws_secret_access_key=self.aws_secret)

    def fetch(self, photo_file, model="haiku"):
        response = ""
        client = self.get_client()

        with open(photo_file, 'rb') as image:
            image_content = base64.b64encode(image.read()).decode('utf8')

        # Invoke the model with the prompt and the encoded image
        model_id = ""
        if model == "haiku":
            model_id = "anthropic.claude-3-haiku-20240307-v1:0"

        if model == "opus":
            model_id = "us.anthropic.claude-3-opus-20240229-v1:0"

        if model == "sonnet":
            model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

        if model == "sonnet-v-2":
            model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image:",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_content,
                            },
                        },
                    ],
                }
            ],
        }
        try: 
            awsresponse = client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body),
            )

            # Process and print the response
            result = json.loads(awsresponse.get("body").read())
            response = result
            response['status'] = 200
            return response
        
        except ( client.exceptions.AccessDeniedException, client.exceptions.ResourceNotFoundException, client.exceptions.ThrottlingException, client.exceptions.ModelTimeoutException, client.exceptions.InternalServerException, client.exceptions.ValidationException, client.exceptions.ModelNotReadyException, client.exceptions.ServiceQuotaExceededException) as e:
            response = {"status": 400}
            return response
            
        except client.exceptions.ModelErrorException as e:
            response = {"status": 500}
            return response


