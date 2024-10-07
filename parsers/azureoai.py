from openai import AzureOpenAI
import os

class AzureOAI(object):

    def __init__(self):
        self.api_key = os.getenv("AZURE_OPENAI_KEY")
        self.model = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    def fetch(self, photo_file, model="gpt-4"):
        client = AzureOpenAI(
            api_version = self.api_version,
            api_key = self.api_key,
            azure_endpoint = self.endpoint
        )

        if model == "gpt-4":
            self.model = "HAM-GPT-4-V-D1"

        if model == "gpt-4o":
            self.model = "HAM-GPT-4o-V-D1"

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
                            "url": photo_file
                        }
                    }
                ] } 
            ]

        response = client.chat.completions.create(
            model = self.model,
            messages = prompt,
            max_tokens = 2000
        )
        
        result = {
            "description": response.model_dump(),
            "prompt": prompt
        }

        return result
    