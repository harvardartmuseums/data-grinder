# Data Grinder

A simple script and service for running an image through third party computer vision services that extract faces, text, colors, and tags.  

## Requirements

* Python 3.9.*
* Virtualenv


## Setup

### Clone the repository
```
> git clone https://github.com/harvardartmuseums/data-grinder.git
```

### Create a virtual environment
We recommend creating a virtual environment with [Virtualenv](https://pypi.org/project/virtualenv/) and running everything within it.

```
> cd data-grinder
> virtualenv venv
```

### Activate the virtual environment
```
> venv\Scripts\activate.bat
```

### Install dependencies
```
> pip install -r requirements.txt
```

### Set configuration values
* Clone .env-template to .env
* Open .env in a text editor.
* Enter API keys and credentials for the services you want to use.  

## Services Implemented

HAM Color Service: extract colors  
Hashing: compute average, color, perceptual, difference, wavelet hashes  
Clarifai: tag features, tag objects, write captions  
Imagga: tag features, extract colors, categorize, faces  
Google Vision: tag features, find faces, find text  
Microsoft Cognitive Services: categories, tags, description, faces, color, objects  
AWS Rekognition: labels, faces, text  
OpenAI GPT-4 on Azure: description   
OpenAI GPT-4o on Azure: description   
Anthropic Claude 3 Haiku on AWS Bedrock: description  
Anthropic Claude 3 Opus on AWS Bedrock: description  
Anthropic Claude 3.5 Sonnet on AWS Bedrock: description  
Anthropic Claude 3.5 Sonnet v2 on AWS Bedrock: description  
Meta Llama 3.2 11b on AWS Bedrock: description  
Meta Llama 3.2 90b on AWS Bedrock: description  
Amazon Nova Lite 1.0 on AWS Bedrock: description  
Amazon Nova Pro 1.0 on AWS Bedrock: description  
Google Gemini 2.0 Flash: description  
Google Gemini 2.0 Flash-Lite: description  
Mistral Pixtral Large 25.02: description  

<img width="600" alt="API Tools-Data-Process Diagrams" src="https://github.com/user-attachments/assets/d6ae133b-f8ab-4df5-b890-67ba72fb3049" />

## Usage

Run as a script from the command line:
```sh
$ python main.py -url https://some.image/url
```
OR

Run as a local service: 
```sh
$ flask run 
```
Then open a browser to http://127.0.0.1:5000 to check the status of the application.

### List available services and functions

```
http://127.0.0.1:5000/list/services
```

### Run an image against one or more services

```
http://127.0.0.1:5000/extract
```

Parameters |  | Values
------------ | ------------- | -------------
url | Any Harvard NRS URL that resolves to a IIIF compatible image
services | (optional, default uses all services) One or more from the list of valid services separated by commas | `imagga, gv, mcs, clarifai, color, aws, hash, gpt-4, gpt-4o, claude-3-haiku, claude-3-opus, claude-3-5-sonnet, claude-3-5-sonnet-v-2, llama-3-2-11b, llama-3-2-90b, nova-lite-1-0, nova-pro-1-0, gemini-2-0-flash, gemini-2-0-flash-lite, pixtral-large-2502`


Example request:  

```
http://127.0.0.1:5000/extract?url=https://nrs.harvard.edu/urn-3:HUAM:DDC251942_dynmc&services=imagga,clarifai:caption|classification
```

Example response:
```json
{
    "lastupdated": "2018-02-07 21:58:28",
    "drsstatus": "ok",
    "width": 581,
    "height": 768,
    "widthFull": 775,
    "heightFull": 1024,
    "scalefactor": 1.333907056798623,    
    "url": "https://nrs.harvard.edu/urn-3:huam:75033B_dynmc",    
    "iiifbaseuri": "https://ids.lib.harvard.edu/ids/iiif/14178676",
    "idsid": "14178676",
    "colors": [],
    "hashes": {},
    "clarifai": {
        "caption": {},
        "classification": {},
        "object": {}
    },
    "microsoftvision": {
        "analyze": {
            "categories": [],
            "color": {},
            "description": {
                "captions": [],
                "tags": []                
            },
            "faces": [],
            "objects": [],
            "tags": []            
        },
        "describe": {
            "description": {
                "captions": [],
                "tags": []
            }
        }
    },
    "googlevision": {
        "faceAnnotations": [],
        "fullTextAnnotation": {},
        "labelAnnotations": [],
        "landmarkAnnotations": [],
        "textAnnotations": []
    },    
    "imagga": {
        "categories": {},
        "colors": {},
        "faces": {},
        "tags": {}
    },
    "aws": {
        "faces": {},
        "labels": {},
        "text": {}
    },
    "gpt-4": {},
    "gpt-4o": {},
    "claude-3-haiku": {},
    "claude-3-opus": {},
    "claude-3-5-sonnet": {},
    "claude-3-5-sonnet-v-2": {},
    "llama-3-2-11b": {},
    "llama-3-2-90b": {},
    "nova-lite-1-0": {},
    "nova-pro-1-0": {},
    "gemini-2-0-flash": {}, 
    "gemini-2-0-flash-lite": {},
    "pixtral-large-2502": {}
}
```

Data in the response:  
`url`: The original image URL passed in to this script. This must be in the form of a NRS URL for an image in the Harvard DRS.  
`drsstatus`: The response from the DRS. The value will be "ok" or "bad".  
`width`: The width in pixels of the image supplied on the command line.  
`height`: The height in pixels of the image supplied on the command line.  
`widthFull`: The width in pixels of the full image file in the DRS.  
`heightFull`: The height in pixels of the full image file in the DRS.  
`scalefactor`: The propotional difference between teh supplied image and the full image. In the example response, the full image is 1.334 times larger than the supplied image.  
`iiifbaseuri`: A fully formed IIIF URI for the image in the DRS.  
`idsid`: The image file ID in the DRS returned when requesting the original image URL.  
`colors`:  
`hashes`:  
`clarifai`:   
`microsoftvision`:  
`googlevision`:  
`imagga`:  
`aws`:  
`gpt-4`:  
`gpt-4o`:  
`claude-3-haiku`:  
`claude-3-opus`:  
`claude-3-5-sonnet`:  
`claude-3-5-sonnet-v-2`:  
`llama-3-2-11b`:  
`llama-3-2-90b`:  
`nova-lite-1-0`:  
`nova-pro-1-0`:  
`gemini-2-0-flash`:  
`gemini-2-0-flash-list`:  
`pixtral-large-2502`:  
