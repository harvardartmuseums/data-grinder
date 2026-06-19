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

## Image Cache

Downloaded images are cached to avoid redundant HTTP requests on repeat calls. Each image is stored in multiple size tiers in a single pass. The full-size image is always cached; additional scaled variants are generated from it locally without a second HTTP request.

Files are organized as `<cache_root>/<tier>/<domain>/<filename>.jpg`, where the domain and filename are derived from the original IIIF input URL. For example:

```
temp/
  full/
    nrs.harvard.edu/
      urn-3_HUAM_DDC250728_dynmc.jpg
  1110/
    nrs.harvard.edu/
      urn-3_HUAM_DDC250728_dynmc.jpg
  512/
    www.artic.edu/
      e966799b-97ee-1cc6-bd2f-a94b4b8bb8f9.jpg
```

### Cache configuration

Set these in `.env`:

| Variable | Default | Description |
|---|---|---|
| `IMAGE_CACHE_DAYS` | `30` | Number of days before a cached file is considered stale |
| `IMAGE_CACHE_SIZES` | `full,1110` | Comma-separated list of size tiers to cache. `full` is always included. Scaled values are the max pixel dimension for a proportional fit |
| `IMAGE_CACHE_DIR` | `./temp` | Local directory for the cache. Defaults to a `temp` folder next to `main.py` |
| `IMAGE_CACHE_S3_BUCKET` | _(empty)_ | S3 bucket name. When set, S3 is used as a shared cache backend |
| `IMAGE_CACHE_S3_PREFIX` | _(empty)_ | Optional key prefix within the bucket, e.g. `image-cache/` |

### S3 cache mode

When `IMAGE_CACHE_S3_BUCKET` is set, the cache uses S3 as a shared backend alongside the local directory. The lookup order for each size tier is:

1. **Local** — return immediately if a fresh local copy exists
2. **S3** — download from the bucket to local if a fresh object exists there
3. **Origin / generate** — for `full`, fetch from the source URL; for scaled tiers, resize from the full image. Write locally then upload to S3

S3 reuses the existing `AWS_ACCESS_KEY`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION` credentials. S3 errors are logged as warnings and fall through gracefully — a misconfigured bucket will not break a request.

## Services Implemented

HAM Color Service: extract colors  
Hashing: compute average, color, perceptual, difference, wavelet hashes  
Clarifai: tag features, tag objects, write captions  
Imagga: tag features, extract colors, categorize, faces, write caption, structured tags
Google Vision: tag features, find faces, find text  
Microsoft Cognitive Services: categories, tags, description, faces, color, objects  
AWS Rekognition: labels, faces, text  
OpenAI GPT-4 on Azure: description   
OpenAI GPT-4o on Azure: description   
OpenAI GPT-4.1 Mini on Azure: description   
Anthropic Claude 3 Haiku on AWS Bedrock: description  
Anthropic Claude 4.5 Haiku on AWS Bedrock: description  
Anthropic Claude 3 Opus on AWS Bedrock: description (deprecated by AWS on January 15, 2026)  
Anthropic Claude 4.1 Opus on AWS Bedrock: description  
Anthropic Claude 4.5 Opus on AWS Bedrock: description  
Anthropic Claude 3.5 Sonnet on AWS Bedrock: description (deprecated by AWS)  
Anthropic Claude 3.5 Sonnet v2 on AWS Bedrock: description (deprecated by AWS)  
Anthropic Claude 3.7 Sonnet on AWS Bedrock: description (deprecated by AWS on April 28, 2026)  
Anthropic Claude 4 Sonnet on AWS Bedrock: description  
Anthropic Claude 4.5 Sonnet on AWS Bedrock: description  
Meta Llama 3.2 11b on AWS Bedrock: description  
Meta Llama 3.2 90b on AWS Bedrock: description  
Meta Llama 4 Maverick 17b on AWS Bedrock: description  
Meta Llama 4 Scout 17b on AWS Bedrock: description  
Amazon Nova Lite 1.0 on AWS Bedrock: description  
Amazon Nova Lite 2 on AWS Bedrock: description  
Amazon Nova Pro 1.0 on AWS Bedrock: description  
Google Gemini 2.0 Flash: description  
Google Gemini 2.5 Flash: description  
Google Gemini 2.0 Flash-Lite: description  
Google Gemini 2.5 Flash-Lite: description  
Google Gemini 3.1 Flash-Lite: description  
Mistral Pixtral Large 25.02: description  
Mistral Magistral Small 2509: description  
Mistral Ministral 3 3B: description  
Mistral Ministral 3 8B: description  
Mistral Ministral 3 14B: description  
Mistral Large 3 675B: description  
Qwen 2.5 VL 7B Instruct on Hyperbolic: description (deprecated)  
Qwen 2.5 VL 72B Instruct on Hyperbolic: description (deprecated)  
Qwen 3 VL 235B on AWS Bedrock: description  
Moonshot Kimi K2.5 on AWS Bedrock: description  
Writer Palmyra Vision 7B on AWS Bedrock: description  


<img width="600" alt="API Tools-Data-Process Diagrams" src="https://github.com/user-attachments/assets/d6ae133b-f8ab-4df5-b890-67ba72fb3049" />

## LLM Dispatch

All LLM/vision model calls run in parallel using `ThreadPoolExecutor`. When multiple models are requested, the total response time is approximately the slowest individual model rather than the sum of all models.

### Configuration

Set these in `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_CONNECT_TIMEOUT` | `10` | Seconds to wait for a connection to be established |
| `LLM_READ_TIMEOUT` | `60` | Seconds to wait for a response from the model |
| `LLM_WORKERS` | `10` | Max parallel threads for concurrent model calls. Lower values reduce peak memory usage |
| `REQUEST_BUDGET` | `90` | Max seconds the request may spend in the parallel LLM phase. Models still running when the budget expires are skipped and logged |
| `MAX_PROMPT_LEN` | `500` | Maximum allowed characters in the `prompt` query parameter |

If a model exceeds `LLM_READ_TIMEOUT`, its SDK raises a timeout exception which is caught and returned as an error dict — other models in the request are unaffected.

### Gunicorn

Set `--timeout` higher than `LLM_READ_TIMEOUT` so gunicorn does not kill workers before the SDK timeouts fire. Use `--max-requests` to recycle workers and reclaim memory:

```sh
gunicorn wsgi:app --timeout 100 --workers 2 --max-requests 500 --max-requests-jitter 50
```

On memory-constrained hosts (e.g. Digital Ocean), lower `--workers` and `LLM_WORKERS` to reduce peak RSS. Each concurrent model call holds a copy of the image bytes in memory, so `LLM_WORKERS` directly controls the memory spike during parallel dispatch.

## Testing

Run the test suite with pytest:

```sh
pytest
```

The suite covers parsing logic, cache tier decisions, IIIF URL construction, and Flask endpoint routing. All external API and S3 calls are mocked — no credentials or network access required.

```sh
# Unit tests only (fastest, no I/O)
pytest tests/test_main_unit.py

# With coverage report
pytest --cov=. --cov-report=term-missing --cov-omit="venv/*,tests/*,parsers/colors.py"
```

Integration tests hit live APIs with real credentials and known images. They are skipped by default and must be opted into explicitly:

```sh
# Run integration tests (requires credentials in .env)
RUN_INTEGRATION=1 pytest tests/test_integration.py -v -s
```

The `-s` flag allows print output to show which content-blocking flag (`filtered` vs `content_policy_violation`) each model returns for the flagged image.

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
url | Any URL that resolves to a IIIF compatible image
services | (optional, default uses all services) One or more from the list of valid services separated by commas | `imagga, gv, mcs, clarifai, color, aws, hash, gpt-4, gpt-4o, gpt-4-1-mini, claude-3-haiku, claude-4-5-haiku, claude-3-opus, claude-4-1-opus, claude-4-5-opus, claude-3-5-sonnet, claude-3-5-sonnet-v-2, claude-3-7-sonnet, claude-4-sonnet, claude-4-5-sonnet, llama-3-2-11b, llama-3-2-90b, llama-4-maverick-17b, llama-4-scout-17b, nova-lite-1-0, nova-pro-1-0, nova-lite-2-0, gemini-2-0-flash-lite, gemini-2-5-flash-lite, gemini-2-0-flash, gemini-2-5-flash, gemini-3-1-flash-lite, pixtral-large-2502, magistral-small-2509, ministral-3-3b, ministral-3-8b, ministral-3-14b, mistral-large-3-675b, qwen-3-vl-235b-a22b, kimi-k2-5, palmyra-vision-7b, blip-2, blip-2-6-7B`
prompt | (optional) Custom text prompt sent to all LLM/vision model calls. When omitted, defaults to `"Describe this image:"`. Max 500 characters; control characters are stripped. | Any plain text string


Example request:  

```
http://127.0.0.1:5000/extract?url=https://nrs.harvard.edu/urn-3:HUAM:DDC251942_dynmc&services=imagga,clarifai:caption|classification
```

Example response:
```json
{
    "height": 4749,
    "heightFull": 5313,
    "idsid": 505401002,
    "iiifFullImageURL": "https://nrs.harvard.edu/urn-3:HUAM:819780/full/full/0/default.jpg",
    "iiifbaseuri": "https://nrs.harvard.edu/urn-3:HUAM:819780",
    "lastupdated": "2026-05-01 14:12:48",
    "runtime": 21.511509656906128,
    "scalefactor": 1.1188,
    "status": "ok",
    "url": "https://nrs.harvard.edu/urn-3:HUAM:819780",
    "width": 5000,
    "widthFull": 5594,  
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
        "tags": {},
        "structuredTags": {},
        "caption": {}
    },
    "aws": {
        "faces": {},
        "labels": {},
        "text": {}
    },
    "gpt-4": {},
    "gpt-4o": {},
    "gpt-4-1-mini": {},
    "claude-3-haiku": {},
    "claude-3-opus": {},
    "claude-3-5-sonnet": {},
    "claude-3-5-sonnet-v-2": {},
    "claude-3-7-sonnet": {},
    "claude-4-sonnet": {},
    "claude-4-1-opus": {},
    "claude-4-5-haiku": {},
    "claude-4-5-opus": {},
    "claude-4-5-sonnet": {},
    "gemini-2-0-flash": {}, 
    "gemini-2-5-flash": {}, 
    "gemini-2-0-flash-lite": {},
    "gemini-2-5-flash-lite": {},
    "gemini-3-1-flash-lite": {},
    "llama-3-2-11b": {},
    "llama-3-2-90b": {},
    "llama-4-maverick-17b": {}, 
    "llama-4-scout-17b": {},
    "nova-lite-1-0": {},
    "nova-lite-2-0": {},
    "nova-pro-1-0": {},
    "magistral-small-2509": {}, 
    "ministral-3-3b": {}, 
    "ministral-3-8b": {}, 
    "ministral-3-14b": {}, 
    "mistral-large-3-675b": {},    
    "pixtral-large-2502": {},
    "qwen-3-vl-235b-a22b": {},
    "kimi-k2-5": {},
    "palmyra-vision-7b": {}
}
```

Data in the response:  
`url`: The original image URL passed in to this script.    
`status`: The response from the image service. The value will be "ok" or "bad".  
`width`: The width in pixels of the image supplied to the service.  
`height`: The height in pixels of the image supplied to the service.  
`widthFull`: The width in pixels of the full image as read from /info.json.  
`heightFull`: The height in pixels of the full image as read from /info.json.  
`scalefactor`: The proportional difference between the supplied image and the full image. In the example response, the full image is 1.334 times larger than the supplied image.  
`runtime`: The amount of time it took to run the entire request, in seconds. Each LLM model result object also contains a `runtime` field recording how long that individual model call took.  
`iiifFullImageURL`: A fully formed IIIF URI for delivering the full image (e.g. /full/full/0/default.jpg).  
`iiifbaseuri`: A base URI for the IIIF image.  
`colors`:  
`hashes`:  
`clarifai`:   
`microsoftvision`:  
`googlevision`:  
`imagga`:  
`aws`:  
`gpt-4`:  
`gpt-4o`:  
`gpt-4-1-mini`:  
`claude-3-haiku`:  
`claude-3-opus`:  
`claude-3-5-sonnet`:  
`claude-3-5-sonnet-v-2`:  
`claude-3-7-sonnet`:  
`claude-4-sonnet`:  
`claude-4-1-opus`:  
`claude-4-5-haiku`:  
`claude-4-5-opus`:  
`claude-4-5-sonnet`:  
`gemini-2-0-flash`:   
`gemini-2-5-flash`:   
`gemini-2-0-flash-lite`:  
`gemini-2-5-flash-lite`:  
`gemini-3-1-flash-lite`:  
`llama-3-2-11b`:  
`llama-3-2-90b`:  
`llama-4-maverick-17b`:   
`llama-4-scout-17b`:  
`nova-lite-1-0`:  
`nova-lite-2-0`:  
`nova-pro-1-0`:  
`magistral-small-2509`:   
`ministral-3-3b`:   
`ministral-3-8b`:   
`ministral-3-14b`:   
`mistral-large-3-675b`:      
`pixtral-large-2502`:  
`qwen-3-vl-235b-a22b`:  
`kimi-k2-5`:  
`palmyra-vision-7b`:  

### LLM model response structure

Each LLM/vision model key in the response (e.g. `gpt-4o`, `claude-4-5-sonnet`, `gemini-2-5-flash`) contains an object with the following fields:

| Field | Type | Description |
|---|---|---|
| `body` | string or null | The model's text response. `null` on error or when the response was blocked |
| `model` | string | The exact model version ID used (e.g. `gpt-4o-2024-11-20`) |
| `provider` | string | The provider name (e.g. `OpenAI`, `Anthropic`, `Google Gemini`) |
| `status` | integer | HTTP-style status code. `200` on success; `400`, `429`, `500`, etc. on error |
| `runtime` | float | Seconds elapsed for this individual model call |
| `full` | object or null | The complete raw response object from the provider SDK. `null` on error |
| `description` | string or object | Present on error. A human-readable description of the error, or a structured error object for content policy violations |
| `filtered` | boolean | Present and `true` when the response was blocked by the provider's content filter. `body` is `null`. For Azure OpenAI, `body` contains a human-readable summary of which content categories were flagged |
| `content_policy_violation` | boolean | Present and `true` when the request itself was rejected before a response was generated due to a content policy violation. Distinct from `filtered`: `filtered` means a response was generated but withheld; `content_policy_violation` means the request was refused outright |
| `truncated` | boolean | Present and `true` when the model stopped because it hit the token limit rather than reaching a natural end |

**Normal success example:**
```json
"gpt-4o": {
    "body": "This painting depicts...",
    "model": "gpt-4o-2024-11-20",
    "provider": "OpenAI",
    "status": 200,
    "runtime": 4.231,
    "full": { ... }
}
```

**Error example:**
```json
"gpt-4o": {
    "body": null,
    "model": "gpt-4o-2024-11-20",
    "provider": "OpenAI",
    "status": 429,
    "runtime": 1.04,
    "description": "Rate limit exceeded. Please retry after 60 seconds.",
    "full": null
}
```