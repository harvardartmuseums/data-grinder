import os
import argparse
import base64
import httplib2
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

class Vision(object):

    def fetch(self, photo_file):
        # The url template to retrieve the discovery document for trusted testers.
        DISCOVERY_URL='https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'

        MAX_RESULTS = 20

        """Run a label request on a single image"""

        # [START authenticate]
        creds = os.getenv("GOOGLE_VISION_CREDENTIALS")
        path = os.path.dirname(os.path.realpath(__file__))
        creds_file = path.replace("parsers", "temp/vision-credentials.json")
        with open(creds_file, "w") as outfile:
            outfile.write(creds)

        credentials = GoogleCredentials.from_stream(creds_file)
        service = discovery.build('vision', 'v1', cache_discovery=False, credentials=credentials,
                                  discoveryServiceUrl=DISCOVERY_URL)
        # [END authenticate]

        # [START construct_request]
        with open(photo_file, 'rb') as image:
            image_content = base64.b64encode(image.read())
            service_request = service.images().annotate(body={
                'requests': [{
                    'image': {
                        'content': image_content.decode('UTF-8')
                    },
                    'features': [{
                            'type': 'LABEL_DETECTION',
                            'maxResults': MAX_RESULTS
                        },
                        {
                            'type': 'FACE_DETECTION',
                            'maxResults': MAX_RESULTS
                        },
                        {
                            'type': 'TEXT_DETECTION',
                            'maxResults': MAX_RESULTS
                        },
                        {
                            'type': 'LANDMARK_DETECTION',
                            'maxResults': MAX_RESULTS
                        }
                    ]
                }]
            })
            # [END construct_request]
            # [START parse_response]
            response = service_request.execute()
            return response