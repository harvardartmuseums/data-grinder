from urllib2 import urlopen, Request
import json
import config
import uuid

class CalaisAPI(object):
    """
    Interacts with the `OpenCalais API <http://www.opencalais.com/documentation/calais-web-service-api>`_.

    :param api_key: OpenCalais API key, defaults to ``CALAIS_API_KEY``
        config variable.
    :param endpoint: OpenCalais API endpoint URL, defaults to
        ``CALAIS_ENDPOINT`` config variable.
    :type api_key: string
    :type endpoint: string
    """

    def __init__(self, api_key=None, endpoint=None):
       self.api_key = api_key or config.CALAIS_API_KEY
       self.endpoint = endpoint or config.CALAIS_ENDPOINT

    def call_api(self, payload):
        """
        Calls the Calais API endpoint with the given payload.

        :param payload: Fulltext payload
        :type payload: string
        :return: dictionary with JSON response from the Calais API
        """
        headers = {
           'X-AG-Access-Token': '%s' % self.api_key,
           'Content-Type': 'text/raw',
           'OutputFormat': 'application/json',
           'X-Calais_Language': 'English'
        }
        r = Request(self.endpoint, data=payload.encode('utf-8'))
        for k,v in headers.items():
           r.add_header(k,v)
        resp = urlopen(r)
        return json.loads(resp.read())

    def process_results(self, results):
        """
        Takes a set of entity results such as those returned by
        :py:meth:`parserbot.calais.CalaisAPI.call_api`,
        scores them and formats them for storage.

        :param results: Results from the OpenCalais API.
        :type results: dict
        :return: Sorted list of topics, entities, and tags from OpenCalais.
        """
        formatted_results = []
        for key, entity in results.iteritems():
            try:
                typeGroup = entity["_typeGroup"]
            except KeyError:
                continue
            if typeGroup == "socialTag":
                name = entity["name"]
                calais_id = entity["socialTag"]
                score = 0.8 if int(entity["importance"]) > 1 else 0.6
                entity_type = ""
            elif typeGroup == "topics":
                name = entity["name"]
                calais_id = key
                score = float(entity["score"]) if "score" in entity else 0.0
                entity_type = ""
            elif typeGroup == "entities":
                name = entity["name"]
                calais_id = key
                score = float(entity["relevance"])
                entity_type = entity.get("_type", "")
            else:
                continue
            formatted_results.append({
                'name': name,
                'calais_id': calais_id,
                'score': score,
                'entity_type': entity_type,
                'type_group': typeGroup
                })
        return sorted(formatted_results, key=lambda x: x['score'], reverse=True)

    def extract_entities(self, payload):
        """
        Takes a fulltext natural language payload, calls
        :py:meth:`parserbot.calais.CalaisAPI.call_api`, then runs
        :py:meth:`parserbot.calais.CalaisAPI.process_results` on the results.

        :param payload: Fulltext natural language payload.
        :type payload: string
        :return: Sorted list of topics, entities, and tags from OpenCalais.
        """
        results = self.call_api(payload)
        return self.process_results(results)