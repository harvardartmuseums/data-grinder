import config
import urllib3
import json

def get_ham_object(id):
	huam_url = config.HAM_API_ENDPOINT + "%s/%s?apikey=%s" % ('object', id, config.HAM_API_KEY)

	http = urllib3.PoolManager()
	response = http.request('GET', huam_url)
	huam = response.data

	if (huam.decode('utf8')==''):
		return (False, HttpResponse("The object ID %s does not exist" % id, status=404))

	return (True, huam.decode('utf8'))


def get_ham_image(image_url, id):
	http = urllib3.PoolManager()
	r = http.request('GET', image_url, preload_content=False)

	path = "samples/%s.jpg" % (id)

	with open(path, 'wb') as out:
		while True:
			data = r.read(2048)
			if not data:
				break
			out.write(data)

	r.release_conn()

	return path


def get_ham_object_id_list(page=1, person=None, has_image=1, technique=None, object=None):
	huam_url = config.HAM_API_ENDPOINT + "%s" % ('object')
	query_string = {
			"apikey": config.HAM_API_KEY,
			"fields": "id",
			"size": 100,
			"page": page,
			"hasimage": has_image
		}

	if person is not None:
		query_string["person"] = person
	
	if technique is not None:
		query_string["technique"] = technique

	if object is not None:
		query_string["q"] = "objectid:" + object
	
	http = urllib3.PoolManager()
	response = http.request('GET', huam_url, fields = query_string)
	huam = response.data
	records = json.loads(huam.decode('utf8'))["records"]
	ids = [record["id"] for record in records]

	return (True, ids)