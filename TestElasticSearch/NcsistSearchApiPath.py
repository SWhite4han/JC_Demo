import elasticsearch as es
import cv2
import json
import pprint
from super_postman import super_post, img2string


class InfinitySearch:
    def __init__(self, conn_str):
        """

        :param conn_str: connection string
            hosts=<<host1>>,<<host2>>;port=<<port number>>;id=<<user id>>;passwd=<<password>>
        """

        """
        self.server_list = host.split(';')
        self.elasticsearch = es.Elasticsearch(self.server_list)
        """

        self.es_conn_str = conn_str
        self.process_conn_str()
        """
        # Simple HTTP connection
        self.elasticsearch = es.Elasticsearch(
            self.hosts,
            port=self.port,
            sniff_on_start=True,
            sniff_on_connection_fail=True,
            sniffer_timeout=60
        )
        """

        # self.api_url = 'http://10.10.53.209:9527'
        self.api_url = 'http://10.10.53.205:9527'

        # HTTPS/AUTH connection
        self.elasticsearch = es.Elasticsearch(
            self.hosts,
            http_auth=(self.id, self.password),
            scheme="https",  # https when using ssl. http otherwise.
            port=self.port,
            ca_certs=None,
            verify_certs=False,
            # --- For Test ---
            timeout=60,
            max_retries=10,
            retry_on_timeout=True
        )

        """
        self.elasticsearch = es.Elasticsearch(
            ['localhost', 'otherhost'],
            http_auth=('user', 'secret'),
            scheme="https",
            port=443,
            sniff_on_start=True,
            sniff_on_connection_fail=True,
            sniffer_timeout=60
        )
        """
        # self.elasticsearch = es.Elasticsearch(host)

        """
        # Move algorithm to %ES_HOME%/config/scripts
        try:
            requests.get(url, timeout=3.0)
            self.set_algorithm()
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            raise e
        """

    def process_conn_str(self):
        for item in self.es_conn_str.split(';'):
            key, value = item.split('=')
            if key == 'hosts':
                self.hosts = value.split(',')
            if key == 'port':
                self.port = value
            if key == 'id':
                self.id = value
            if key == 'passwd':
                self.password = value

    def status(self):
        response = self.elasticsearch.info()
        response = pprint.pformat(response)
        return response
        """
        url = self.es_url + '/'
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return 'ERROR:{0}'.format(response.status_code)
        """

    def count(self, target_index='infinity'):
        response = self.elasticsearch.count(index=target_index)
        response = pprint.pformat(response)
        return response
        """
        url = self.es_url + '/_count'
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return 'ERROR:{0}'.format(response.status_code)
        """

    def set_algorithm(self):
        json_script = json.dumps(
            {
                "script": {
                    "lang": "painless",
                    "code": "double s1 = 0; " +
                            "double s2 = 0; " +
                            "double s3 = 0; " +
                            "for (int i=0; i < params._source['imgVec'].length; i++) { " +
                            "    s1 = s1 + params._source['imgVec'][i] * params.imgVec[i]; " +
                            "    s2 = s2 + params._source['imgVec'][i] * params._source['imgVec'][i]; " +
                            "    s3 = s3 + params.imgVec[i] * params.imgVec[i]; " +
                            "} " +
                            "double f = s1 / (Math.sqrt(s2) * Math.sqrt(s3));" +
                            "return f;"
                }
            }
        )
        self.elasticsearch.put_script("im_cosine", body=json_script)

        json_script = json.dumps(
            {
                "script": {
                    "lang": "painless",
                    "code": "double dist = 0; " +
                            "for (int i=0; i < params._source['imgVec'].length; i++) { " +
                            "    double tmp = params._source['imgVec'][i] - params.imgVec[i]; " +
                            "    dist = dist + (tmp * tmp); " +
                            "} " +
                            "double f = Math.sqrt(dist); " +
                            "return (f == 0) ? 1 : 1.0/f; "
                }
            }
        )
        self.elasticsearch.put_script("im_euclidean", body=json_script)

        json_script = json.dumps(
            {
                "script": {
                    "lang": "painless",
                    "code": "double dist = 0; " +
                            "double tmp = doc['imgVec'][0] - params.imgVec[0]; " +
                            "double f = params._source['imgVec'][0]; " +
                            "return f; "
                }
            }
        )
        self.elasticsearch.put_script("im_dummy", body=json_script)

    def load_data(self, json_file, target_index='infinity'):
        """
        json_file should be json format
        {
           "imgVec": [0.14320628345012665, ...],
           "imgPath": "img/known_people_id/PER_1pgjoczr/000001.jpg",
           "category": "face"
        }
        """
        # data = json.loads(json_file)
        with open(json_file) as f:
            data = json.load(f)
        self.elasticsearch.index(index=target_index, doc_type='image_vector', body=data)

    def push_data(self, data, target_index='infinity'):
        self.elasticsearch.index(index=target_index, doc_type='image_vector', body=data)

    def delete_all(self, target_index='infinity'):
        response = self.elasticsearch.indices.delete(index=target_index)
        return response
        """
        url = self.es_url + '/_all'
        response = requests.delete(url)
        if response.status_code == 200:
            return response.text
        else:
            return 'ERROR:{0}'.format(response.status_code)
        """

    def query_result(self, json_file, target_index='infinity'):
        """
        query data should be json format
        {
           "imgVec": [0.14320628345012665, ...],
           "category": "face",
           "method": "im_cosine | im_euclidean",
           "top":10
        }

        response will be python list (or need json)
           [
                {
                    "_source": {
                        "imgPath": "img/known_people_id/PER_1pgjoczr/000001.jpg",
                        "category": "face"
                    "_score": 0.987654321
                }, ...
            ]

        """
        with open(json_file) as f:
            data = json.load(f)

        json_query = json.dumps(
            {
                "from": 0, "size": data["top"],
                "_source": ["imgPath", "category"],

                "query": {
                    "function_score": {
                        "query": {
                            "match": {"category": data["category"]},
                        },
                        "script_score": {
                            "script": {
                                "stored": data['method'],
                                "params": {
                                    "imgVec": data['imgVec']
                                }
                            }
                        },
                        "boost_mode": "replace"
                    }
                }
            }
        )
        response = self.elasticsearch.search(index=target_index, doc_type='image_vector', body=json_query)

        pprint.pprint(response)

        return response['hits']['hits']

    def query_face_result(self, img_vec, top=10, method='im_cosine', target_index='infinity'):
        """
        task: 0: yolo + facenet
              1: imagenet
              2: ocd + ocr
              3: ner
        """

        json_query = json.dumps(
            {
                "from": 0, "size": top,
                "_source": ["imgPath", "category"],

                "query": {
                    "function_score": {
                        "query": {
                            "match": {"category": 'face'},
                        },
                        "script_score": {
                            "script": {
                                "stored": method,
                                "params": {
                                    "imgVec": img_vec
                                }
                            }
                        },
                        "boost_mode": "replace"
                    }
                }
            }
        )
        response = self.elasticsearch.search(index=target_index, doc_type='image_vector', body=json_query)

        pprint.pprint(response)

        return response['hits']['hits']

    def query_image_result(self, img_vec, top=10, method='im_cosine', target_index='infinity'):
        """
        task: 0: yolo + facenet
              1: imagenet
              2: ocd + ocr
              3: ner
        """

        if img_vec and len(img_vec) == 2048:
            json_query = json.dumps(
                {
                    "from": 0, "size": top,
                    "_source": ["imgPath", "category"],

                    "query": {
                        "function_score": {
                            "query": {
                                "match": {"category": 'img'},
                            },
                            "script_score": {
                                "script": {
                                    "stored": method,
                                    "params": {
                                        "imgVec": img_vec
                                    }
                                }
                            },
                            "boost_mode": "replace"
                        }
                    }
                }
            )
            response = self.elasticsearch.search(index=target_index, doc_type='image_vector', body=json_query)

            pprint.pprint(response)

            return response['hits']['hits']
        else:
            return None

    def query_ocr_result(self, file_path, top=10, method='im_cosine'):
        """
        task: 0: yolo + facenet
              1: imagenet
              2: ocd + ocr
              3: ner
        """
        image = cv2.imread(file_path)
        b64 = img2string(image)
        post_data = {
            'task': '2',
            'image': b64,
        }
        rlt = super_post(data=post_data, url=self.api_url)
        rlt_data = json.loads(rlt.text)
        return rlt_data

    def search_result(self, field, search_text, target_index='infinity'):
        json_query = json.dumps(
            {
                "_source": ["imgPath", "category", "keywords"],
                "query": {"match": {field: search_text}}
            }
        )
        response = self.elasticsearch.search(index=target_index, doc_type='image_vector', body=json_query)

        pprint.pprint(response)

        return response['hits']['hits']

    def update_fields(self, id_no, field, new_text, target_index='infinity'):
        json_update = json.dumps(
            {
                "doc": {
                    field: new_text
                }
            }
        )
        response = self.elasticsearch.update(index=target_index, doc_type='image_vector', id=id_no, body=json_update)

        pprint.pprint(response)

        return response
