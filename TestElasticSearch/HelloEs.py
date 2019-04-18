from datetime import datetime
from elasticsearch5 import Elasticsearch
es = Elasticsearch(hosts='localhost:9200')

doc = {
    'author': 'kimchy',
    'text': 'Elasticsearch! cool. bonsai cool.',
    'timestamp': datetime.now(),
}
res = es.index(index="test-index", doc_type='tweet', id=1, body=doc)
print("Index: " + res['result'])

res = es.get(index="test-index", doc_type='tweet', id=1)
print("Get: {0}".format(res['_source']))

es.indices.refresh(index="test-index")

res = es.search(index="test-index", body={"query": {"match_all": {}}})
print("Got %d Hits:".format(res['hits']['total']))
for hit in res['hits']['hits']:
    print("%(timestamp)s %(author)s: %(text)s" % hit["_source"])

es.delete(index="test-index", doc_type='tweet', id=1)