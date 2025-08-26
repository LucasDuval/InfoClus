import os, pickle, base64

from infoclus import InfoClus
from caching import from_cache
from config import PROJECT_ROOT


def build_infoclus(dataset_name: str='german_socio_eco', emb_name='tsne', linkage='single', modify=True):

    cache_folder = os.path.join(PROJECT_ROOT, 'data', dataset_name, 'cache')
    file_path = os.path.join(cache_folder, emb_name +'_'+ linkage + '_' + 'modify_' + str(modify))

    if os.path.exists(file_path):
        infoclus = from_cache(file_path)
    else:
        infoclus = InfoClus(dataset_name=dataset_name, emb_name=emb_name, linkage=linkage, modify_hierarchical=modify)
    return infoclus

def serialize_obj(obj):
    try:
        pickled = pickle.dumps(obj)
        encoded = base64.b64encode(pickled).decode('utf-8')
        return encoded
    except Exception as e:
        print(f"Serialization failed: {e}")
        return None

def deserialize_obj(data: str):
    try:
        decoded = base64.b64decode(data.encode('utf-8'))
        obj = pickle.loads(decoded)
        return obj
    except Exception as e:
        print(f"Deserialization failed: {e}")
        return None
