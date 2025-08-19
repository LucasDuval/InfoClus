import os, pickle, base64
from sklearn.cluster import AgglomerativeClustering

from infoclus2 import InfoClus
from caching import from_cache
from config import PROJECT_ROOT


def build_infoclus(dataset_name: str='german_socio_eco'):

    cache_folder = os.path.join(PROJECT_ROOT, 'data', dataset_name, 'cache')
    file_path = os.path.join(cache_folder, dataset_name + '_modify_True')
    is_infoclus_exist = os.path.exists(file_path)


    if is_infoclus_exist:
        infoclus = from_cache(file_path)
    else:
        infoclus = InfoClus(dataset_name)
    return infoclus

def serialize_infoclus(obj: InfoClus):
    try:
        pickled = pickle.dumps(obj)
        encoded = base64.b64encode(pickled).decode('utf-8')
        return encoded
    except Exception as e:
        print(f"Serialization failed: {e}")
        return None

def deserialize_infoclus(data: str) -> InfoClus:
    try:
        decoded = base64.b64decode(data.encode('utf-8'))
        obj = pickle.loads(decoded)
        return obj
    except Exception as e:
        print(f"Deserialization failed: {e}")
        return None
