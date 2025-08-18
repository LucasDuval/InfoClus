import os, pickle, base64
from sklearn.cluster import AgglomerativeClustering

from infoclus2 import InfoClus
from caching import from_cache
from config import PROJECT_ROOT


def build_infoclus(dataset_name: str='german_socio_eco'):

    cache_folder = os.path.join(PROJECT_ROOT, 'data', dataset_name, 'cache')
    is_infoclus_exist = os.path.exists(os.path.join(cache_folder, dataset_name))

    if is_infoclus_exist:
        infoclus = from_cache(os.path.join(cache_folder, dataset_name))
    else:
        infoclus = InfoClus(dataset_name)
    # infoclus.optimise()

    # data_path = os.path.join(PROJECT_ROOT, 'data', dataset_name, f'{dataset_name}_{embedding_name}.pkl')
    # if os.path.exists(data_path):
    #     with open(data_path, 'rb') as file:
    #         infoclus = pickle.load(file)
    # else:
    #     model = AgglomerativeClustering(linkage='single', distance_threshold=0, n_clusters=None)
    #     infoclus = InfoClus(dataset_name=dataset_name, main_emb=embedding_name,
    #                             model=model,
    #                             Allow_cache=False,
    #                             Modify_hierarchical=False,
    #                             Base_Clusters=1000)
    #     infoclus.optimise()
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
