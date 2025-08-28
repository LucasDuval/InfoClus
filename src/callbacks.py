import os

import dash
from dash.dependencies import Input, Output, State

from layout import *
from dash_utils import build_infoclus, serialize_obj, deserialize_obj
from config import PROJECT_ROOT

def register_callbacks(app):

    @app.callback(
        Output('infoclus_store', 'data'),
        Output('dataset_store', 'data'),
        Output('embedding_store', 'data'),
        [Input('dataset-select', 'value'),
         Input('recalc-hyperparameters', 'value')],
        [State("embedding-select", "value"),
        State("alpha-slider", "value"),
        State("beta-slider", "value"),
        State("min-att-input", "value"),
        State("max-att-input", "value")]
    )
    def update_store(dataset, recalc_hyperparams, embedding_name, alpha, beta, min_att, max_att):

        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        info_cache_update = None
        data_update = None
        embeddings_update = None
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'dataset-select':
            infoclus_obj = build_infoclus(dataset)
            info_cache_update = infoclus_obj.optimise()

            df_data = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', dataset, f'{dataset}.csv'))
            data_update = serialize_obj(df_data)

            embeddings_load = np.load(os.path.join(PROJECT_ROOT, 'data', dataset, 'cache', 'embeddings.npz'))
            embeddings = {k: embeddings_load[k] for k in embeddings_load.files}
            embeddings_update = serialize_obj(embeddings)

        elif trigger_id == 'recalc-hyperparameters':
            infoclus_obj = build_infoclus(dataset_name=dataset, emb_name=embedding_name)
            info_cache_update = infoclus_obj.optimise(alpha=alpha, beta=beta, min_att=min_att,max_att=max_att, run_id=recalc_hyperparams)
            data_update = dash.no_update
            embeddings_update = dash.no_update
        else:
            print('unknown trigger id')

        return info_cache_update, data_update, embeddings_update


    @app.callback(
        Output('dashboard-content', 'children'),
        Input('infoclus_store', 'data'),
        [State('dataset_store', 'data'),
        State('embedding_store', 'data')]
    )
    def update_content(infoc_store, data_store, embedding_store):
        infoc_dict = infoc_store
        df_data = deserialize_obj(data_store)
        embeddings_dict = deserialize_obj(embedding_store)
        return config_layout(infoc_para_res=infoc_dict, df_data=df_data, embeddings=embeddings_dict)


    @app.callback(
        Output('explanation', 'children'),
        Input('cluster-select', 'value'),
        [State('infoclus_store', 'data'),
         State('dataset_store', 'data'),]

    )
    def select_cluster_explanation(cluster_id, infoc_dict, data_store):
        df_data = deserialize_obj(data_store)
        return config_explanations(infoc_para_res=infoc_dict, df_data=df_data, cluster_label=cluster_id)

    @app.callback(
        Output('embedding-scatterPlot', 'figure'),
        Input('embedding-for-show', 'value'),
        [State('infoclus_store', 'data'),
         State('embedding_store', 'data')]
    )
    def select_embedding(emb_name, infoc_dict, embedding_store):
        embeddings_dict = deserialize_obj(embedding_store)
        return config_scatter_graph(infoc_dict,embeddings_dict[emb_name])