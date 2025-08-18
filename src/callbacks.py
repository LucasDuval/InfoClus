import dash
from dash.dependencies import Input, Output, State

from layout import *
from dash_utils import build_infoclus, serialize_infoclus, deserialize_infoclus
from config import PROJECT_ROOT

def register_callbacks(app):

    @app.callback(
        Output('infoclus_store', 'data'),
        [Input('dataset-select', 'value'),
         Input('recalc-hyperparameters', 'n_clicks')],
        [State("embedding-select", "value"),
        State("alpha-slider", "value"),
        State("beta-slider", "value"),
        State("runtime-slider", "value"),
        State("minAtt-slider", "value")]
    )
    def update_store(dataset, recalc_hyperparams, embedding_name, alpha, beta, runtime_id, minAtt):

        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'dataset-select':
            infoclus_obj = build_infoclus(dataset, 'tsne')
            infoclus_obj.optimise()
        elif trigger_id == 'recalc-hyperparameters':
            infoclus_obj = build_infoclus(dataset, embedding_name)
            infoclus_obj.optimise(alpha=alpha,beta=beta,min_att=minAtt,runtime_id=runtime_id)
        else:
            print('unknown trigger id')

        return serialize_infoclus(infoclus_obj)

    @app.callback(
        Output('dashboard-content', 'children'),
        Input('infoclus_store', 'data')
    )
    def update_content(serialized_data):
        infoclus_obj = deserialize_infoclus(serialized_data)
        return config_layout(infoclus_obj)


    @app.callback(
        Output('explanation', 'children'),
        Input('cluster-select', 'value'),
        State('infoclus_store', 'data')

    )
    def select_cluster_explanation(cluster_id, serialized_infoclus):
        infoclus = deserialize_infoclus(serialized_infoclus)
        return config_explanations(infoclus, cluster_id)

    @app.callback(
        Output('embedding-scatterPlot', 'figure'),
        Input('embedding-select', 'value'),
        State('infoclus_store', 'data')
    )
    def select_embedding(embedding_name_show, serialized_infoclus):
        infoclus = deserialize_infoclus(serialized_infoclus)
        return config_scatter_graph(infoclus,embedding_name_show)