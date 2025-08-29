import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from sklearn.neighbors import KernelDensity
from dash import dcc, html
import dash_bootstrap_components as dbc


RUNTIME_MARKERS = ["0.01s", "0.5s", "1s", "5s", "10s", "30s", "1m","3m", "5m", "10m", "30m", "1h"]

SIDEBAR_STYLE = {
    "overflow-y": "scroll",
    "height": "800px"
    # "width": "fit-content"
}

top_bar_style={'display': 'inline-block', 'padding': '5px 10px',
                                                   'background-color': '#e0e0e0', 'border-radius': '5px',
                                                   'font-size': '14px', 'margin-right': '10px'}

KERNALS = ["gaussian", "tophat", "epanechnikov"]
KERNAL = KERNALS[0]

INFOCLUS_OBJ = None

def get_kde(data_att: np.ndarray, cluster_att: np.ndarray, att_name: str):
    """
    :return: return kernal desity estimation of one attribute for a cluster
    """
    percentage = len(cluster_att) / len(data_att)
    # Note: two kde's need to have the same bandwidth to ensure that they are comparable
    kde_data = KernelDensity(kernel='gaussian', bandwidth='scott').fit(data_att.reshape(-1,1))
    kde_cluster = KernelDensity(kernel='gaussian', bandwidth=kde_data.bandwidth_).fit(cluster_att.reshape(-1,1))

    x_vals = np.linspace(min(min(data_att), min(cluster_att)), max(max(data_att), max(cluster_att)), 1000)
    kde_data_vals = np.exp(kde_data.score_samples(x_vals.reshape(-1, 1)))
    kde_cluster_vals = np.exp(kde_cluster.score_samples(x_vals.reshape(-1, 1)))

    cluster_proportion = len(cluster_att) / len(data_att)
    overlap_density = kde_cluster_vals * cluster_proportion

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=kde_data_vals, mode='lines', name=f'kde of {att_name} on full data',
                             line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=x_vals, y=kde_cluster_vals, mode='lines', name=f'kde of {att_name} on cluster',
                             line=dict(color='green', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=x_vals, y=overlap_density, fill='tozeroy', name=f'{percentage}% Overlapped by Cluster',
                             line=dict(color='orange', width=1)))

    fig.update_layout(
        xaxis=dict(
            title="Value",
            showline=True,
            linecolor="gray",
            linewidth=1
        ),
        yaxis=dict(
            title="Densities",
            showline=True,
            linecolor="gray",
            linewidth=1
        ),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0)
    )

    # fig.update_layout(
    #                   showlegend=True,
    #                   width=600,  # Set the figure width in pixels
    #                   height=400
    #                   )

    return fig

def config_scatter_graph(infoc_para_res: dict, embedding: np.ndarray):

    clustering = infoc_para_res['clustering']

    df = pd.DataFrame({
        'x': embedding[:, 0],  # X coordinates
        'y': embedding[:, 1],  # Y coordinates
        'class': pd.Categorical(clustering),  # Classifications
        'customdata': list(range(len(clustering)))
    })
    fig = px.scatter(df, x='x', y='y', color='class', custom_data=['customdata'])
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    fig.update_traces(customdata=df["customdata"])

    return fig

def config_explanations(infoc_para_res: dict, df_data: pd.DataFrame, cluster_label: int = 0):
    """
    :return: kde distributions for all selected features in a cluster, default as 0
    """

    instance_cluster_idx = infoc_para_res['clusters_idxes_opt'][cluster_label]
    cluster = df_data.iloc[instance_cluster_idx]
    percentage = len(instance_cluster_idx)/df_data.shape[0] * 100

    figures = []
    figures.append(html.Br())
    figures.append(dbc.Alert("Contains " + format(percentage, '.2f') + ' % of data', color="info"))

    att_names = df_data.columns
    ics_cluster = np.array(infoc_para_res['ic_opt'][cluster_label])
    for att_id in infoc_para_res['attributes_opt'][cluster_label]:
        data_att = df_data.values[:, att_id]
        cluster_att = cluster.values[:, att_id]
        att_name = att_names[att_id]
        if infoc_para_res['global_arr_type'] == 'categorical':
            # fig = get_barchart(infoclus, att_id, cluster_label, att_name)
            pass
        elif infoc_para_res['global_arr_type'] == 'numeric':
            fig = get_kde(data_att, cluster_att, att_name)
        else:
            print('unsupported attribute type for visualization:', infoc_para_res['global_arr_type'])

        figures.append(html.H6([att_name, dbc.Badge(format(ics_cluster[att_id], '.1f') + " IC", color="success", className="ml-1")]))
        figures.append(dcc.Graph(id=f"Cluster {cluster_label}, {att_name}",
                                 figure=fig,
                                 style = {'width': '100%', 'height': '40%'},
                                 config = {'responsive': True}
                                 )
                       )

    return figures

def config_selected_explanations(infoc_para_res: dict = None, df_data: pd.DataFrame=None, selected_idxes=None, ics_cluster=None, attributes=None):

    if selected_idxes is None:
        return 'exploring dataset by selecting points by lasso in the above scatter plot '

    cluster = df_data.iloc[selected_idxes]
    percentage = len(selected_idxes) / df_data.shape[0] * 100

    figures = []
    figures.append(html.Br())
    figures.append(dbc.Alert("Contains " + format(percentage, '.2f') + ' % of data', color="info"))

    att_names = df_data.columns
    for att_id in attributes:
        data_att = df_data.values[:, att_id]
        cluster_att = cluster.values[:, att_id]
        att_name = att_names[att_id]
        if infoc_para_res['global_arr_type'] == 'categorical':
            # fig = get_barchart(infoclus, att_id, cluster_label, att_name)
            pass
        elif infoc_para_res['global_arr_type'] == 'numeric':
            fig = get_kde(data_att, cluster_att, att_name)
        else:
            print('unsupported attribute type for visualization:', infoc_para_res['global_arr_type'])
        figures.append(html.H6(
            [att_name, dbc.Badge(format(ics_cluster[att_id], '.1f') + " IC", color="success", className="ml-1")]))
        figures.append(dcc.Graph(
                                 figure=fig,
                                 # style={'width': '100%', 'height': '100%'},
                                 config={'responsive': True}
                                 )
                       )

        return figures

def get_dataset_dropdown_items():
    items =[
        {'label': 'cytometry_2500', 'value': 'cytometry_2500'},
        {'label': 'german_socio_eco', 'value': 'german_socio_eco'}
    ]
    return items

def get_embedding_dropdown_items():
    items =[
        {'label': 'tsne', 'value': 'tsne'},
        {'label': 'pca', 'value': 'pca'}
    ]
    return items

def get_runtime_dropdown_items():
    items =[
        {'label': 'recalculate in 1 s', 'value': '1'},
        {'label': 'recalculate in 5 s', 'value': '5'}
    ]
    return items

def get_auxiliary_text_for_clustering():
    return "The clustering result is computed under parameters ..."

def config_layout(infoc_para_res: dict, df_data: pd.DataFrame, embeddings: dict, cluster_id: int = 0, datasets_config: str = 'datasets_info.yaml'):

    dataset_name = infoc_para_res['data_name']
    with open(datasets_config, 'r') as file:
        datasets_info = yaml.safe_load(file)

    if dataset_name not in datasets_info['datasets']:
        print("Error! Dataset not found.")


    count_clusters = infoc_para_res['count_clusters']
    main_emb_name = infoc_para_res['emb_name']

    return html.Div([

        dbc.Row(
            id='layout',
            children=[
                dbc.Col(
                    xs=12,
                    sm=3,
                    md=3,
                    id='selection-panel',
                    children=dbc.Card(
                        dbc.CardBody([
                        html.Div(
                            children=[
                                dbc.Row('Welcome to InfoClus, '
                                        'a new clustering method that also explains its clusters. '
                                        'Play with existed datasets or import your own dataset!',
                                        id='welcome-block'),

                                dbc.Row(
                                    children=[
                                        html.Span(
                                            children=[
                                                'Select dataset: ',
                                                dcc.Dropdown(
                                                    options=get_dataset_dropdown_items(),
                                                    value=dataset_name,
                                                    id='dataset-select',
                                                    style={'width': '15em',
                                                           'display': 'inline-block',
                                                           'verticalAlign': 'middle'
                                                           }
                                                )
                                            ],
                                        ),
                                    ]
                                ),

                                dbc.Row(
                                    children=[
                                        html.Span(
                                            children=[
                                                'Select embedding: ',
                                                dcc.Dropdown(
                                                    options=get_embedding_dropdown_items(),
                                                    value=main_emb_name,
                                                    id='embedding-select',
                                                    style={'width': '13em',
                                                           'display': 'inline-block',
                                                           'verticalAlign': 'middle'},
                                                )
                                            ]
                                        ),
                                    ]
                                ),

                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H5('Hyper-parameters tuning', className='card-title'),
                                            dbc.Row(
                                                [
                                                    dbc.Col('alpha', width='auto'),
                                                    dbc.Col(children=dcc.Slider(
                                                                        id='alpha-slider',
                                                                        min=int(infoc_para_res['alpha']/5),
                                                                        max=int(infoc_para_res['alpha']*5),
                                                                        marks={
                                                                            int(infoc_para_res['alpha'] / 5): {'label': str(int(infoc_para_res['alpha'] / 5))},
                                                                            int(infoc_para_res['alpha'] * 5): {'label': str(int(infoc_para_res['alpha'] * 5))},
                                                                        },
                                                                        step=1,
                                                                        value=infoc_para_res['alpha'],
                                                                        tooltip={"always_visible": True, 'placement': 'bottom'},
                                                                    ),
                                                            ),
                                                ]
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col('beta', width='auto'),
                                                    dbc.Col(children=dcc.Slider(
                                                                        id='beta-slider',
                                                                        min=1,
                                                                        max=2,
                                                                        step=0.1,
                                                                        marks={
                                                                            1: {'label': str(1)},
                                                                            2: {'label': str(2)},
                                                                        },
                                                                        value=infoc_para_res['beta'],
                                                                        tooltip={"always_visible": True, 'placement': 'bottom'}
                                                                    ),
                                                            ),
                                                ]
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col('min_att', width='auto'),
                                                    dbc.Col(children=dcc.Input(type="number", value=2, step=1, min=0,max=10, id='min-att-input'),
                                                            width='auto'),
                                                    dbc.Col('max_att', width='auto'),
                                                    dbc.Col(children=dcc.Input(type="number", value=5, step=1, min=3, max=10, id='max-att-input',),
                                                            width='auto'),
                                                ],
                                            ),
                                            dcc.Dropdown(
                                                value ='1',
                                                options=get_runtime_dropdown_items(),
                                                id = 'recalc-hyperparameters'
                                            )
                                        ]
                                    ),
                                    color='white'
                                )])]),
                        className='h-100')
                ),
                dbc.Col(
                    xs=12,
                    sm=6,
                    md=6,
                    id = 'clustering-panel',
                    children=dbc.Card(
                        dbc.CardBody([
                        html.Div(
                            [
                                dbc.Row(
                                children=[
                                    html.Span(children=[
                                        html.Span('clustering', id='clustering-text', style={'font-style': 'italic'}),
                                        ' shown on embedding ',
                                         dcc.Dropdown(
                                             options=get_embedding_dropdown_items(),
                                             value=main_emb_name,
                                             id='embedding-for-show',
                                             style={'width': '8em',
                                                    'display': 'inline-block',
                                                    'verticalAlign': 'middle'
                                                    }
                                         )
                                    ], ),
                                ],
                                ),
                                dbc.Tooltip(
                                    get_auxiliary_text_for_clustering(),
                                    target='clustering-text',
                                    placement='top'
                                ),
                                dcc.Graph(
                                    id="embedding-scatterPlot",
                                    figure=config_scatter_graph(infoc_para_res, embeddings[main_emb_name]),
                                    style={'width': '100%', 'height': '100%'},
                                    config={"editable": False, "modeBarButtonsToAdd": ["lasso2d", "select2d"]},
                                    # config={'responsive': True}
                                ),
                                dcc.Markdown(
                                    r"$R_{\alpha,\beta}(\mathcal{C}, \mathcal{E}) = \frac{\sum_{i=1}^r{\sum_{j=1}^{|e_i|}{I_i^j}}}{\alpha + (\sum_{i=1}^r{\sum_{j=1}^{|e_i|}{|a_i^j|}})^\beta}$ is ...",
                                    mathjax=True
                                ),
                                dbc.Row(
                                    id = 'selected-explanation',
                                    children=config_selected_explanations()
                                )
                            ],)]),
                        className='h-100')
                ),
                dbc.Col(
                    xs=12,
                    sm=3,
                    md=3,
                    id='explanation-panel',
                    children=dbc.Card(
                        dbc.CardBody([
                        html.Div(
                            [html.Span(
                            [html.H5("Cluster explanation"),
                             dcc.Dropdown(
                                 id='cluster-select',
                                 options=[
                                     {'label': "Cluster " + str(i), 'value': i} for i in range(infoc_para_res['count_clusters'])
                                 ],
                                 value=cluster_id
                             ),]),
                        dbc.Row(id='explanation',
                                children=config_explanations(infoc_para_res, df_data, cluster_id),
                                style={
                                    'height': '78vh',
                                    'overflowY': 'auto'
                                }
                                )]


                        )

                    ]),
                        className='h-100'
                    )
                )
            ]
        )
        ])

#
# def get_barchart(infoclus: InfoClus, att_id: int, cluster_id: int, att_name: str):
#
#     df_mapping_chain = infoclus.ls_mapping_chain_by_col[att_id]
#     real_labels = df_mapping_chain.iloc[:,0]
#     nuniques = len(df_mapping_chain)
#     dist_of_fixed_cluster_att = infoclus._clustersRelatedInfo[cluster_id][0].iloc[:nuniques, att_id].values
#     dist_of_att_in_data = infoclus._priors.iloc[:nuniques, att_id].values
#
#     dist_pre_cluster_att = pd.Series(dist_of_fixed_cluster_att, index=real_labels)
#     dist_prior_per_att = pd.Series(dist_of_att_in_data, index=real_labels)
#     sorted_dist_pre_cluster_att = dist_pre_cluster_att.sort_values(ascending=False)
#     sorted_dist_prior_per_att = dist_prior_per_att.loc[sorted_dist_pre_cluster_att.index]
#     sorted_labels = sorted_dist_pre_cluster_att.index
#     sorted_distribution = []
#     types = []
#     group_labels = []
#     for label in sorted_labels:
#         sorted_distribution.append(sorted_dist_pre_cluster_att[label])
#         sorted_distribution.append(sorted_dist_prior_per_att[label])
#         types.extend(['Cluster', 'Prior'])
#         group_labels.extend([label, label])
#
#     data = pd.DataFrame({
#         "Labels": group_labels,
#         "Distribution": sorted_distribution,
#         "Type": types
#     })
#     fig = px.bar(
#         data,
#         x="Labels",
#         y="Distribution",
#         color="Type",
#         barmode="group",
#         # title=f"Cluster {cluster_id} - Attribute {att_id}",
#         labels={"Distribution": "Distribution", "Labels": "Labels"}
#     )
#     fig.update_layout(
#         width=600,
#         height=400
#     )
#
#     return fig
