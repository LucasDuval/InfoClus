import pickle
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from sklearn.neighbors import KernelDensity
from dash import dcc, html
import dash_bootstrap_components as dbc

from infoclus2 import InfoClus
from config import PROJECT_ROOT

# sys.path.append(PROJECT_ROOT)
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    fig.update_layout(xaxis_title="Value",
                      yaxis_title="Densities",
                      showlegend=True,
                      width=600,  # Set the figure width in pixels
                      height=400
                      )

    return fig

def get_barchart(infoclus: InfoClus, att_id: int, cluster_id: int, att_name: str):

    df_mapping_chain = infoclus.ls_mapping_chain_by_col[att_id]
    real_labels = df_mapping_chain.iloc[:,0]
    nuniques = len(df_mapping_chain)
    dist_of_fixed_cluster_att = infoclus._clustersRelatedInfo[cluster_id][0].iloc[:nuniques, att_id].values
    dist_of_att_in_data = infoclus._priors.iloc[:nuniques, att_id].values

    dist_pre_cluster_att = pd.Series(dist_of_fixed_cluster_att, index=real_labels)
    dist_prior_per_att = pd.Series(dist_of_att_in_data, index=real_labels)
    sorted_dist_pre_cluster_att = dist_pre_cluster_att.sort_values(ascending=False)
    sorted_dist_prior_per_att = dist_prior_per_att.loc[sorted_dist_pre_cluster_att.index]
    sorted_labels = sorted_dist_pre_cluster_att.index
    sorted_distribution = []
    types = []
    group_labels = []
    for label in sorted_labels:
        sorted_distribution.append(sorted_dist_pre_cluster_att[label])
        sorted_distribution.append(sorted_dist_prior_per_att[label])
        types.extend(['Cluster', 'Prior'])
        group_labels.extend([label, label])

    data = pd.DataFrame({
        "Labels": group_labels,
        "Distribution": sorted_distribution,
        "Type": types
    })
    fig = px.bar(
        data,
        x="Labels",
        y="Distribution",
        color="Type",
        barmode="group",
        # title=f"Cluster {cluster_id} - Attribute {att_id}",
        labels={"Distribution": "Distribution", "Labels": "Labels"}
    )
    fig.update_layout(
        width=600,
        height=400
    )

    return fig

def config_scatter_graph(infoclus: InfoClus, emb_name: str = None):

    clustering = infoclus._clustering_opt
    embedding = infoclus.all_embeddings[emb_name]

    df = pd.DataFrame({
        'x': embedding[:, 0],  # X coordinates
        'y': embedding[:, 1],  # Y coordinates
        'class': pd.Categorical(clustering)  # Classifications
    })
    #todo: error, change the title to another bar, update with optimise and dataset switch
    fig = px.scatter(df, x='x', y='y', color='class')
    fig.update_layout(
        width=810,
        height=540
    )
    return fig


def config_explanations(infoclus: InfoClus, cluster_label: int = 0):
    """
    :return: kde distributions for all selected features in a cluster, default as 0
    """
    # todo: optimise clustering as instance_cluster_idx in parameter transfer
    # instance_cluster_idx = np.where(clustering == cluster_label)
    instance_cluster_idx = infoclus._clusters_idxes_opt[cluster_label]
    cluster = infoclus.data.iloc[instance_cluster_idx]
    percentage = cluster.size/infoclus.data.size * 100

    figures = []
    figures.append(html.Br())
    figures.append(dbc.Alert("Contains " + format(percentage, '.2f') + ' % of data', color="info"))

    att_names = infoclus.data.columns.values
    ics_cluster = np.array(infoclus._ic_opt[cluster_label])
    for att_id in infoclus._attributes_opt[cluster_label]:
        data_att = infoclus.data.values[:, att_id]
        cluster_att = cluster.values[:, att_id]
        att_name = att_names[att_id]
        if infoclus.global_var_type == 'categorical':
            fig = get_barchart(infoclus, att_id, cluster_label, att_name)
        elif infoclus.global_var_type == 'numeric':
            fig = get_kde(data_att, cluster_att, att_name)
        else:
            print('unsupported attribute type for visualization:', infoclus.global_var_type)

        figures.append(html.H6([att_name, dbc.Badge(format(ics_cluster[att_id], '.1f') + " IC", color="success", className="ml-1")]))
        figures.append(dcc.Graph(id=f"Cluster {cluster_label}, {att_name}",
                                 figure=fig,
                                 config={'displayModeBar': False},
                                 style={
                                     "width": "100%",
                                     "height": "100%",
                                     "marginBottom": "1rem"}
                                 ))

    return figures


def config_hyperparameter_tuning(infoclus: InfoClus):

    alpha = infoclus.alpha
    alpha_max = alpha * 5
    alpha_min = 0

    beta = infoclus.beta
    beta_max = 2
    beta_min = 1

    mina = infoclus.min_att
    mina_max = mina * 5
    mina_min = 0
    maxa = infoclus.max_att
    maxa_max = maxa * 5
    maxa_min = 0

    runid = infoclus.runtime_id
    runid_max = len(RUNTIME_MARKERS)
    runid_min = 0


    return dbc.Row(
        [
            dbc.Col(
                [
                    # Alpha
                    dbc.Row(
                        [
                            dbc.Col(html.H6(u"\u03B1"), width=2),
                            dbc.Col(
                                dcc.Slider(
                                    id='alpha-slider',
                                    min=alpha_min,
                                    max=alpha_max,
                                    step=10,
                                    marks={i: str(i) for i in range(alpha_min, alpha_max, int((alpha_max-alpha_min)/10))},
                                    # todo: align initial values to be the same with initial webpage activation
                                    value=alpha,
                                    tooltip={"always_visible": False}
                                )
                            )
                        ]
                    ),
                    # Beta
                    dbc.Row(
                        [
                            dbc.Col(html.H6(u"\u03B2"), width=2),
                            dbc.Col(
                                dcc.Slider(
                                    id='beta-slider',
                                    min=beta_min,
                                    max=beta_max,
                                    step=0.05,
                                    marks={round(i, 1): format(i, '.1f') for i in
                                           np.arange(beta_min, beta_max, 0.1)},
                                    value=beta,
                                    tooltip={"always_visible": False}
                                )
                            )
                        ]
                    ),
                    # Runtime
                    dbc.Row(
                        [
                            dbc.Col(html.H6("runtime"), width=2),
                            dbc.Col(
                                dcc.Slider(
                                    id='runtime-slider',
                                    min=runid_min,
                                    max=runid_max,
                                    step=1,
                                    marks={i: RUNTIME_MARKERS[i] for i in range(runid_max)},
                                    value=runid,
                                    tooltip={"always_visible": False}
                                )
                            )
                        ]
                    ),
                    # Min Attributes
                    dbc.Row(
                        [
                            dbc.Col(html.H6("minAtt"), width=2),
                            dbc.Col(
                                dcc.Slider(
                                    id='minAtt-slider',
                                    min=mina_min,
                                    max=mina_max,
                                    step=1,
                                    marks={i: str(i) for i in range(mina_max, 1)},
                                    value=mina,
                                    tooltip={"always_visible": False}
                                )
                            )
                        ]
                    ),
                ],
                align="center",
            ),
            dbc.Col(
                [
                    dbc.Row(dbc.Col(
                        # Recalc restarts from scratch
                        dbc.Button("Recalc", color="primary", size="md", id="recalc-hyperparameters")),
                        justify="center"
                    ),
                    dbc.Tooltip(
                        "Restart calculation from scratch",
                        target="recalc-hyperparameters",
                        placement="right"
                    )
                ],
                align="center", width="auto"
            )
        ],
        justify="center"
    )


def config_layout(infoclus: InfoClus, cluster_id: int = 0, datasets_config: str = 'datasets_info.yaml'):

    dataset_name = infoclus.name
    with open(datasets_config, 'r') as file:
        datasets_info = yaml.safe_load(file)

    if dataset_name not in datasets_info['datasets']:
        print("Error! Dataset not found.")


    count_clusters = len(infoclus._clusters_idxes_opt)
    main_emb_name = infoclus.emb_name

    return html.Div([

        # Dashboard with general info
        dbc.Card(
            dbc.CardBody(
                [
                    # Dropdowns
                    dbc.Row(
                        [
                        dbc.Col(
                            dcc.Dropdown(
                                id='dataset-select',
                                options=[
                                    {'label': dataset, 'value': dataset} for dataset in datasets_info['datasets']
                                ],
                                value=dataset_name
                            )
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id='embedding-select',
                                options=[
                                    {'label': embedding, 'value': embedding} for embedding in datasets_info['embeddings']['method']
                                ],
                                value=main_emb_name
                            )
                        )
                    ]),
                    dbc.Row(
                        html.Div([html.Span("embedding used for clustering: ", style={'margin-right': '10px'}),
                                  dcc.Input(id='embedding-used-for-clustering', type='text',
                                            value=f"{infoclus.emb_name}", readOnly=True,
                                            style=top_bar_style),
                                  html.Span("alpha: ", style={'margin-right': '10px'}),
                                  dcc.Input(id='alpha-value', type='text', value=f"{infoclus.alpha}", readOnly=True,
                                            style=top_bar_style),
                                  html.Span("beta: ", style={'margin-right': '10px'}),
                                  dcc.Input(id='beta-value', type='text', value=f"{infoclus.beta}", readOnly=True,
                                            style=top_bar_style),
                                  html.Span("min_att: ", style={'margin-right': '10px'}),
                                  dcc.Input(id='min-att', type='text', value=f"{infoclus.min_att}", readOnly=True,
                                            style=top_bar_style),
                                  html.Span("max_att: ", style={'margin-right': '10px'}),
                                  dcc.Input(id='max-att', type='text', value=f"{infoclus.max_att}", readOnly=True,
                                            style=top_bar_style),
                                  html.Span("run time: ", style={'margin-right': '10px'}),
                                  dcc.Input(id='run time id', type='text', value=f"{RUNTIME_MARKERS[infoclus.runtime_id]}", readOnly=True,
                                            style=top_bar_style),
                                  ]),
                    ),
                    html.Br(),
                    dbc.Row(
                        [
                            # Scatter plot and hyperparameter tuning
                            dbc.Col(
                                [
                                    # Scatter plot
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5(children=dataset_name, className="card-title"),
                                                dcc.Graph(
                                                        id="embedding-scatterPlot",
                                                        figure=config_scatter_graph(infoclus, main_emb_name)
                                                    )
                                            ]
                                        )
                                    ),
                                    html.Br(),
                                    # Hyperparameter tuning
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5(children="Tune hyperparameters", className="card-title"),
                                                config_hyperparameter_tuning(infoclus)
                                            ]
                                        )
                                    ),
                                ],
                                width=7
                            ),

                            # Explanation
                            dbc.Col(
                                [
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5(children="Cluster explanation", className="card-title"),
                                                dcc.Dropdown(
                                                    id='cluster-select',
                                                    options=[
                                                        {'label': "Cluster " + str(i), 'value': i} for i in range(count_clusters)
                                                    ],
                                                    value=0
                                                ),
                                                dbc.Row(id='explanation',
                                                        children=config_explanations(infoclus,cluster_id),
                                                        className="g-3")

                                                # html.Div(config_explanations(infoclus, cluster_id),
                                                #          id="explanation", style=SIDEBAR_STYLE,
                                                #          )
                                            ]
                                        ),
                                    )
                                ],
                            width=5
                            )
                        ], align="start", justify="start", className="g-3"
                    )
                ]
            )
        )
    ])


