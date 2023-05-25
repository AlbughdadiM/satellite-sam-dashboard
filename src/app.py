"""
Author: Mohanad Albughdadi
"""
import base64
import os
import zipfile
import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import dash_leaflet as dl
import pandas as pd
import geopandas as gpd
import shapely
from dash_extensions.javascript import assign
from sam_utils import generate_automatic_mask, sam_prompt_bbox
from utils import (
    shape_to_table_row,
    download_from_wms,
)
from config import WMS_URL, LAYER, IMAGE_FORMAT, WORK_DIR, RESOLUTION

point_to_layer = assign(
    """function(feature, latlng, context){
    const p = feature.properties;
    if(p.type === 'circlemarker'){return L.circleMarker(latlng, radius=p._radius)}
    if(p.type === 'circle'){return L.circle(latlng, radius=p._mRadius)}
    return L.marker(latlng);
}"""
)

DEBUG = True

annotation_types = ["ROI BBox", "Object BBox", "Foreground Point", "Background Point"]

columns = ["type", "x_min", "y_min", "x_max", "y_max"]
models = ["sam_vit_b_01ec64.pth", "sam_vit_h_4b8939.pth", "sam_vit_l_0b3195.pth"]


external_stylesheets = [dbc.themes.BOOTSTRAP, "src/assets/image_annotation_style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


server = app.server

# Define the WMS layer configuration
wms_layer = dl.WMSTileLayer(
    url="https://tiles.maps.eox.at/wms",
    layers="s2cloudless-2020_3857",
    format="image/jpeg",
    transparent=True,
    attribution="Sentinel-2 cloudless layer for 2020 by EOX",
)

# Cards
image_annotation_card = dbc.Card(
    id="imagebox",
    children=[
        dbc.CardHeader(html.H2("Satellite Map")),
        dbc.CardBody(
            [
                dl.Map(
                    children=[
                        dl.LayersControl(
                            [
                                dl.Overlay(dl.TileLayer(), name="OSM", checked=True),
                                dl.Overlay(wms_layer, name="Sentinel-2", checked=True),
                            ],
                            id="layers-control",
                            collapsed=True,
                        ),
                        dl.FeatureGroup(
                            [
                                dl.LocateControl(
                                    options={
                                        "locateOptions": {"enableHighAccuracy": True}
                                    }
                                ),
                                dl.MeasureControl(
                                    position="topleft",
                                    primaryLengthUnit="kilometers",
                                    primaryAreaUnit="sqmeters",
                                    id="measure_control",
                                ),
                                dl.EditControl(
                                    id="edit_control",
                                    draw={
                                        "polyline": False,
                                        "polygon": False,
                                        "circle": False,
                                        "circlemarker": False,
                                    },
                                ),
                            ]
                        ),
                    ],
                    id="map",
                    style={
                        "width": "100%",
                        "height": "80vh",
                        "margin": "auto",
                        "display": "block",
                    },
                    center=[43.6045, 1.4442],
                    zoom=12,
                )
            ],
            id="map-card",
        ),
        dbc.CardFooter(
            [
                dcc.Markdown(
                    "To annotate the above image, select an appropriate label on the right and then draw a "
                    "rectangle with your cursor around the area of the image you wish to annotate.\n\n"
                    "**Choose a different image to annotate**:"
                ),
                dbc.ButtonGroup(
                    [
                        dbc.Button("Previous image", id="previous", outline=True),
                        dbc.Button("Next image", id="next", outline=True),
                    ],
                    size="lg",
                    style={"width": "100%"},
                ),
            ]
        ),
    ],
)

annotated_data_card = dbc.Card(
    [
        dbc.CardHeader(html.H2("Annotated data")),
        dbc.CardBody(
            [
                dbc.Row(dbc.Col(html.H3("Coordinates of annotations"))),
                dbc.Row(
                    dbc.Col(
                        [
                            dash_table.DataTable(
                                id="annotations-table",
                                columns=[
                                    dict(
                                        name=n,
                                        id=n,
                                        presentation=(
                                            "dropdown" if n == "type" else "input"
                                        ),
                                    )
                                    for n in columns
                                ],
                                editable=True,
                                style_data={"height": 40},
                                style_cell={
                                    "overflow": "visible",
                                    "textOverflow": "ellipsis",
                                    "maxWidth": 0,
                                },
                                dropdown={
                                    "type": {
                                        "options": [
                                            {"label": o, "value": o}
                                            for o in annotation_types
                                        ],
                                        "clearable": False,
                                    }
                                },
                                style_cell_conditional=[
                                    {
                                        "if": {"column_id": "type"},
                                        "textAlign": "left",
                                    }
                                ],
                                fill_width=True,
                                css=[
                                    {
                                        "selector": ".Select-menu-outer",
                                        "rule": "display: block !important",
                                    }
                                ],
                            ),
                        ],
                    ),
                ),
            ]
        ),
    ]
)
model_card = dbc.Card(
    [
        dbc.CardHeader([html.H2("SAM Configuration")]),
        dbc.CardBody(
            [
                dbc.Row(
                    dbc.Col(
                        [
                            dbc.Stack(
                                [
                                    html.H4("Model Type"),
                                    dcc.Dropdown(
                                        id="sam-model",
                                        options=[
                                            {
                                                "label": "_".join(t.split("_")[0:3]),
                                                "value": t,
                                            }
                                            for t in models
                                        ],
                                        value=models[0],
                                        clearable=False,
                                    ),
                                    html.H4("Predicition IoU Threshold"),
                                    dcc.Input(
                                        id="pred-iou-thresh",
                                        type="number",
                                        placeholder="IoU threshold: input between 0 and 1",
                                        min=0.0,
                                        max=1.0,
                                        step=0.01,
                                        value=0.88,
                                    ),
                                    html.H4("Stability Score Threshold"),
                                    dcc.Input(
                                        id="stability-score-thresh",
                                        type="number",
                                        placeholder="Stability score threshold: input between 0 and 1",
                                        min=0.0,
                                        max=1.0,
                                        step=0.01,
                                        value=0.95,
                                    ),
                                ],
                                gap=3,
                            )
                        ],
                        align="center",
                    )
                ),
            ]
        ),
        dbc.CardFooter(
            [
                html.Div(
                    [
                        dbc.Button(
                            "Segment ROI",
                            id="segment-button",
                            outline=True,
                            color="primary",
                            n_clicks=0,
                            style={"horizontalAlign": "left"},
                            className="me-md-2",
                        ),
                        html.Div(id="dummy1", style={"display": "none"}),
                        dbc.Tooltip(
                            "You can run the SAM model by clicking on this button",
                            target="segment-button",
                        ),
                        dbc.Button(
                            "Download Results",
                            id="download-button",
                            outline=True,
                            color="info",
                            n_clicks=0,
                            disabled=True,
                            style={"horizontalAlign": "middle"},
                            className="me-md-2",
                        ),
                        html.Div(id="dummy2", style={"display": "none"}),
                        dbc.Tooltip(
                            "You can download the results by clicking here",
                            target="download-button",
                        ),
                        dcc.Download(
                            id="download-link",
                        ),
                        dbc.Button(
                            "Refresh",
                            id="refresh-button",
                            outline=True,
                            color="success",
                            n_clicks=0,
                            style={"horizontalAlign": "right"},
                            className="me-md-2",
                        ),
                        dbc.Tooltip(
                            "Click here to refresh page and segment a new zone",
                            target="refresh-button",
                        ),
                        dcc.Location(id="url", refresh=True),
                    ],
                    className="d-grid gap-2 d-md-flex justify-content-center",
                )
            ]
        ),
    ],
)
# Navbar
navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.NavbarBrand("Grounding DINO and SAM for Satellite Images")
                    ),
                ],
                align="center",
            ),
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
    className="mb-5",
)

app.layout = html.Div(
    [
        dbc.Spinner(
            id="loading-1",
            type="grow",
            color="success",
            children=[
                dcc.Store(id="downloaded_image_path"),
                dcc.Store(id="prev-table-data"),
                navbar,
                dbc.Container(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    image_annotation_card,
                                    md=7,
                                ),
                                dbc.Col(
                                    dbc.Stack([annotated_data_card, model_card]),
                                    md=5,
                                ),
                            ],
                        ),
                    ],
                    fluid=True,
                ),
            ],
        )
    ]
)


@app.callback(
    Output("url", "href"),
    [Input("refresh-button", "n_clicks"), Input("downloaded_image_path", "data")],
    prevent_initial_call=True,
)
def refresh_page(n_clicks, downloaded_imgs):
    if n_clicks is not None and n_clicks > 0:
        if downloaded_imgs is not None:
            for img in downloaded_imgs:
                os.remove(img)
        return "/"
    return dash.no_update


@app.callback(
    Output("prev-table-data", "data"),
    [Input("edit_control", "geojson"), State("annotations-table", "data")],
)
def get_polygons(geojson_data, prev_data):
    if not geojson_data:
        raise PreventUpdate
    if not geojson_data["features"]:
        raise PreventUpdate
    gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
    gdf.set_geometry = gdf["geometry"]
    annotations_table_data = [shape_to_table_row(geom.bounds) for geom in gdf.geometry]
    annotations_table_data = [
        {**annotations_table_data[i], "type": "Object BBox"}
        for i in range(len(annotations_table_data))
    ]
    gdf_bounds = gdf.total_bounds
    gdf_bounds = shapely.geometry.box(*gdf_bounds).buffer(0.005).bounds

    bounds_row = shape_to_table_row(gdf_bounds)
    bounds_row["type"] = "ROI BBox"
    annotations_table_data.append(bounds_row)
    if prev_data is not None:
        prev_data = [d for d in prev_data if d["type"] != "ROI BBox"]
        intersection = [
            d1
            for d1 in prev_data
            for d2 in annotations_table_data
            if all(d1[key] == d2[key] for key in columns[1:])
        ]

        annotations_table_data = [
            d2 for d2 in annotations_table_data if d2 not in intersection
        ]

        annotations_table_data = annotations_table_data + prev_data

    return annotations_table_data


@app.callback(
    [
        Output("annotations-table", "data"),
    ],
    Input("prev-table-data", "data"),
)
def update_table(prev_data):
    if not prev_data:
        raise PreventUpdate
    return [prev_data]


@app.callback(
    [
        Output("downloaded_image_path", "data"),
        Output("segment-button", "disabled"),
        Output("map-card", "children"),
        Output("download-button", "disabled"),
    ],
    [
        Input("annotations-table", "data"),
        Input("segment-button", "n_clicks"),
        Input("sam-model", "value"),
        Input("pred-iou-thresh", "value"),
        Input("stability-score-thresh", "value"),
    ],
)
def run_segmentation(
    table_data, n_clicks, sam_model, pred_iou_thresh, stability_score_thresh
):
    if n_clicks == 0 or table_data is None:
        raise PreventUpdate
    if n_clicks == 1 and table_data is not None:
        roi = [row for row in table_data if row["type"] == "ROI BBox"][0]
        roi_bbox = [
            float(roi["x_min"]),
            float(roi["y_min"]),
            float(roi["x_max"]),
            float(roi["y_max"]),
        ]
        tmp_img_path = download_from_wms(
            WMS_URL, roi_bbox, LAYER, IMAGE_FORMAT, WORK_DIR, RESOLUTION
        )
        types = [row["type"] for row in table_data]
        unique_types = list(set(types))
        if len(table_data) == 2 and unique_types == ["ROI BBox"]:
            segmetnation_path, png_path = generate_automatic_mask(
                tmp_img_path, sam_model, pred_iou_thresh, stability_score_thresh
            )
        else:  # len(table_data)>=2 and unique_types!=["bounding box"]:
            geom_df = pd.DataFrame(table_data)
            bboxes_df = geom_df.loc[geom_df["type"] != "ROI BBox"]
            bboxes_geo = bboxes_df.iloc[:, :4].astype(float).values
            segmetnation_path, png_path = sam_prompt_bbox(
                tmp_img_path, bboxes_geo, sam_model, roi_bbox
            )

        image_bounds = [[roi_bbox[1], roi_bbox[0]], [roi_bbox[3], roi_bbox[2]]]
        encoded_img = base64.b64encode(open(png_path, "rb").read()).decode("ascii")
        encoded_img = "{}{}".format("data:image/png;base64, ", encoded_img)
        list_children = dl.Map(
            [
                # dl.TileLayer(),
                wms_layer,
                dl.ImageOverlay(opacity=0.5, url=encoded_img, bounds=image_bounds),
                dl.FeatureGroup(
                    [
                        dl.LocateControl(
                            options={"locateOptions": {"enableHighAccuracy": True}}
                        ),
                        dl.MeasureControl(
                            position="topleft",
                            primaryLengthUnit="kilometers",
                            primaryAreaUnit="sqmeters",
                            id="measure_control",
                        ),
                        dl.EditControl(
                            id="edit_control",
                            draw={
                                "polyline": False,
                                "polygon": False,
                                "circle": False,
                                "circlemarker": False,
                            },
                        ),
                    ]
                ),
            ],
            id="map",
            style={
                "width": "100%",
                "height": "80vh",
                "margin": "auto",
                "display": "block",
            },
            center=[
                (float(roi["y_min"]) + float(roi["y_max"])) / 2.0,
                (float(roi["x_min"]) + float(roi["x_max"])) / 2.0,
            ],
            zoom=15,
            bounds=image_bounds,
        )
        return [tmp_img_path, segmetnation_path, png_path], True, list_children, False


@app.callback(
    Output("download-link", "data"),
    [Input("download-button", "n_clicks"), Input("downloaded_image_path", "data")],
)
def prepare_downloadble(n_clicks, download_data):
    if n_clicks == 0 or download_data is None:
        raise PreventUpdate

    def write_archive(memory_file):
        # memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, "a", zipfile.ZIP_DEFLATED) as z_file:
            for file_path in download_data:
                z_file.write(file_path)
        memory_file.seek(0)

    return dcc.send_bytes(
        write_archive,
        os.path.basename(download_data[0]).split(".")[0] + ".zip",
    )


if __name__ == "__main__":
    app.run_server(debug=DEBUG)
