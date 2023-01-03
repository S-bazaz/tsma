# -*- coding: utf-8 -*-
"""
functions used to creat a dashboard on a local server in order to visualize 
a summary of a simulation with different sets of parameters

"""
__author__ = ["Samuel Bazaz"]
__credits__ = ["Samuel Bazaz"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Samuel Bazaz"]

##############
#  Packages  #
##############

import os
import sys

import time
import dash_bootstrap_components as dbc
import plotly.io as pio
from dash import (
    Dash,
    dcc,
    html,
    Input,
    Output,
    State,
    ALL,
    callback_context,
    dependencies,
)
from jupyter_dash import JupyterDash

#################
#  Importation  #
#################

path = os.path.dirname(os.getcwd())
sys.path.append(path)

from basics.transfers import position_inputs, numerical_sub_dict
from basics.text_management import (
    encode,
    decode,
    gross_f_latex,
    dict_to_html,
    list_to_lines,
)
from collect.output_management import get_save_path
from visuals.fig_management import save_fig, adapted_mos_default, okun_phillips_f_img


#########################
#  Custom html objects  #
#########################


def myslider(name: str, minv: float, maxv: float, n: int, v0: float) -> dcc.Slider:
    """ Create a dcc.Slider in a standard format for dashboard construction
    
    Parameters
    ----------
    name : string,
        name of the parameter associated with the Slider
    minv : float,
        minimum value of the Slider
    maxv : float,
        maximum value of the Slider
    n : int,
        number of steps 
    v0 : float,
        initial value of the Slider 
        
    Returns
    -------
    slider : dcc.Slider html_object
    
    """
    mid = (maxv - minv) / 2
    slider = dcc.Slider(
        id={"type": "slider", "index": name},
        min=minv,
        max=maxv,
        step=((maxv - minv) / n),
        marks={minv: str(minv), mid: str(mid), maxv: str(maxv),},
        value=v0,
        tooltip={"always_visible": True, "placement": "bottomLeft"},
    )
    return slider


def myinput(name: str, v0: float) -> dbc.Input:
    """ Create a dbc.Input in a standard format for dashboard construction
    
    Parameters
    ----------
    name : str
        name of the parameter associated with the Slider
    
    v0 : float
        initial value of the Slider 
        
    Returns
    -------
    dbc_input :dbc.Input
        a dbc.Input html object
    """
    dbc_input = dbc.Input(
        className="mb-4 bg-light text-center",
        size="sm",
        id={"type": "input", "index": name},
        type="number",
        placeholder="input number",
        value=v0,
    )
    return dbc_input


def mybutton(name: str) -> html.Button:
    """ Create a dbc.Button in a standard format for dashboard construction
    Parameters
    ----------
    name : str
        name of the parameter associated with the Slider
    
    Returns
    -------
    html.Button : dbc.Button html_object
    
    """
    return dbc.Button(
        name.replace("-", " "),
        id=name,
        color="secondary",
        className="me-1",
        outline=True,
        size="sm",
    )


def myradio(name: str, lst_values: list[str]) -> html.Div:
    """
    Create a dbc.RadioItems with it's label in a standard format for dashboard construction.
    
    Parameters
    ----------
    name : string
        Name of the parameter associated with the Slider.
    lst_values : list string
        List of the possible values of the radioitems.
        
    Returns
    -------
    html.Div : dbc.RadioItems html_object
    
    """
    lst_child = [
        html.H5(name.replace("-", " ")),
        dbc.RadioItems(
            className="",
            id=name,
            options=[{"label": a, "value": a} for a in lst_values],
            value=lst_values[0],
            inline=True,
            labelCheckedClassName="text-success",
            inputCheckedClassName="border border-success bg-success",
        ),
    ]
    return html.Div(lst_child)


def mydropdown(name: str, items: list) -> dbc.DropdownMenu:
    """Create a dbc.DropdownMenu in a standard format for dashboard construction
    
    Parameters
    ----------
    name : string
        name of the parameter associated with the DropdownMenu
    items : list of strings or tuples
        list of items to be displayed in the DropdownMenu. Each item can be either a string
        or a tuple with two strings, where the first string is the item label and the second
        string is the item value.
    
    Returns
    -------
    dropdown : dbc.DropdownMenu html_object
        a DropdownMenu object
    """
    return dbc.DropdownMenu(
        items, label=name, color="secondary", className="me-1", size="sm"
    )


##########################
#  Columns constructors  #
##########################


def columns_of_sliders(
    lst_var: list, params: dict, params_sliders: dict, f_latex=lambda x: x
) -> list:
    """
    Create a list of html objects to build a column of Sliders with labels on top.
                               
    Parameters
    ----------
    lst_var: list,
        list of variables in the column
    params: dict,
        all parameters in undimensional form
    params_sliders: dict,
        parameters of the Sliders {var: [min, max, n]}
        
    Returns
    -------
    lst_res: list of html_objects, 
    """
    lst_res = []
    for var in lst_var:

        lst_dec = var.split("__")
        lst_dec = lst_dec[1:]
        var_name = f_latex(var)
        lst_res.append(
            dcc.Markdown(
                var_name,
                mathjax=True,
                style={"text-align": "center", "margin-top": "20px"},
            )
        )
        lst_res.append(
            myslider(
                var,
                params_sliders[var][0],
                params_sliders[var][1],
                params_sliders[var][2],
                params[var],
            )
        )
    return lst_res


def columns_of_inputs(
    lst_var: list[str], params: dict[str, float], f_latex=lambda x: x
) -> list:
    """ Create a list of html object in order to build a column of inputs with labels on the top

    Parameters
    ----------
    lst_var : list of str,
        list of the variables in the column
    params : dict,
        all parameters in undimensional form
    f_latex : function, optional
        function to convert the variable name into LaTeX string,
        by default lambda x: x
        
    Returns
    -------
    lst_res : list of html_object, 
    """
    lst_res = []
    for var in lst_var:
        lst_dec = var.split("__")
        lst_dec = lst_dec[1:]
        var_name = f_latex(var)

        lst_res.append(
            dcc.Markdown(
                var_name,
                mathjax=True,
                style={"text-align": "center", "margin-top": "20px"},
            )
        )
        lst_res.append(myinput(var, params[var]))
    return lst_res


##########################
#  Sidebar constructors  #
##########################


def top_buttons(lst_name: list) -> html.Div:
    """
    Create a block of buttons
                               
    Parameters
    ----------
    lst_name : list
        List of the names of the buttons
        
    Returns
    -------
    Div
        An html_object containing the buttons.
    """
    lst_button = []
    for var in lst_name:
        lst_button.append(mybutton(var))
    return html.Div(children=lst_button)


def sidebar_container(
    params: dict, dct_id: dict, ncol: int, func_col, id_container: str
) -> html.Div:
    """
    Create a container of the controls of the parameters, based on a function `func_col`.
    ex: func_col(col) = columns_of_sliders(col, params, params_sliders)
                               
    Parameters
    ----------
    params : dict
        All parameters in undimensional form.
    dct_id : dict
        Give the name of the initial dictionary,
        ex { "1": "Parameters", "2": "Hyper_Parameters", "3":"Initial Values"}
    ncol : int
        Number of columns of the sidebar
    func_col : function
        ex: col -> columns_of_sliders(col, params, params_sliders)
    id_container : str
        ex: slider-container
        
    Returns
    -------
    Div
        An html_object containing the controls.
    """
    lst_block = []
    for block in position_inputs(params, ncol):
        lst_col = []
        for col in block:
            lst_col.append(
                html.Div(
                    children=func_col(col),
                    style={
                        "width": (str(100 / ncol) + "%"),
                        "display": "inline-block",
                        "vertical-align": "top",
                    },
                )
            )

            if col != []:
                block_name = dct_id[col[0][1]]
        lst_label = [html.Hr(), html.H5(block_name), html.Hr()]

        lst_block.append(html.Div(children=(lst_label + lst_col)))
    lst_block.append(html.H1("."))
    lst_block.append(html.H1("."))
    lst_block.append(html.Hr(style={"margin-left": "50px"}))

    return html.Div(lst_block, id=id_container)


################################
#  html structure constructor  #
################################


def html_struct(
    num_params: dict,
    params_sliders: dict,
    ncol: int,
    title: str,
    f_latex=lambda x: x,
    var_compared: str = "unemployment_rate",
    has_image: bool = False,
) -> html.Div:
    """
    Final function of the dashboard layout build. 
    It monitors the application's global html structure.
    
    Parameters
    ----------
    num_params : dict
        All parameters in undimensional form.
    params_sliders : dict
        Parameters of the Sliders { var : [min, max, n] }
    ncol : int
        Number of columns of sliders.
    title : str
        Title of the app.
    f_latex : function
        Function to transform the labels into LaTeX format.
    var_compared : str
        Variable to be compared.
    has_image : bool
        Whether to display an image.
        
    Returns
    -------
    Div
        An html_object containing the layout of the dashboard.
    """

    dct_id = {"1": "Parameters", "2": "Hyper Parameters", "3": "Initial Values"}
    lst_container = [
        top_buttons(["update", "save-fig", "save-data"]),
        dbc.Checklist(
            options=[{"label": "Overwrite", "value": True}],
            value=False,
            id="overwrite",
            switch=True,
        ),
        html.Hr(),
        myradio("fig-format", ["html", "png", "pdf", "jpeg"]),
        html.Hr(),
        dbc.Checklist(
            options=[{"label": var_compared, "value": True}],
            value=False,
            id=var_compared,
            switch=True,
        ),
        html.Hr(),
        myradio("parameters-mode", ["inputs", "sliders"]),
        sidebar_container(
            num_params,
            dct_id,
            ncol,
            lambda col: columns_of_inputs(col, num_params, f_latex),
            "inputs-container",
        ),
    ]

    if len(params_sliders) > 0:
        lst_container.append(
            sidebar_container(
                num_params,
                dct_id,
                ncol,
                lambda col: columns_of_sliders(
                    col, num_params, params_sliders, f_latex
                ),
                "slider-container",
            )
        )
    else:
        lst_container.append(html.Div([], id="slider-container"))
    sidebar_style = {
        "overflow": "scroll",
        "position": "fixed",
        "top": 0,
        "right": 0,
        "bottom": 0,
        # 'width': (str(6*ncol)+'%'),
        "width": "14rem",
        "padding": "2rem 1rem",
        "flex": 1,
        "background-color": "#f8f9fa",
        "display": "inline-block",
    }

    sidebar = html.Div(lst_container, style=sidebar_style)

    lst_fig = [dcc.Graph(id="graph")]
    if has_image:
        lst_fig.append(
            html.Img(id="html-img", src="", style={"height": "100%", "width": "100%"})
        )
    figures = html.Div(lst_fig)
    div_res = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.H1(title, className="text-center text-primary"), width=12
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        figures,
                        style={"overflow": "scroll"},
                        # width={ 'size': 9, 'offset':0, 'order': 1}
                        xs=12,
                        sm=12,
                        md=11,
                        lg=11,
                        xl=11,
                    ),
                    dbc.Col(sidebar, width={"size": 1, "offset": 0, "order": 2}),
                ],
                align="start",
            ),
        ],
        fluid=True,
    )

    return div_res


######################
#  app constructors  #
######################


def init_app(
    parameters: dict,
    hyper_parameters: dict,
    initial_values: dict,
    model,
    in_jupyter: bool = False,
) -> tuple[str, dict, dict, str, [JupyterDash, Dash], list[str]]:
    """
    Initialize the dashboard app.
    
    Parameters
    ----------
    parameters : dict
        Parameters of the model.
    hyper_parameters : dict
        Hyperparameters of the model.
    initial_values : dict
        Initial values of the model.
    model : class
        Model class.
    in_jupyter : bool
        Whether the app is being run in a Jupyter notebook.
        
    Returns
    -------
    str
        Name of the model.
    dict
        All parameters in undimensional form.
    dict
        Numerical parameters.
    str
        Path to the figures folder.
    Union[JupyterDash, Dash]
        The dashboard app.
    List[str]
        List of agent IDs.
    """
    m = model(parameters, hyper_parameters, initial_values)
    params = encode(parameters, hyper_parameters, initial_values, m.agent_ids)
    num_params = numerical_sub_dict(params)
    path_figures = os.sep.join([get_save_path(m, False), "figures"])

    if in_jupyter:
        f_dash = JupyterDash
    else:
        f_dash = Dash
        
    app = f_dash(
        __name__,
        external_stylesheets=[dbc.themes.LUX],  # SLATE
        meta_tags=[
            {"name": "viewport", "content": "with=device_with initial-scale=1.0"},
        ],
        assets_folder=path_figures,
    )

    return m.name, params, num_params, path_figures, app, m.agent_ids


def run_f_img(
    m,
    fig_format: str,
    path_figures: str,
    bool_update: bool,
    overwrite_data: bool,
    img_id: dict,
    app: [JupyterDash, Dash],
    f_img,
) -> str:
    """
    Update or save an image file for a given Dashboard and return its url.
    
    Parameters
    ----------
    m : Model
        The model instance.
    fig_format : str
        Format of the figure.
    path_figures : str
        Path to the figures folder.
    bool_update : bool
        Whether to update the figure.
    overwrite_data : bool
        Whether to overwrite the data.
    img_id : dict
        Dictionary containing the image ID.
    app : Union[JupyterDash, Dash]
        The dashboard app.
    f_img : function
        Function to generate the image.
        
    Returns
    -------
    str
        The source URL for the image.
    """

    img_format = "png" if fig_format == "html" else fig_format
    save_path = (
        path_figures
        + "/Dash_img_"
        + (f_img.__name__.split("_"))[0].upper()
        + f"_{m.sim_id}_{img_id['img_id']}.{img_format}"
    )

    if bool_update and ((not os.path.exists(save_path)) or overwrite_data):
        f_img(m, save_path, img_format)
    if os.path.exists(save_path):
        src = app.get_asset_url(save_path.split("/")[-1])
    else:
        src = ""
    return src


###########################
#  Dashboard application  #
###########################


def app_mosaique(
    parameters: dict,
    hyper_parameters: dict,
    initial_values: dict,
    params_sliders: dict,
    model,
    dct_groups: dict,
    var_compared: str = "unemployment_rate",
    in_jupyter: bool = False,
    ncol_latex: int = 2,
    f_latex=gross_f_latex,
    f_fig=adapted_mos_default,
) -> [JupyterDash, Dash]:
    """
    Final function to create the application.
    It controls the plots and the update system.
    
    Parameters
    ----------
    parameters: dict
        a dictionary of model parameters
    hyper_parameters: dict
        a dictionary of hyperparameters
    initial_values: dict
        a dictionary of initial values for the model
    params_sliders: dict
        a dictionary of parameters of the sliders {var: [min, max, n]}
    model: Callable[[dict, dict, dict], Model]
        a model object, for example: threesector()
    dct_groups: dict
        a dictionary of groups {key: value}
    var_compared: str, optional
        the name of the variable to compare, 
        default is "unemployment_rate"
    in_jupyter: bool, optional
        a flag indicating whether the application is being run in a Jupyter notebook,
        default is False
    ncol_latex: int, optional
        the number of columns in the LaTeX representation of the parameters,
        default is 2
    f_latex: function, optional
        a function to convert the parameter names to LaTeX, default is gross_f_latex
    f_fig: function, optional
        a function to create a figure for the model
    
    Returns
    -------
    app : Dash or JupyterDash object
    
    """
    ## model and parameters initialization-----------------------------------------------------
    mod_name, params, num_params, path_figures, app, agent_ids = init_app(
        parameters,
        hyper_parameters,
        initial_values,
        model,
        in_jupyter=in_jupyter,
    )

    app.layout = html_struct(
        num_params,
        params_sliders,
        ncol_latex,
        mod_name,
        f_latex,
        var_compared=var_compared,
    )
    dct_groups2 = {}
    dct_groups2_comp = {}
    for k, v in dct_groups.items():
        if v != [var_compared]:
            dct_groups2[k] = v
            dct_groups2_comp[k] = v + [var_compared]
    ## update management------------------------------------------------------------------------

    @app.callback(
        [
            dependencies.Output("slider-container", "style"),
            dependencies.Output("inputs-container", "style"),
        ],
        [dependencies.Input("parameters-mode", "value"),],
    )
    def toggle_container(toggle_value):
        """ 
        Show Sliders or Inputs based on parameters' mode
        
        """
        if toggle_value == "sliders" and len(params_sliders) > 0:
            return {"display": "block"}, {"display": "none"}
        return {"display": "none"}, {"display": "block"}

    @app.callback(
        Output("graph", "figure"),
        Input("update", "n_clicks"),
        Input("save-fig", "n_clicks"),
        Input("save-data", "n_clicks"),
        Input("overwrite", "value"),
        Input(var_compared, "value"),
        Input("parameters-mode", "value"),
        Input("fig-format", "value"),
        Input({"type": "slider", "index": ALL}, "value"),
        Input({"type": "input", "index": ALL}, "value"),
        State({"type": "slider", "index": ALL}, "id"),
        State({"type": "input", "index": ALL}, "id"),
    )
    def update_save_figures(
        update_clicks,
        save_fig_clicks,
        save_data_clicks,
        overwrite_data,
        compare,
        container,
        fig_format,
        values_slider,
        values_inputs,
        ids_sliders,
        ids_inputs,
    ):
        """ 
        Main callback function. 
        
        """

        changed_id = [p["prop_id"] for p in callback_context.triggered][0]
        update = "update" in changed_id
        save_data = "save-data" in changed_id
        b_save_fig = "save-fig" in changed_id
        bool_update = update or save_data or b_save_fig

        # Parameters updates----------------------------------------------------------------
        if bool_update:
            if container == "sliders" and len(params_sliders) > 0:
                for (i, value) in enumerate(values_slider):
                    params[ids_sliders[i]["index"]] = value
            else:
                for (i, value) in enumerate(values_inputs):
                    params[ids_inputs[i]["index"]] = value
        # Simulation, data saves and queries-------------------------------------------------
        m = model(*decode(params, agent_ids))
        start = time.time()
        output = m.simulate(overwrite=overwrite_data, save=save_data)
        end = time.time()

        # Mosaique update--------------------------------------------------------------------

        if compare:
            grouping = dct_groups2_comp
        else:
            grouping = dct_groups2
        dct1 = {"Simulation ": m.sim_id}
        dct2 = {
            "updated ": bool_update,
            "nupdates": update_clicks,
            "time": (round(end - start, 2)),
        }
        title = list_to_lines(
            dict_to_html(dct1, k_dec="b") + dict_to_html(dct2), ncol=1
        )

        fig = f_fig(output, title, grouping, var_compared)

        # Images update and backup--------------------------------------------------------------

        # Mosaique saves
        if b_save_fig:
            save_fig(fig, path_figures, "mosaique " + m.sim_id, fig_format)
        return fig

    return app


def app_mos_and_image(
    parameters: dict,
    hyper_parameters: dict,
    initial_values: dict,
    params_sliders: dict,
    model: object,
    dct_groups: dict,
    var_compared: str = "unemployment_rate",
    in_jupyter: bool = False,
    ncol_latex: int = 2,
    f_latex=gross_f_latex,
    f_fig=adapted_mos_default,
    f_img=okun_phillips_f_img,
) -> object:
    """
    Final function to create the application.
    It controls the plots and the update system.
    
    Parameters
    ----------
    parameters: dict
        a dictionary of model parameters
    hyper_parameters: dict
        a dictionary of hyperparameters
    initial_values: dict
        a dictionary of initial values for the model
    params_sliders: dict
        a dictionary of parameters of the sliders {var: [min, max, n]}
    model: Callable[[dict, dict, dict], Model]
        a model object, for example: threesector()
    dct_groups: dict
        a dictionary of groups {key: value}
    var_compared: str, optional
        the name of the variable to compare, 
        default is "unemployment_rate"
    in_jupyter: bool, optional
        a flag indicating whether the application is being run in a Jupyter notebook,
        default is False
    ncol_latex: int, optional
        the number of columns in the LaTeX representation of the parameters,
        default is 2
    f_latex: function, optional
        a function to convert the parameter names to LaTeX, default is gross_f_latex
    f_fig: function, optional
        a function to create a figure for the model
    f_img: function, optional
        a function to save an image for the model
    
    Returns
    -------
    app : Dash or JupyterDash object
    
    """
    ## model and parameters initialization-----------------------------------------------------
    mod_name, params, num_params, path_figures, app, agent_ids = init_app(
        parameters,
        hyper_parameters,
        initial_values,
        model,
        in_jupyter=in_jupyter,
    )
    app.layout = html_struct(
        num_params,
        params_sliders,
        ncol_latex,
        mod_name,
        f_latex,
        var_compared=var_compared,
        has_image=True,
    )

    dct_groups2 = {}
    dct_groups2_comp = {}
    for k, v in dct_groups.items():
        if v != [var_compared]:
            dct_groups2[k] = v
            dct_groups2_comp[k] = v + [var_compared]
    # img_id = 0
    img_id = {"img_id": 0}
    ## update management------------------------------------------------------------------------

    @app.callback(
        [
            dependencies.Output("slider-container", "style"),
            dependencies.Output("inputs-container", "style"),
        ],
        [dependencies.Input("parameters-mode", "value"),],
    )
    def toggle_container(toggle_value):
        """ 
        Show Sliders or Inputs based on parameters' mode
        
        """
        if toggle_value == "sliders" and len(params_sliders) > 0:
            return {"display": "block"}, {"display": "none"}
        return {"display": "none"}, {"display": "block"}

    @app.callback(
        Output("graph", "figure"),
        Output("html-img", "src"),
        Input("update", "n_clicks"),
        Input("save-fig", "n_clicks"),
        Input("save-data", "n_clicks"),
        Input("overwrite", "value"),
        Input(var_compared, "value"),
        Input("parameters-mode", "value"),
        Input("fig-format", "value"),
        Input({"type": "slider", "index": ALL}, "value"),
        Input({"type": "input", "index": ALL}, "value"),
        State({"type": "slider", "index": ALL}, "id"),
        State({"type": "input", "index": ALL}, "id"),
    )
    def update_save_figures(
        update_clicks,
        save_fig_clicks,
        save_data_clicks,
        overwrite_data,
        compare,
        container,
        fig_format,
        values_slider,
        values_inputs,
        ids_sliders,
        ids_inputs,
    ):
        """ 
        Main callback function. 
        
        """

        changed_id = [p["prop_id"] for p in callback_context.triggered][0]
        update = "update" in changed_id
        save_data = "save-data" in changed_id
        b_save_fig = "save-fig" in changed_id
        bool_update = update or save_data or b_save_fig

        # Parameters updates----------------------------------------------------------------

        if bool_update:
            img_id["img_id"] += 1
            # img_id +=1
            if container == "sliders" and len(params_sliders) > 0:
                for (i, value) in enumerate(values_slider):
                    params[ids_sliders[i]["index"]] = value
            else:
                for (i, value) in enumerate(values_inputs):
                    params[ids_inputs[i]["index"]] = value
        # Simulation, data saves and queries-------------------------------------------------
        m = model(*decode(params, agent_ids))
        start = time.time()
        output = m.simulate(overwrite=overwrite_data, save=save_data)
        end = time.time()

        # Mosaique update--------------------------------------------------------------------

        if compare:
            grouping = dct_groups2_comp
        else:
            grouping = dct_groups2
        dct1 = {"Simulation ": m.sim_id}
        dct2 = {
            "updated ": bool_update,
            "nupdates": update_clicks,
            "time": (round(end - start, 2)),
        }
        title = list_to_lines(
            dict_to_html(dct1, k_dec="b") + dict_to_html(dct2), ncol=1
        )

        fig = f_fig(output, title, grouping, var_compared)

        # Images update and backup--------------------------------------------------------------

        src = run_f_img(
            m, fig_format, path_figures, bool_update, overwrite_data, img_id, app, f_img
        )

        # Mosaique saves
        if b_save_fig:
            save_fig(fig, path_figures, "mosaique " + m.sim_id, fig_format)
        return fig, src

    return app


#############
#  run app  #
#############


def run_app(
    app: [JupyterDash, Dash],
    in_jupyter: bool = False,
    mode_jupyter: str = "inline",
    host: str = "127.0.0.1",
    port: str = "8050",
) -> None:
    """
    Run the app with the specified parameters.
    
    Parameters
    ----------
    app: dash object,
        the app to run
    in_jupyter: bool, optional
        run the app in jupyter (True) or in standalone mode (False)
    mode_jupyter: str, optional
        'inline', 'jupiter_lab', or 'external'
    host: str, optional
        the host to run the app on
    port: str, optional
        the port to run the app on
    
    Returns
    -------
    None
    """
    if in_jupyter:
        pio.renderers.default = "browser"
        app.run_server(debug=True, host=host, port=port)
    else:
        app.run_server(debug=True, mode=mode_jupyter, host=host, port=port)
