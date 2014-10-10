from bokeh.objects import  ServerDataSource, Plot
from bokeh.widgets import HBox, VBox
from bokeh.plot_object import PlotObject
from bokeh.properties import (
    Datetime, HasProps, Dict, Enum, Either, Float, Instance, Int,
    List, String, Color, Include, Bool, Tuple, Any
)
from bokeh.plotting import image

class AjaxDataSource(ServerDataSource):
    url = String()

class TaxiApp(VBox):
    extra_generated_classes = [["TaxiApp", "TaxiApp", "VBox"]]
    extra_scripts = ['/bokehjs/static/app/src/js/ajax_data_source.js']
    extra_js=['window.ajax_data_source.main();']
    pickup_plot = Instance(Plot)
    dropoff_plot = Instance(Plot)
    @classmethod
    def create(cls):
        app = cls()
        data = AjaxDataSource(
            data_url="/bokeh/taxidata/pickup/",
            data=dict(
                x=[1], y=[2], dw=[10], dh=[10], palette=["Reds-9"]
            )
        )
        plot = image(source=data,
                     image="image",
                     x="x",
                     y="y",
                     dw="dw",
                     dh="dh",
                     plot_width=600,
                     plot_height=400,
                     palette='palette',
                     x_range=[1, 11], y_range=[2, 11],
                     tools="pan,wheel_zoom,box_zoom,reset,previewsave",
                     title='pickup'
        )
        app.pickup_plot = plot
        data = AjaxDataSource(
            data_url="/bokeh/taxidata/dropoff/",
            data=dict(
                x=[1], y=[2], dw=[10], dh=[10], palette=["Reds-9"]
            )
        )
        plot = image(source=data,
                     image="image",
                     plot_width=600,
                     plot_height=400,
                     x="x",
                     y="y",
                     dw="dw",
                     dh="dh",
                     palette='palette',
                     x_range=[1, 11], y_range=[2, 11],
                     tools="pan,wheel_zoom,box_zoom,reset,previewsave",
                     title='dropoff'
        )
        app.dropoff_plot = plot
        app.children = [app.pickup_plot, app.dropoff_plot]
        return app
