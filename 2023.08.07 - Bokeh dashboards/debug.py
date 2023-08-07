from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Slider
from bokeh.layouts import column

# Create a ColumnDataSource to store the scatter plot data
source = ColumnDataSource(data=dict(x=[], y=[]))

# Create a figure for the scatter plot
plot = figure()
plot.scatter(x='x', y='y', size=8, color='blue', source=source)

# Create a slider to control the number of data points
num_points_slider = Slider(title='Number of Points', start=10, end=100, step=10, value=50)

# Define the update function to generate data and update the scatter plot
def update(attr, old, new):
    num_points = num_points_slider.value
    x = list(range(num_points))
    y = [i ** 2 for i in x]  # Square the x values for demonstration
    source.data = dict(x=x, y=y)

# Attach the update function to the 'value' property of the slider
num_points_slider.on_change('value', update)

# Create a layout for the slider and the scatter plot
layout = column(num_points_slider, plot)

# Call the update function to initialize the plot with the initial parameter values
update(None, None, None)

# Add the layout to the current document
curdoc().add_root(layout)
