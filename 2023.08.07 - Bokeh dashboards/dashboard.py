from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import ColumnDataSource, Select
from bokeh.plotting import figure
from bokeh.events import SelectionGeometry
from bokeh.models.widgets import Slider
from bokeh.layouts import column, row

'''
Model parameters
'''
import random
import numpy as np

M = 4 # Number of Spiking motifs
N = 20 # Number of input neurons
D = 31 # temporal depth of receptive field
T = 1000
dt = 1
nrn_fr = 20 # hz
pg_fr = 6 # hz
background_noise_fr = 5 # h

np.random.seed(41)

'''
Setup
'''
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import sys
np.set_printoptions(threshold=sys.maxsize)
disp_figs = True
import colorsys

def create_color_spectrum(num_labels):
    golden_ratio_conjugate = 0.618033988749895
    hues = np.arange(num_labels)
    hues = (hues * golden_ratio_conjugate) % 1.0
    saturations = np.ones(num_labels) * 0.8
    lightness = np.ones(num_labels) * 0.6

    # Convert HSL to RGB and then to hexadecimal
    colors = []
    for h, s, l in zip(hues, saturations, lightness):
        r, g, b = [int(255 * x) for x in colorsys.hls_to_rgb(h, l, s)]
        colors.append(f'#{r:02x}{g:02x}{b:02x}')

    return colors

# Existing colors represented as hexadecimal strings
existing_colors = np.array(['#000000','#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

# Create a palette with 101 colors (11 existing + 90 new)
num_new_colors = 90
new_colors = create_color_spectrum(num_new_colors)
palette = np.concatenate([existing_colors, new_colors])

from bokeh.plotting import figure, show
from bokeh.layouts import gridplot

# Create sliders for the parameters you want to change dynamically
M_slider = Slider(title='Number of Spiking motifs', start=1, end=10, step=1, value=M)
N_slider = Slider(title='Number of input neurons', start=1, end=50, step=1, value=N)
D_slider = Slider(title='Temporal depth of receptive field', start=1, end=100, step=1, value=D)
T_slider = Slider(title='T', start=100, end=2000, step=100, value=T)
dt_slider = Slider(title='dt', start=1, end=10, step=1, value=dt)
nrn_fr_slider = Slider(title='Neuron firing rate (Hz)', start=1, end=50, step=1, value=nrn_fr)
pg_fr_slider = Slider(title='Spiking motif firing rate (Hz)', start=1, end=20, step=1, value=pg_fr)
background_noise_fr_slider = Slider(title='Background noise firing rate (Hz)', start=1, end=20, step=1, value=background_noise_fr)

# Create a ColumnDataSource to store the plot data
source_A = ColumnDataSource(data=dict(x=[], y=[], category=[]))
source_B = ColumnDataSource(data=dict(x=[], y=[], category=[]))
source_C = ColumnDataSource(data=dict(x=[], y=[], category=[]))
source_D = ColumnDataSource(data=dict(x=[], y=[], category=[]))
source_E = ColumnDataSource(data=dict(x=[], y=[], category=[],color=[]))
source_F = ColumnDataSource(data=dict(x=[], y=[], category=[],color=[]))
source_G = ColumnDataSource(data=dict(x=[], y=[],color=[]))

# Create empty figures for the subplots
width = 300
height = 200
fig_A = figure(width=width, height=height, title="Subplot A")
fig_B = figure(width=width, height=height, title="Subplot B")
fig_C = figure(width=width, height=height, title="Subplot C")
fig_D = figure(width=width, height=height, title="Subplot D")

combined_width = fig_A.width + fig_B.width + fig_C.width + fig_D.width

fig_E = figure(width=combined_width, height=height, title="Subplot E")
fig_F = figure(width=combined_width, height=height, title="Subplot F")
fig_G = figure(width=combined_width, height=height, title="Subplot G")

# Create four empty scatter renderers for each subplot
scatter_A = fig_A.scatter(x='x', y='y', color=palette[1], size=10, source=source_A, alpha=0.9)
scatter_B = fig_B.scatter(x='x', y='y', color=palette[2], size=10, source=source_B, alpha=0.9)
scatter_C = fig_C.scatter(x='x', y='y', color=palette[3], size=10, source=source_C, alpha=0.9)
scatter_D = fig_D.scatter(x='x', y='y', color=palette[4], size=10, source=source_D, alpha=0.9)
scatter_E = fig_E.scatter(x='x', y='y', color='color', size=10, source=source_E, alpha=0.9)
scatter_F = fig_F.scatter(x='x', y='y', color='color', size=10, source=source_F, alpha=0.9)

# Create a line renderer for subplot G

# Create a list to store the line renderers for subplot G
line_renderers = []


# Create a list of ColumnDataSources and scatter renderers to update the data
K_data_sources = [source_A, source_B, source_C, source_D]
scatter_renderers = [scatter_A, scatter_B, scatter_C, scatter_D]

other_data_sources = [source_E, source_F]
other_renderers = [scatter_E, scatter_F]

# Define the update function with the data generation and plotting
def update(attr, old, new):
    # Get the slider values
    M = M_slider.value
    N = N_slider.value
    D = D_slider.value
    T = T_slider.value
    dt = dt_slider.value
    nrn_fr = nrn_fr_slider.value
    pg_fr = pg_fr_slider.value
    background_noise_fr = background_noise_fr_slider.value

    # Data Generation
    K_dense = np.random.randint(0, 999, (N, D, M))
    K_dense[K_dense < nrn_fr] = 1
    K_dense[K_dense >= nrn_fr] = 0
    K_sparse = np.where(K_dense)
    K_sparse = (K_sparse[0], K_sparse[1], K_sparse[2] + 1)

    B_dense = np.random.randint(0, 999, (M, T))
    B_dense[B_dense < pg_fr] = 1
    B_dense[B_dense >= pg_fr] = 0
    B_sparse = np.where(B_dense)
    B_sparse = (B_sparse[0] + 1, B_sparse[1])

    A_dense = np.zeros((N, T + D, M + 1))
    A_dense[..., 0] = np.random.randint(0, 999, (N, T + D))
    A_dense[..., 0] = (A_dense[..., 0] < background_noise_fr).astype('int')
    for i in range(len(B_sparse[0])):
        t = B_sparse[1][i]
        b = B_sparse[0][i]
        A_dense[:, t:t + D, b] += K_dense[..., b - 1]

    A_sparse = np.where(A_dense)
    A_dense = np.sum(A_dense, axis=2)
    A_dense[A_dense > 1] = 1
    
    # Take a ground truth pattern from K_dense and convolute it with A_dense to make sure that perfect knowledge can pull out
    # the timings of the pattern

    test = np.zeros((T,M))
    for j in range(M):
        for i in range(T):
            test[i,j] = np.sum(K_dense[...,j]*A_dense[:,i:i+D])
        test[:,j] = test[:,j]/np.max(test[:,j])

    
    # Update the scatter plot data sources for each subplot
    
    for i, (scatter_renderer, data_source) in enumerate(zip(scatter_renderers, K_data_sources)):
        indices = (K_sparse[1][K_sparse[2] == i + 1], K_sparse[0][K_sparse[2] == i + 1])
        data_source.data = dict(x=indices[0], y=indices[1], category=np.full_like(indices[0], i + 1))
    

    indices = (B_sparse[1], B_sparse[0])
    source_E.data = dict(x=indices[0], y=indices[1], category=np.full_like(indices[0], i + 1), color=palette[B_sparse[0]])
    
    indices = (A_sparse[1], A_sparse[0])
    source_F.data = dict(x=indices[0], y=indices[1], category=np.full_like(indices[0], i + 1), color= palette[A_sparse[2]]) 
    
#     source_G.data = dict(x=np.arange(0,(len(test[:, 0]))), y=test[:, 0])

    # Update the line renderers and their data for subplot G
    for i in range(M):
        # Check if the line renderer for the current "y" series already exists
        if i >= len(line_renderers):
            # Create a new line renderer if it doesn't exist
            new_line_renderer = fig_G.line(x=np.arange(0,(len(test[:, i]))), y=test[:, i], line_color=palette[i + 1], line_width=2,alpha=0.3)
            line_renderers.append(new_line_renderer)
        else:
            # If the line renderer exists, update its line color
            line_renderers[i].glyph.line_color = palette[i + 1]

        # Update the data of the line renderer for the current "y" series
        line_renderers[i].data_source.data = dict(x=np.arange(0,(len(test[:, i]))), y=test[:, i])

    # If there are more line renderers than needed, remove the extra ones
    while len(line_renderers) > M:
        line_renderer = line_renderers.pop()
        fig_G.renderers.remove(line_renderer)
    


# Attach the update function to the 'value' property of the sliders
M_slider.on_change('value', update)
N_slider.on_change('value', update)
D_slider.on_change('value', update)
T_slider.on_change('value', update)
dt_slider.on_change('value', update)
nrn_fr_slider.on_change('value', update)
pg_fr_slider.on_change('value', update)
background_noise_fr_slider.on_change('value', update)

# Create a column layout for the sliders
slider_layout = column(M_slider, N_slider, D_slider, T_slider, dt_slider, nrn_fr_slider, pg_fr_slider, background_noise_fr_slider)

# Combine the four subplots A and B into a single row layout
row_ABCD = row(fig_A, fig_B,fig_C, fig_D, sizing_mode="stretch_width")
# Combine the E, F, and G into a single column layout
col_EFG = column(fig_E, fig_F, fig_G, sizing_mode="stretch_width")

# Combine the four subplots into a single grid layout
subplot_layout = gridplot([[row_ABCD], [col_EFG]], sizing_mode="scale_width")

# Combine the slider layout and the subplot layout into a row layout
layout = row(slider_layout, subplot_layout)

# Call the update function to initialize the plots with the initial parameter values
update(None, None, None)

# Add the layout to the current document
curdoc().add_root(layout)
