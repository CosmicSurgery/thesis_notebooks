from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import ColumnDataSource, Select
from bokeh.plotting import figure
from bokeh.events import SelectionGeometry
from bokeh.models.widgets import Slider
from bokeh.layouts import column, row

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

# Create a ColumnDataSource to store the scatter plot data
source_A = ColumnDataSource(data=dict(x=[], y=[], category=[]))
source_B = ColumnDataSource(data=dict(x=[], y=[], category=[]))
source_C = ColumnDataSource(data=dict(x=[], y=[], category=[]))
source_D = ColumnDataSource(data=dict(x=[], y=[], category=[]))

# Create four empty figures for the subplots
fig_A = figure(width=300, height=300, title="Subplot A")
fig_B = figure(width=300, height=300, title="Subplot B")
fig_C = figure(width=300, height=300, title="Subplot C")
fig_D = figure(width=300, height=300, title="Subplot D")

# Create four empty scatter renderers for each subplot
scatter_A = fig_A.scatter(x=[], y=[], color='blue', size=10)
scatter_B = fig_B.scatter(x=[], y=[], color='red', size=10)
scatter_C = fig_C.scatter(x=[], y=[], color='green', size=10)
scatter_D = fig_D.scatter(x=[], y=[], color='purple', size=10)

# Create a list of ColumnDataSources and scatter renderers to update the data
data_sources = [source_A, source_B, source_C, source_D]
scatter_renderers = [scatter_A, scatter_B, scatter_C, scatter_D]

# Create a 2x2 grid layout for the four subplots
grid = gridplot([[fig_A, fig_B], [fig_C, fig_D]])

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
    
    # Update the scatter plot data sources for each subplot
    for i, (scatter_renderer, data_source) in enumerate(zip(scatter_renderers, data_sources)):
        indices = (K_sparse[1][K_sparse[2] == i + 1], K_sparse[0][K_sparse[2] == i + 1])
        data_source.data = dict(x=indices[0], y=indices[1], category=np.full_like(indices[0], i + 1))


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

# Combine the four subplots into a single grid layout
subplot_layout = gridplot([[fig_A, fig_B], [fig_C, fig_D]])

# Combine the slider layout and the subplot layout into a row layout
layout = row(slider_layout, subplot_layout)

# Call the update function to initialize the plots with the initial parameter values
update(None, None, None)

# Show the layout
show(layout)
