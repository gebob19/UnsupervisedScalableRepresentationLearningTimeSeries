#%%
%load_ext autoreload
%autoreload 2

import sys 
import pathlib 
sys.path.append(str(pathlib.Path.home()/'Documents/stocks/penguin/'))

import torch
import numpy as np 

# 30-day split data 
og_split_data = torch.from_numpy(np.load('data/stock_data_30daysplit_cont.npy'))
split_data = torch.from_numpy(np.load('data/stock_features_30daysplit_cont.npy'))

#%%
from bokeh.plotting import figure, show
from bokeh.io import output_notebook

from stock.reinforcement.USTS import scikit_wrappers 

def get_encoder(load_path):
    encoder_yearly = scikit_wrappers.CausalCNNEncoderClassifier()
    encoder_yearly.set_params(**hyperparameters)
    encoder_yearly.load_encoder(load_path)

def plot(data, title):
    obj = {
        'open': data[:, 0],
        'close': data[:, 1],
        'high': data[:, 2],
        'low': data[:, 3],
    }
    # bokeh plotting
    p = figure(title=title,plot_height=100, plot_width=100)
    bar_width = 1 # 1-day 
    
    h, l, o, c = obj['high'].cpu().numpy(), obj['low'].cpu().numpy(), obj['open'].cpu().numpy(), obj['close'].cpu().numpy()
    inc, dec = c > o, c < o
    dt = np.arange(len(h))

    # plot candles 
    p.segment(dt, h, dt, l, color="black")
    p.vbar(dt[inc], bar_width, o[inc], c[inc], fill_color="green", line_color="black")
    p.vbar(dt[dec], bar_width, o[dec], c[dec], fill_color="red", line_color="black")

    p.outline_line_color = None 
    p.axis.visible = False
    p.grid.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    return p

def close_plot(data, title):
    # bokeh plotting
    p = figure(title=title, plot_height=100, plot_width=100)
    data = data.squeeze()
    p.line(np.arange(len(data)), data)

    p.outline_line_color = None 
    p.axis.visible = False
    p.grid.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    return p 

#%%
def plot_closest_n(sdata, n):
    from data.db import Router, TickerStream
    from tqdm.notebook import tqdm
    r = Router('daily')
    symbols = r.available()

    data = torch.load('last30_data.pt')
    data = data.permute(0, 2, 1)

    import scikit_wrappers

    encoder_yearly = get_encoder('saved_models/stockmodel_continued2')
    tmp = data[:, 1, :].unsqueeze(1).numpy()
    features = encoder_yearly.encode_window(tmp, 30)

    distance = ((sdata - torch.from_numpy(features)) ** 2).squeeze().mean(-1)

    min_d_idxs = distance.numpy().argsort()[:n]

    from scan.run_scan import plot_ticker
    min_symbols = [symbols[i] for i in min_d_idxs]
    p = plot_ticker(' '.join(min_symbols), 8080)
    return p

#%%
import bokeh 
from bokeh.plotting import show
plots = []
idx = torch.randperm(og_split_data.size(0))[:50]
for data, i in zip(og_split_data[idx], idx):
    plots.append(close_plot(data.numpy(), str(i.numpy())))
show(bokeh.layouts.grid(plots, ncols=6))

#%%
idx = 2756
sdata = split_data[idx]
ogdata = og_split_data[idx]
plots = [close_plot(ogdata.numpy(), str(idx))]
show(bokeh.layouts.grid(plots, ncols=6))

#%%
plot_closest_n(sdata, 10)

# #%%
# r = Router('daily')
# symbols = r.available()
# last30_data = []
# for t in tqdm(symbols):
#     data = TickerStream(t, r).torch_most_recent(30) 
#     if len(data) == 30:
#         last30_data.append(data)

# data = torch.stack(last30_data).double()

# %%
