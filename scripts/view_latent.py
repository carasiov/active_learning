from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import RadioButtons


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "output_test1.npz"
LABELS_PATH = BASE_DIR / "data" / "labels.csv"


data = np.load(DATA_PATH)


class LatentSpaceViewer:
    def __init__(self, data):
        self.path_labels = LABELS_PATH
        self.true_labels = data['labels']
        self.user_labels = pd.read_csv(self.path_labels, header=0).dropna()
        self.scatter_plot_coords = data['tsne']
        self.latent_space = data['latent']
        self.org_data = data['input']
        self.recon_data = data['reconstruted']
        self.pred_classes = data['pred_classes']
        self.pred_certainty = data['pred_certainty']
        self.selected_sample = 0

        fig = plt.figure()
        fig.canvas.mpl_connect('button_press_event', lambda event: self.on_scatter_click(event))
        fig.canvas.mpl_connect('key_press_event', lambda event: self.on_key_press(event))

        self.fig = fig

        gs = matplotlib.gridspec.GridSpec(2, 2)
        self.axes = {}

        ax = fig.add_subplot(gs[:, 0])
        ax.grid()
        ax.set_xlabel("Component #1", fontsize=13)
        ax.set_ylabel("Component #2", fontsize=13)
        self.axes['latent'] = ax
        self.plots = {
            'point': ax.plot([], [], color='black', linestyle='None', marker='x', mew=5, markersize=10, zorder=3, label='selected')[0]
        }
        self.plots['latent_space'] = ax.scatter(self.scatter_plot_coords[:, 0], self.scatter_plot_coords[:, 1], s=3)
        ax.legend(loc='upper left', fontsize=13)

        ax = fig.add_subplot(gs[0, 1])
        ax.set_title('org', fontsize=13)
        plt.imshow(self.org_data[0], cmap='gray')
        self.axes['org'] = ax
        ax = fig.add_subplot(gs[1, 1])
        ax.set_title('recon', fontsize=13)
        plt.imshow(self.recon_data[0], cmap='gray')
        self.axes['recon'] = ax

        rax = plt.axes([0.0, 0.0, 0.12, 0.08])
        self.radio = RadioButtons(rax, ('user labels', 'pred_class', 'true_class', 'certainty'), active=0)
        self.coloring = 'user labels'
        self.radio.on_clicked(self.change_coloring)

        self.axes['latent'].legend(loc='upper left', fontsize=13)
        print(self.coloring)
        self.update_colors()
        self.update()

    def change_coloring(self, label):
        self.coloring = label
        print(self.coloring)
        self.update_colors()

    def update_colors(self):
        if self.coloring == "user labels":
            labels_indexed = self.user_labels.set_index(['Serial'])
            labels_indexed = pd.DataFrame(self.org_data.reshape(-1, 28 * 28))[[]].join(labels_indexed).astype('float').fillna(4.5)
            sc = labels_indexed[('label')].astype(float).values
            sc_normalized = (sc - sc.min()) / (sc.max() - sc.min())
            cols = plt.cm.coolwarm(sc_normalized)
            self.plots['latent_space'].set_color(cols)
            self.update()
            return
        elif self.coloring == "pred_class":
            sc = self.pred_classes
            sc_normalized = (sc - sc.min()) / (sc.max() - sc.min())
            cols = plt.cm.coolwarm(sc_normalized)
            self.plots['latent_space'].set_color(cols)
            self.update()
            return
        elif self.coloring == "true_class":
            sc = self.true_labels
            sc_normalized = (sc - sc.min()) / (sc.max() - sc.min())
            cols = plt.cm.coolwarm(sc_normalized)
            self.plots['latent_space'].set_color(cols)
            self.update()
            return
        elif self.coloring == 'certainty':
            sc = self.pred_certainty
            sc_normalized = (sc - sc.min()) / (sc.max() - sc.min())
            cols = plt.cm.coolwarm(sc_normalized)
            self.plots['latent_space'].set_color(cols)
            self.update()
            return
        else:
            print(f"Unknown coloring: {self.coloring}")
            sc = np.zeros(len(self.org_data))

        lower_percentile = np.percentile(sc, 15)
        upper_percentile = np.percentile(sc, 99)
        sc_clipped = np.clip(sc, lower_percentile, upper_percentile)
        sc_normalized = (sc_clipped - sc_clipped.min()) / (sc_clipped.max() - sc_clipped.min())
        cols = plt.cm.coolwarm(sc_normalized)
        self.plots['latent_space'].set_color(cols)

        self.update()
        return

    def on_scatter_click(self, event):
        """Wählt ein Rohr nur aus, wenn im Scatterplot geklickt wurde."""
        if event.inaxes != self.axes['latent']:
            return

        try:
            print(f"Clicked {event.button} at (x,y)=({event.xdata:.2f},{event.ydata:.2f})")
        except TypeError:
            return

        if not event.button == 1:
            return

        toolbar_mode = plt.get_current_fig_manager().toolbar.mode
        if toolbar_mode != '':
            print(f"Selection not possible while {toolbar_mode} is active!")
            return

        dist = (self.scatter_plot_coords[:, 0] - event.xdata) ** 2 + (self.scatter_plot_coords[:, 1] - event.ydata) ** 2
        self.selected_sample = np.nanargmin(dist)
        print(self.selected_sample)
        self.update_colors()
        self.update()

    def on_key_press(self, event):
        if (event.key == '0') | (event.key == '1') | (event.key == '2') | (event.key == '3') | (event.key == '4') | (event.key == '5') | (event.key == '6') | (event.key == '7') | (event.key == '8') | (event.key == '9'):
            print(str(self.selected_sample) + ' labeled with ' + event.key)
        elif event.key == 'd':
            print(str(self.selected_sample) + ' label deleted')
        else:
            print('Unknown key pressed')
            print('Available keys: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, d')
            return
        self.save_label(event.key)

    def save_label(self, label):
        new_row = pd.DataFrame([{'Serial': self.selected_sample, 'label': label}])

        print(self.selected_sample)
        self.user_labels = pd.concat([self.user_labels, new_row], ignore_index=True)

        self.user_labels.drop_duplicates(subset=['Serial'], keep='last', inplace=True)
        self.user_labels = self.user_labels[self.user_labels['label'] != 'd'].reset_index(drop=True)
        print(self.user_labels)
        self.user_labels.to_csv(self.path_labels, index=False)

    def update(self):
        selected_sample = self.selected_sample
        x = [self.scatter_plot_coords[selected_sample, 0]]
        y = [self.scatter_plot_coords[selected_sample, 1]]
        self.axes['latent'].set_title(f"Selected sample: {selected_sample}")
        self.plots['point'].set_data(x, y)
        self.plots['point'].set_label(f"{selected_sample}")
        self.axes['org'].imshow(self.org_data[selected_sample], cmap='gray')
        self.axes['recon'].imshow(self.recon_data[selected_sample], cmap='gray')
        plt.show()


lsv = LatentSpaceViewer(data)
