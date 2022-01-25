import os
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.font import BOLD, Font
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
#from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from k_means import k_means
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.decomposition import PCA

# Class for the overall application.
class Application(tk.Tk):
    def __init__(self):
        self.window = tk.Tk()
        self.window.wm_title("K-Means visualization tool")
        self.window.geometry("800x600+560+210")

        # Path to default dataset - this will change based on the filedialog input.
        self.data_path = os.getcwd() + "\\data\\abalone.csv"
        self.data = []
        self.load_data()
        w, h = 800,600

        # x, y and k to use in the scatter plot/k-means; a widget function will change these
        self.x = 0
        self.y = 1
        self.k = 3

        # Dictionary to contain the frames that represent pages
        self.frames = {'Main': MainMenu(self.window, self, w, h),
                       'Visualization': VisualizationFrame(self.window, self, w, h)}
        self.frames['Main'].grid(row=0, column=0, sticky="nsew")
        self.frames['Visualization'].grid(row=0, column=0, sticky="nsew")

        self.load_main()

        # Start program loop.
        tk.mainloop()

    def load_main(self):
        frame = self.frames['Main']
        frame.tkraise()

    def load_data(self):
        # Load dataset.
        df = pd.read_csv(self.data_path, index_col=0)

        # Remove entries that have infinity or nan attributes.
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna()

        # Get attribute names
        attributes = list(df.columns)

        # Normalize dataset and convert to np array.
        scaler = Normalizer(norm='l2')
        self.data = scaler.fit_transform(df)

    def load_visualization(self):
        # Raise visualization frame
        frame = self.frames['Visualization']
        frame.tkraise()

        # Pass data to visualization frame for k-means algorithm and animation.
        frame.visualize_k_means(self.data)


# Class for the main menu; extends the tk.Frame class
class MainMenu(tk.Frame):
    def __init__(self, window, app, w, h):
        # Call super constructor.
        super().__init__(master=window,
                         background="black",
                         width=w,
                         height=h)
        self.pack_propagate(0)

        # Main application for control
        self.app = app

        # Title label widget
        tk.Label(master=self,
                 text="K-Means Visualization",
                 font=Font(family="Arial", size=25, weight=BOLD),
                 foreground="white",
                 background="black",
                 width=50,
                 height=2).pack(side=tk.TOP)

        # Start button widget
        btn_load = tk.Button(master=self,
                             command=app.load_visualization,
                             relief=tk.RAISED,
                             text="Start",
                             foreground="white",
                             background="grey",
                             width=25,
                             height=2).pack(side=tk.BOTTOM, pady=50)

        ## PCA Frame for the label and checkbutton.
        self.pca_flag = tk.BooleanVar(master=window, value=False)
        self.check_pca = tk.Checkbutton(master=self,
                                   variable=self.pca_flag,
                                   onvalue=True,
                                   offvalue=False,
                                   text="Enable Principal Component Analysis (PCA)?",
                                   font=Font(family="Arial", size=10)
                                   )
        self.check_pca.pack()

        ## Frame for file browser
        frame_files = tk.Frame(master=self)

        # Text entry widget for the filename.
        self.entry = tk.Entry(master=frame_files,
                         width=65
                         )
        self.entry.insert(0, app.data_path)

        # Browse file button
        browse = tk.Button(master=frame_files,
                          command=self.browse_files,
                          relief=tk.RAISED,
                          text="Browse",
                          foreground="white",
                          background="black",
                          width=10,
                          height=1)
        frame_files.pack(side=tk.LEFT, padx=50)
        self.entry.pack(side=tk.LEFT)
        browse.pack(side=tk.LEFT)




    def browse_files(self):

            filename = fd.askopenfilename(
                title='Choose a dataset',
                initialdir='/',
                filetypes=[('.csv files', '*.csv')]
            )
            if filename != "":
                self.entry.delete(0, tk.END)
                self.entry.insert(0, filename)
                self.app.data_path = filename
                self.app.load_data()

# Class for the k-means visualization, extends tkinter Frame.
class VisualizationFrame(tk.Frame):
    def __init__(self, window, app, w, h):
        super().__init__(master=window,
                         background="black",
                         width=w,
                         height=h
                         )
        self.grid_propagate(0)

        self.app = app
        btn_exit = tk.Button(master=self,
                             command=app.load_main,
                             relief=tk.RAISED,
                             text="Exit",
                             foreground="white",
                             background="black",
                             width=25,
                             height=2).pack()

        # Conduct K-Means in its entirety and plot graph animation afterwards.

        # Create figure and subplot.
        self.fig = plt.Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.scatter = None

        # Aesthetic options
        self.ax.set_axis_off()
        #self.ax.set_facecolor('xkcd:black')
        self.fig.patch.set_facecolor('xkcd:black')



        # Define the colours for each k.
        colour1 = (0.69411766529083252, 0.3490196168422699, 0.15686275064945221, 1.0)
        colour2 = (0.65098041296005249, 0.80784314870834351, 0.89019608497619629, 1.0)
        colour3 = (0.45098041296005249, 0.50784314870834351, 0.69019608497619629, 1.0)
        self.colour_map = np.array([colour1, colour2, colour3])


        # Embed matplotlib graph
        self.canvas = FigureCanvasTkAgg(figure=self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)


    def visualize_k_means(self, np_data):
        # Perform k-means on the given dataset.
        mu_viz, r_viz, iters = k_means(np_data, k=self.app.k, max_iter=100)

        # Create the scatter plot
        self.scatter = self.ax.scatter([],[])
        #
        anim = animation.FuncAnimation(self.fig,
                                       func=self.animate,
                                       frames=range(iters),
                                       fargs=(mu_viz, r_viz, np_data),
                                       interval=3000,
                                       repeat_delay=1000,
                                       blit=False)
        self.canvas.draw()
        print("here")

    # Animation function called at each frame.
    def animate(self, i, *args):
        print("Frame: %i" % i)
        # Unpack required data: args = (mu_viz, r_viz, np_data).
        mu_k = args[0][i]  # Current cluster centres
        r = args[1][:,i]  # Current cluster assignments
        data = args[2]  # Data to plot


        # Add the cluster assignments to the data as a 'label'.
        #plot_data = np.hstack([np.copy(data), r.reshape(-1, 1)])  # Convert r from 1d to 2d ndarray
        mu_k = np.hstack([mu_k, [[0], [1], [2]]])

        # Plot data -- update here.
        self.scatter.set_offsets(data[:,[self.app.x, self.app.y]])
        self.scatter.set(color=self.colour_map[r])#, color=plot_data[:,-1])

        # Set x/y limits for an appropriate scale.
        self.ax.set_ylim(min(data[:,self.app.y]), max(data[:,self.app.y]))
        self.ax.set_xlim(min(data[:,self.app.x]), max(data[:,self.app.x]))

        return self.scatter,













if __name__ == "__main__":

    # Run Tkinter event loop; listens for events (clicks/keypresses), any code after this will not run
    # until window is closed.
    k_means_app = Application()

