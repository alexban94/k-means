import os
from datetime import datetime
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.font import BOLD, Font
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import matplotlib.pyplot as plt
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
        self.attributes = []

        w, h = 800,600

        # x, y and k to use in the scatter plot/k-means; a widget function will change these
        self.x = 0
        self.y = 1
        self.k = 3

        # PCA and save flag for check button on the Main menu frame.
        self.pca_flag = tk.BooleanVar(master=self.window, value=False)
        self.save_flag = tk.BooleanVar(master=self.window, value=False)

        # Dictionary to contain the frames that represent pages
        self.frames = {'Main': MainMenu(self.window, self, w, h),
                       'Visualization': VisualizationFrame(self.window, self, w, h)}
        self.frames['Main'].grid(row=0, column=0, sticky="nsew")
        self.frames['Visualization'].grid(row=0, column=0, sticky="nsew")

        self.load_data()
        self.load_main()

        # Start program loop.
        tk.mainloop()

    def load_main(self):
        frame = self.frames['Main']
        frame.tkraise()

    def load_data(self):
        # Load dataset only if data path exists and is a .csv file.
        if os.path.exists(self.data_path) and ".csv" in self.data_path:
            df = pd.read_csv(self.data_path, index_col=0)

            # Remove attributes that are strings.
            df = df.iloc[:,
                 [i for i in range(len(df.columns)) if not (df.dtypes.iloc[i] == object or df.dtypes.iloc[i] == str)]]

            # Remove entries that have infinity or nan attributes.
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)

            # Get attribute names
            self.attributes = list(df.columns)

            # Update combo boxes
            frame = self.frames['Main']

            frame.combo_x['values'] = self.attributes
            frame.combo_y['values'] = self.attributes

            frame.combo_x.current(self.x)
            frame.combo_y.current(self.y)

            # Normalize dataset and convert to np array.
            #scaler = Normalizer(norm='l2')
            #self.data = scaler.fit_transform(df)
            self.data = df
        else:
            showinfo("Warning", "Dataset path does not exist.")

    def load_visualization(self):
        # Update x and y variables based on combobox.
        self.x = self.frames['Main'].combo_x.current()
        self.y = self.frames['Main'].combo_y.current()
        # Update k based on spinbox
        self.k = self.frames['Main'].spin_k.get()

        # Copy data in case of PCA calculations.
        if isinstance(self.data, np.ndarray):
            data = self.data.copy().to_numpy()
        else:
            data = self.data

        # Check everything is correct and ready to use.
        if self.x == self.y:
            showinfo("Warning", "The x and y attributes for the plot are the same.")
            return
        elif self.data is []:
            showinfo("Warning", "Please select a dataset.")
            return
        # Record axis labels
        axis_labels = [self.frames['Main'].combo_x.get(), self.frames['Main'].combo_y.get()]

        # Check if PCA flag has been checked.
        if self.pca_flag.get():
            # Override selected attributes to use only x=0 and y=1, to display only the first and second
            # principal components PC1 and PC2 (the most relevant).
            self.x = 0
            self.y = 1
            axis_labels = ['PC1', 'PC2']

            # Normalize data.
            scaler = Normalizer(norm='l2')
            data = scaler.fit_transform(data)

            # Perform PCA on data.
            pca = PCA(n_components=2)
            data = pca.fit_transform(data)

        # Raise visualization frame
        frame = self.frames['Visualization']
        frame.tkraise()

        # Pass data to visualization frame for k-means algorithm and animation.
        frame.visualize_k_means(data[:, [self.x, self.y]], axis_labels, save=self.save_flag.get())


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
                 height=2).pack(pady=(50,0))


        ## PCA checkbutton.
        self.check_pca = tk.Checkbutton(master=self,
                                        variable=app.pca_flag,
                                        onvalue=True,
                                        offvalue=False,
                                        text="Enable Principal Component Analysis (PCA)?",
                                        font=Font(family="Arial", size=10),
                                        background='black', foreground='white',
                                        activeforeground='white', selectcolor="black"
                                   )
        self.check_pca.pack()#pady=(25,0))

        ## Save animation checkbutton.
        self.check_save = tk.Checkbutton(master=self,
                                        variable=app.save_flag,
                                        onvalue=True,
                                        offvalue=False,
                                        text="Save animation? (requires FFmpeg)",
                                        font=Font(family="Arial", size=10),
                                        background='black', foreground='white',
                                        activeforeground='white', selectcolor="black"
                                        )
        self.check_save.pack()

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
        frame_files.pack(side=tk.BOTTOM, padx=50, pady=(0,100))
        self.entry.pack(side=tk.LEFT)
        browse.pack(side=tk.LEFT)

        # Spinbox widget for selecting value of k to use.
        frame_spin = tk.Frame(master=self)
        label = tk.Label(master=frame_spin, text="Select k:", background='black', foreground='white')
        self.spin_k = ttk.Spinbox(master=frame_spin, from_=2, to=20, increment=1, wrap=True,
                                        foreground='black', background="black")
        self.spin_k.set(3)

        # Pack spinbox widgets.
        label.pack(side=tk.LEFT)
        self.spin_k.pack(side=tk.RIGHT)
        frame_spin.pack(pady=(40,0))

        # Combo boxes for selecting the 2 attributes to use.
        self.combo_x = self.create_combo_box([],'x')
        self.combo_y = self.create_combo_box([],'y')

        # Start button widget
        btn_load = tk.Button(master=self,
                             command=app.load_visualization,
                             relief=tk.RAISED,
                             text="Start",
                             foreground="white",
                             background="grey",
                             width=25,
                             height=2).pack(pady=(80, 0))



    def create_combo_box(self, attributes, axis):
        # Frame to join the label and combobox together.
        frame = tk.Frame(master=self)
        combo = ttk.Combobox(master=frame, values = attributes)
        combo.set("Select attribute")
        label = tk.Label(master=frame, text="Select %s attribute:" % axis, background='black',foreground='white')

        label.pack(side=tk.LEFT)
        combo.pack(side=tk.RIGHT)
        frame.pack()

        return combo



    def browse_files(self):

            filename = fd.askopenfilename(
                title='Choose a dataset',
                initialdir= os.getcwd(),
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
        ##TODO: what is this
        #self.grid_propagate(0)

        self.app = app
        self.img = tk.PhotoImage(file=r'cancel.png').subsample(13,13) # keep reference to avoid garbage collection
        self.btn_exit = tk.Button(master=self,
                                  text='exit',
                                  command=self.return_to_main,
                                  image=self.img,
                                  width='25',
                                  height='25',
                                  bd=0 # border pixels
                                  )

        # Create figure and subplot.
        self.fig = plt.Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Initialize class variables to be used in the FuncAnimation.
        self.scatter_data = None
        self.scatter_mu = None
        self.canvas = None # Canvas is not created here as it is made/destroyed on start/exit.
        self.anim = None

        ## Aesthetic options
        # Black background
        self.ax.set_facecolor('xkcd:black')
        self.fig.patch.set_facecolor('xkcd:black')
        # White axes
        for spine in self.ax.spines:
            self.ax.spines[spine].set_color('white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.tick_params(colors='white')
        self.ax.set_title("", color="white")
        # Colour map to use for the different clusters.
        self.cmap = plt.get_cmap('tab20') # other options: 'tab20', 'tab20b', 'tab20c', 'rainbow'


    def return_to_main(self):
        # Load main menu
        self.app.load_main()
        # Destroy current animation
        plt.close()
        self.anim.event_source.stop()
        self.canvas.get_tk_widget().destroy()
        del self.anim


    def visualize_k_means(self, np_data, attributes, save=False):
        # Set axis labels.
        self.ax.set_xlabel(attributes[0])
        self.ax.set_ylabel(attributes[1])

        # Embed matplotlib graph animation
        self.canvas = FigureCanvasTkAgg(figure=self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        # Place exit button on top of canvas
        self.btn_exit.place(in_=self.canvas.get_tk_widget(), relx=0,rely=0, anchor="nw")
        self.btn_exit.tkraise()

        # Perform k-means on the given dataset.
        mu_viz, r_viz, iters = k_means(np_data, k=int(self.app.k), max_iter=1000)

        # Create the scatter plots. By default marker size s=36.
        self.scatter_data = self.ax.scatter([],[], marker='x', s=18) # For the data
        self.scatter_mu = self.ax.scatter([],[], marker='D', edgecolors='white', facecolors='lawngreen', s=25) # For the moving cluster centres.

        # FuncAnimation
        self.anim = animation.FuncAnimation(self.fig,
                                            func=self.animate,
                                            frames=iters,
                                            save_count=iters,
                                            fargs=(mu_viz, r_viz, np_data, iters),
                                            interval=100,
                                            repeat_delay=250,
                                            blit=False)
        self.canvas.draw()

        # Save animation if flagged to do so.
        if save:
            date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
            writer = animation.FFMpegWriter(fps=30)
            self.anim.save('%i-means-x_%s_y_%s-%s.mp4' % (int(self.app.k), attributes[0], attributes[1], date), writer=writer)


    # Animation function called at each frame.
    def animate(self, i, *args):
        # print("Frame: %i" % i)
        # Unpack required data: args = (mu_viz, r_viz, np_data).
        mu_k = args[0][i]  # Current cluster centres
        r = args[1][:,i]  # Current cluster assignments
        data = args[2]  # Data to plot
        iters = args[3] # Convergence iteration

        # Plot data and set colours from using the cluster labels with the chosen colour map.
        self.scatter_data.set_offsets(data)
        r_norm = r/max(r) # Equivalent to using norm = matplotlib.colors.Normalize(vmin=0, vmax=k-1)
        self.scatter_data.set_color(self.cmap(r_norm))
        # Plot cluster centres
        self.scatter_mu.set_offsets(mu_k[:,0:2])

        # Set x/y limits for an appropriate scale.
        self.ax.set_ylim(min(data[:,1]), max(data[:,1]))
        self.ax.set_xlim(min(data[:,0]), max(data[:,0]))

        # Update title
        self.ax.set_title("Iteration: %i of %i" % (i + 1, iters))

        return self.scatter_data,



if __name__ == "__main__":

    # Run Tkinter event loop; listens for events (clicks/keypresses), any code after this will not run
    # until window is closed.
    k_means_app = Application()

