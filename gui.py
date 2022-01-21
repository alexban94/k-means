import os
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.font import BOLD, Font
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

# Class for the overall application.
class Application(tk.Tk):
    def __init__(self):
        self.window = tk.Tk()
        self.window.wm_title("K-Means visualization tool")
        self.window.geometry("800x600+560+210")

        # Path to default dataset - this will change based on the filedialog input.
        self.data_path = os.getcwd() + "\\vgsales.csv"
        w, h = 800,600

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

    def load_visualization(self):
        print(self.data_path)
        print(self.frames['Main'].pca_flag.get())
        frame = self.frames['Visualization']
        frame.tkraise()

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







if __name__ == "__main__":

    # Run Tkinter event loop; listens for events (clicks/keypresses), any code after this will not run
    # until window is closed.
    k_means_app = Application()

