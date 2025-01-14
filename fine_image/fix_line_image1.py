#!/usr/bin/env python3

import numpy as np
import sys
from PIL import Image, ImageDraw

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

linetype = 'line'
if len(sys.argv) < 2:
    print(sys.argv[0],'target.png')
    exit(1)

dpi = 72
target_file = sys.argv[1]
if len(sys.argv) > 2:
    for arg in sys.argv[2:]:
        if arg.startswith('dpi'):
            dpi = int(arg[3:])
            print('dpi:', dpi)
        else:
            linetype = arg

class Application(tk.Frame):
    def __init__(self, root, target_file):
        super().__init__(root)
        self.target_file = target_file
        root.title(target_file)
        im0 = Image.open(target_file).convert('RGB')
        self.im0 = im0.resize((im0.width // 4, im0.height // 4), resample=Image.Resampling.BILINEAR)

        if linetype == 'line':
            linesfile = target_file + '.lines.png'
        elif linetype == 'seps':
            linesfile = target_file + '.seps.png'
        self.linesfile = linesfile
        self.lines_all = Image.open(linesfile)
        self.lines_draw = ImageDraw.Draw(self.lines_all)
        self.v_line = False
        self.h_line = False

        self.gen_mpl_graph(root)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def save(self):
        self.lines_all.save(self.linesfile)

    def gen_mpl_graph(self, root):
        self.cid = None
        self.newpoints = []
        frame1 = tk.Frame(root)
        frame2 = tk.Frame(root)

        def onclick1(event):
            x = event.xdata
            y = event.ydata
            if x is None or y is None:
                return
            self.newpoints.append((x,y))
            self.ax_im.plot(x, y, 'r.')

            if len(self.newpoints) >= 2:
                self.fig.canvas.mpl_disconnect(self.cid)
                self.cid = None
                self.btn0.configure(text='new line')
                self.btn2.configure(text='new vline')
                self.btn3.configure(text='new hline')

                if self.h_line:
                    self.lines_draw.line((self.newpoints[0][0],self.newpoints[0][1],self.newpoints[1][0],self.newpoints[0][1]), fill=255, width=3)
                elif self.v_line:
                    self.lines_draw.line((self.newpoints[0][0],self.newpoints[0][1],self.newpoints[0][0],self.newpoints[1][1]), fill=255, width=3)
                else:
                    self.lines_draw.line(self.newpoints, fill=255, width=3)
                self.plot_image()
            else:
                self.fig.canvas.draw_idle()

        def onclick2(event):
            x = event.xdata
            y = event.ydata
            if x is None or y is None:
                return
            self.newpoints.append((x,y))
            self.ax_im.plot(x, y, 'y.')

            if len(self.newpoints) >= 2:
                self.fig.canvas.mpl_disconnect(self.cid)
                self.cid = None
                self.btn1.configure(text='remove area')

                self.lines_draw.rectangle(self.newpoints, fill=0)
                self.plot_image()
            else:
                self.fig.canvas.draw_idle()

        def btn_click0():
            if self.cid is None:
                self.newpoints = []
                self.v_line = False
                self.h_line = False
                self.cid = self.fig.canvas.mpl_connect('button_press_event', onclick1)
                self.btn0.configure(text='<new line>')
            else:
                self.fig.canvas.mpl_disconnect(self.cid)
                self.v_line = False
                self.h_line = False
                self.cid = None
                self.btn0.configure(text='new line')

        def btn_click1():
            if self.cid is None:
                self.newpoints = []
                self.cid = self.fig.canvas.mpl_connect('button_press_event', onclick2)
                self.btn1.configure(text='<remove area>')
            else:
                self.fig.canvas.mpl_disconnect(self.cid)
                self.cid = None
                self.btn1.configure(text='remove area')

        def btn_click2():
            if self.cid is None:
                self.newpoints = []
                self.v_line = True
                self.h_line = False
                self.cid = self.fig.canvas.mpl_connect('button_press_event', onclick1)
                self.btn2.configure(text='<new vline>')
            else:
                self.fig.canvas.mpl_disconnect(self.cid)
                self.v_line = False
                self.h_line = False
                self.cid = None
                self.btn2.configure(text='new vline')

        def btn_click3():
            if self.cid is None:
                self.newpoints = []
                self.v_line = False
                self.h_line = True
                self.cid = self.fig.canvas.mpl_connect('button_press_event', onclick1)
                self.btn3.configure(text='<new hline>')
            else:
                self.fig.canvas.mpl_disconnect(self.cid)
                self.v_line = False
                self.h_line = False
                self.cid = None
                self.btn3.configure(text='new hline')

        self.btn0 = tk.Button(frame2, text='new line', command=btn_click0)
        self.btn0.pack(side=tk.LEFT)

        self.btn1 = tk.Button(frame2, text='remove area', command=btn_click1)
        self.btn1.pack(side=tk.LEFT)

        self.btn2 = tk.Button(frame2, text='new vline', command=btn_click2)
        self.btn2.pack(side=tk.LEFT)

        self.btn3 = tk.Button(frame2, text='new hline', command=btn_click3)
        self.btn3.pack(side=tk.LEFT)

        frame2.pack(side=tk.BOTTOM, fill=tk.X)
        frame1.pack(expand=True, fill=tk.BOTH)
        self.canvas = tk.Canvas(frame1)
        frame=tk.Frame(self.canvas)
        
        self.vbar = tk.Scrollbar(self.canvas, orient=tk.VERTICAL, command=self.canvas.yview)
        self.hbar = tk.Scrollbar(self.canvas, orient=tk.HORIZONTAL, command=self.canvas.xview)

        self.canvas.create_window((0, 0), window=frame, anchor="nw")
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.canvas.configure(xscrollincrement='1p',yscrollincrement='1p')
        frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        root.bind('<MouseWheel>', lambda e: self.canvas.yview_scroll(-e.delta, 'units'))
        root.bind('<Shift-MouseWheel>', lambda e: self.canvas.xview_scroll(-e.delta, 'units'))
        # root.bind("<ButtonPress-1>", self.move_start)
        # root.bind("<B1-Motion>", self.move_move)

        self.fig = plt.figure(figsize=(self.im0.width/dpi, self.im0.height/dpi))
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax_im = self.fig.add_subplot(111)
        self.im1 = FigureCanvasTkAgg(self.fig, frame)

        self.plot_image()

        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        self.im1.get_tk_widget().pack(expand=1)

    # def move_start(self, event):
    #     self.canvas.scan_mark(event.x, event.y)

    # def move_move(self, event):
    #     self.canvas.scan_dragto(event.x, event.y, gain=1)

    def plot_image(self):
        self.ax_im.cla()
        self.ax_im.imshow(self.im0)
        self.ax_im.imshow(self.lines_all, cmap='gray', alpha=0.5)
        self.im1.draw_idle()

root = tk.Tk()
root.geometry('1400x800')
app = Application(root, target_file)
app.mainloop()

app.save()
