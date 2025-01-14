#!/usr/bin/env python3

import numpy as np
import sys
from PIL import Image
import json

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

if len(sys.argv) < 2:
    print(sys.argv[0],'target.png')
    exit(1)

fprop = FontProperties(fname='data/jpfont/NotoSerifJP-Regular.otf')

dpi = 72
target_file = sys.argv[1]
if len(sys.argv) > 2:
    for arg in sys.argv[2:]:
        if arg == 'kr':
            fprop = FontProperties(fname='data/krfont/NotoSerifKR-Regular.otf')
            print('kr font')
        elif arg.startswith('dpi'):
            dpi = int(arg[3:])
            print('dpi:', dpi)

class SubWindow(tk.Frame):
    def __init__(self, root, im0, dict, x, y, refresh, closing):
        super().__init__(root)

        root.title(f"x={x} y={y}")

        self.cid = None
        self.newpoints = []

        frame=tk.Frame(root)
        fig = plt.figure(figsize=(5,5), dpi=100)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax_im = fig.add_subplot(111)
        im = FigureCanvasTkAgg(fig, frame)

        def on_closing():
            refresh()
            closing()
            fig.clear()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)

        for i, pos in enumerate(dict['textbox']):
            cx = pos['cx']
            cy = pos['cy']
            w = pos['w']
            h = pos['h']
            self.current_i = i

            if cx - w / 2 < x < cx + w / 2 and cy - h / 2 < y < cy + h / 2:
                self.ruby = pos['p_code1'] > 0.5
                self.ruby_base = pos['p_code2'] > 0.5
                self.emphasis = pos['p_code4'] > 0.5
                self.space = pos['p_code8'] > 0.5
                text = pos['text']

                im1 = im0.copy().crop((cx - w, cy - h, cx + w, cy + h))
                self.ax_im.imshow(im1)

                points = [
                    [w / 2, h / 2],
                    [3 * w / 2, h / 2],
                    [3 * w / 2, 3 * h / 2],
                    [w / 2, 3 * h / 2],
                    [w / 2, h / 2],
                ]
                points = np.array(points)
                self.ax_im.plot(points[:,0], points[:,1],color='blue')
                break
        else:
            cx = x
            cy = y
            w = 150
            h = 150
            self.current_i = len(dict['textbox'])
            dict['textbox'].append({
                'cx': float(cx),
                'cy': float(cy),
                'w': float(w),
                'h': float(h),
                'text': None,
                'p_loc': float(0),
                'p_chr': float(0),
                'p_code1': float(0),
                'p_code2': float(0),
                'p_code4': float(0),
                'p_code8': float(0),
            })

            self.ruby = False
            self.ruby_base = False
            self.emphasis = False
            self.space = False
            text = None

            im1 = im0.copy().crop((cx - w, cy - h, cx + w, cy + h))
            self.ax_im.imshow(im1)

        frame2=tk.Frame(root)
        frame2.pack(side=tk.RIGHT)

        def onclick2(event):
            x = event.xdata
            y = event.ydata
            if x is None or y is None:
                return
            self.newpoints.append((x,y))
            self.ax_im.plot(x, y, 'r.')

            if len(self.newpoints) >= 2:
                fig.canvas.mpl_disconnect(self.cid)
                self.cid = None
                self.btn0.configure(text='fix box')
            
                cx = dict['textbox'][self.current_i]['cx']
                cy = dict['textbox'][self.current_i]['cy']
                w = dict['textbox'][self.current_i]['w']
                h = dict['textbox'][self.current_i]['h']
                offsetx = cx - w
                offsety = cy - h
                cx = (self.newpoints[0][0] + self.newpoints[1][0]) / 2 + offsetx
                cy = (self.newpoints[0][1] + self.newpoints[1][1]) / 2 + offsety
                w = abs(self.newpoints[0][0] - self.newpoints[1][0])
                h = abs(self.newpoints[0][1] - self.newpoints[1][1])

                dict['textbox'][self.current_i]['cx'] = float(cx)
                dict['textbox'][self.current_i]['cy'] = float(cy)
                dict['textbox'][self.current_i]['w'] = float(w)
                dict['textbox'][self.current_i]['h'] = float(h)

                self.ax_im.cla()
                im2 = im0.copy().crop((cx - w, cy - h, cx + w, cy + h))
                self.ax_im.imshow(im2)

                points = [
                    [w / 2, h / 2],
                    [3 * w / 2, h / 2],
                    [3 * w / 2, 3 * h / 2],
                    [w / 2, 3 * h / 2],
                    [w / 2, h / 2],
                ]
                points = np.array(points)
                self.ax_im.plot(points[:,0], points[:,1],color='blue')

            fig.canvas.draw_idle()
    
        def btn_click0():
            if self.cid is None:
                self.newpoints = []
                self.cid = fig.canvas.mpl_connect('button_press_event', onclick2)
                self.btn0.configure(text='<fix box>')
            else:
                fig.canvas.mpl_disconnect(self.cid)
                self.cid = None
                self.btn0.configure(text='fix box')

        def btn_click1():
            self.ruby = not self.ruby
            self.btn1.configure(text=f'ruby {self.ruby}')
            dict['textbox'][self.current_i]['p_code1'] = 1.0 if self.ruby else 0.0

        def btn_click2():
            self.ruby_base = not self.ruby_base
            self.btn2.configure(text=f'rubybase {self.ruby_base}')
            dict['textbox'][self.current_i]['p_code2'] = 1.0 if self.ruby_base else 0.0

        def btn_click3():
            self.emphasis = not self.emphasis
            self.btn3.configure(text=f'emphasis {self.emphasis}')
            dict['textbox'][self.current_i]['p_code4'] = 1.0 if self.emphasis else 0.0

        def btn_click4():
            self.space = not self.space
            self.btn4.configure(text=f'space {self.space}')
            dict['textbox'][self.current_i]['p_code8'] = 1.0 if self.space else 0.0

        self.btn0 = tk.Button(frame2, text='fix box', command=btn_click0)
        self.btn0.pack()

        self.btn1 = tk.Button(frame2, text=f'ruby {self.ruby}', command=btn_click1)
        self.btn1.pack()

        self.btn2 = tk.Button(frame2, text=f'rubybase {self.ruby_base}', command=btn_click2)
        self.btn2.pack()

        self.btn3 = tk.Button(frame2, text=f'emphasis {self.emphasis}', command=btn_click3)
        self.btn3.pack()

        self.btn4 = tk.Button(frame2, text=f'space {self.space}', command=btn_click4)
        self.btn4.pack()

        def enter_key(event):
            t = entry.get()
            if t == '':
                t = None
            dict['textbox'][self.current_i]['text'] = t

        entry = tk.Entry(frame2, width=10, font=('TkDefaultFont', 32))
        entry.pack()
        entry.bind("<Return>", enter_key)

        def btn_click5():
            del dict['textbox'][self.current_i]
            refresh()
            closing()
            fig.clear()
            root.destroy()

        button = tk.Button(frame2, text='remove', command=btn_click5)
        button.pack(padx=10, pady=10)

        frame.pack(expand=True, fill=tk.BOTH)
        im.get_tk_widget().pack(expand=1)

        if text is not None:
            entry.insert(tk.END, text)

class Application(tk.Frame):
    def __init__(self, root, target_file):
        super().__init__(root)
        self.target_file = target_file
        root.title(target_file)
        self.im0 = Image.open(target_file).convert('RGB')
        self.sub = None

        with open(target_file+'.json', 'r', encoding='utf-8') as file:
            self.dict = json.load(file)

        self.gen_mpl_graph(root)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def save(self):
        with open(self.target_file+'.json', 'w', encoding='utf-8') as file:
            json.dump(self.dict, file, indent=2, ensure_ascii=False)

    def gen_mpl_graph(self, root):
        self.canvas = tk.Canvas(root)
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
        root.bind('<Up>', lambda e: self.canvas.yview_scroll(-1, 'pages'))
        root.bind('<Down>', lambda e: self.canvas.yview_scroll(1, 'pages'))
        root.bind('<Left>', lambda e: self.canvas.xview_scroll(-1, 'pages'))
        root.bind('<Right>', lambda e: self.canvas.xview_scroll(1, 'pages'))

        fig = plt.figure(figsize=(self.im0.width/dpi, self.im0.height/dpi))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax_im = fig.add_subplot(111)
        self.im1 = FigureCanvasTkAgg(fig, frame)

        self.plot_image()

        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        self.im1.get_tk_widget().pack(expand=1)

        def plotClick(event):
            if not event.dblclick:
                return
            x = event.xdata
            y = event.ydata
            if x is None or y is None:
                return
            if self.sub is not None:
                return

            def onclose():
                self.sub = None
            
            window2 = tk.Toplevel(root)
            self.sub = SubWindow(window2, self.im0, self.dict, x, y, refresh=self.plot_image, closing=onclose)

        fig.canvas.mpl_connect('button_press_event', plotClick)

    # def move_start(self, event):
    #     self.canvas.scan_mark(event.x, event.y)

    # def move_move(self, event):
    #     self.canvas.scan_dragto(event.x, event.y, gain=1)

    def plot_image(self):
        self.ax_im.cla()
        self.ax_im.imshow(self.im0)

        for pos in self.dict['textbox']:
            cx = pos['cx']
            cy = pos['cy']
            w = pos['w']
            h = pos['h']
            text = pos['text']

            points = [
                [cx - w / 2, cy - h / 2],
                [cx + w / 2, cy - h / 2],
                [cx + w / 2, cy + h / 2],
                [cx - w / 2, cy + h / 2],
                [cx - w / 2, cy - h / 2],
            ]
            points = np.array(points)
            if pos['p_code8'] > 0.5:
                c = 'red'
            else:
                c = 'cyan'
            self.ax_im.plot(points[:,0], points[:,1],color=c)

            if pos['p_code2'] > 0.5:
                points = [
                    [cx - w / 2 - 1, cy - h / 2 - 1],
                    [cx + w / 2 + 1, cy - h / 2 - 1],
                    [cx + w / 2 + 1, cy + h / 2 + 1],
                    [cx - w / 2 - 1, cy + h / 2 + 1],
                    [cx - w / 2 - 1, cy - h / 2 - 1],
                ]
                points = np.array(points)
                self.ax_im.plot(points[:,0], points[:,1],color='yellow')

            if pos['p_code1'] > 0.5:
                points = [
                    [cx - w / 2 + 1, cy - h / 2 + 1],
                    [cx + w / 2 - 1, cy - h / 2 + 1],
                    [cx + w / 2 - 1, cy + h / 2 - 1],
                    [cx - w / 2 + 1, cy + h / 2 - 1],
                    [cx - w / 2 + 1, cy - h / 2 + 1],
                ]
                points = np.array(points)
                self.ax_im.plot(points[:,0], points[:,1],color='magenta')

            if pos['p_code4'] > 0.5:
                points = [
                    [cx - w / 2 + 2, cy - h / 2 + 2],
                    [cx + w / 2 - 2, cy - h / 2 + 2],
                    [cx + w / 2 - 2, cy + h / 2 - 2],
                    [cx - w / 2 + 2, cy + h / 2 - 2],
                    [cx - w / 2 + 2, cy - h / 2 + 2],
                ]
                points = np.array(points)
                self.ax_im.plot(points[:,0], points[:,1],color='blue')

            if text:
                if pos['p_code1'] > 0.5:
                    c = 'green'
                else:
                    c = 'blue'
                self.ax_im.text(cx, cy, text, fontsize=16, color=c, fontproperties=fprop)
        self.im1.draw_idle()

root = tk.Tk()
root.geometry('1400x800')
app = Application(root, target_file)
app.mainloop()

app.save()
