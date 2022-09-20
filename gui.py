# PlotGui

#from asyncio.streams import _ClientConnectedCallback
import re
import PySimpleGUI as sg
from functools import reduce
import numpy as np
import math
import csv
import time
from functools import partial
from numba import jit
import gc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.figure as figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

matplotlib.use('TkAgg')

sg.theme('DarkTeal')

X = []
Y = []
X1 = []
Y1 = []
fig = figure.Figure()
ax = fig.add_subplot(111)
DPI = fig.get_dpi()
fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))


# ------------------------------- This is to include a matplotlib figure in a Tkinter canvas
def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
    toolbar.update()
    figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)


def line_select_callback(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    rect = plt.Rectangle((min(x1, x2), min(y1, y2)),
                         np.abs(x1-x2), np.abs(y1-y2))
    print(rect)
    ax.add_patch(rect)
    fig.canvas.draw()


class Toolbar(NavigationToolbar2Tk):
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)


@jit(nopython=True, parallel=True)
def bif(q0, x0, r):
    N = 1000
    x = np.zeros(len(range(0, N)))
    x[0] = x0
    q = q0
    for i in range(1, N):
        x[i] = r * (1 + x[i - 1]) * (1 + x[i - 1]) * (2 - x[i - 1]) + q
    return (x[-130:])


@jit(nopython=True, parallel=True)
def le(q0, x0, r):
    N = 1000
    lyapunov = 0
    l1 = 0
    x = x0
    q = q0
    for i in range(1, N):
        x = r * (1 + x) * (1 + x) * (2 - x) + q
        # derivative of the equation you calculate
        lyapunov += np.log(np.abs(-3*r*(x**2-1)))
        l1 = lyapunov/N
    return (l1)


def extralogistic_window():

    layout = [
        [sg.Text('Give Initial Values for Plot', key="new")],
        [sg.Text('Initial x',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Parameter q',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Initial r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('End of r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Steps',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Button('Bifurcation Plot')],
        [sg.Button('Lyapunov Plot')],
        [sg.Button('Combined Plots')],
        [sg.Canvas(key='controls_cv')],
        [sg.Column(
            layout=[
                [sg.Canvas(key='fig_cv',
                               # it's important that you set this size
                               size=(400 * 2, 400)
                           )]
            ],
            background_color='#DAE0E6',
            pad=(0, 0)
        )],
        [sg.Button('Exit')],
    ]
    window = sg.Window("Bifurcation Plot", layout, resizable=True, finalize=True, grab_anywhere=True)

    while True:

        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        print(values)

        values_new = {}
        if event == 'Bifurcation Plot' or event == 'Lyapunov Plot' or event == 'Combined Plots':
            for i, key in enumerate(values.keys()):
                if i <= 4:
                    values_new[key] = values[key]
                    print(values_new)
                o = all((bool(re.fullmatch(
                    "((\+|-)?([0-9]+)(\.[0-9]+)?)|((\+|-)?\.?[0-9])", str(j)))) for j in values_new.values())
                if not o:
                    sg.popup(
                        "Insert only numbers and characters like '+', '-', '.', '*' ")
                    window.close()
                    continue

        r = np.arange(float(values[2]), float(values[3]), float(values[4]))
        x0 = float(values[0])
        q0 = float(values[1])
        bif1 = partial(bif, q0, x0)
        le1 = partial(le, q0, x0)

        #start_time = time.time()

        if event == 'Bifurcation Plot':

            X = []
            Y = []
            # create and configure the process pool
            for i, ch in enumerate(map(bif1, r)):
                x1 = np.ones(len(ch))*r[i]
                X.append(x1)
                Y.append(ch)
            # for i, key in enumerate(values):
            #     window[i].update("")
            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            line = ax.plot(X, Y, ".k", alpha=1, ms=1.2)

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            # window.FindElement().Update('')
            window.refresh()
            continue
        if event == 'Lyapunov Plot':

            X1 = []
            Y1 = []
            for i, ch in enumerate(map(le1, r)):
                X1.append(r[i])
                Y1.append(ch)
            fig = figure.Figure()
            ax = fig.add_subplot(111)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            # ax.cla()
            ax.plot(X1, Y1, ".k", alpha=1, ms=1.2)
            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            window.refresh()
            continue
        if event == 'Combined Plots':
            X = []
            Y = []
            X1 = []
            Y1 = []
            for i, ch in enumerate(map(le1, r)):
                X1.append(r[i])
                Y1.append(ch)
            for i, ch in enumerate(map(bif1, r)):
                x1 = np.ones(len(ch))*r[i]
                X.append(x1)
                Y.append(ch)

            fig = figure.Figure()
            ax = fig.add_subplot(211)
            DPI = fig.get_dpi()
            fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
            ax.cla()
            ax.plot(X, Y, ".k", alpha=1, ms=1.2)
            ax = fig.add_subplot(212)
            ax.cla()
            ax.plot(X1, Y1, ".r", alpha=1, ms=1.2)
            plt.axhline(0)
            plt.xlabel("k")
            plt.ylabel("x,LE")

            rs = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=False, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
            draw_figure_w_toolbar(
                window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            for i, key in enumerate(values):
                window[i].update("")
            window.refresh()
            continue

    window.close()


def main():
    layout = [  [sg.Text('Choose the Plot you want to run')],
                [sg.Button('Variation of Logistic Map', key="open")], 
                [sg.Button( 'Chebyshev Map', key="open1")],
                [sg.Button('Sine-Sinh Map', key="open2")],
                [sg.Button('Exit'),sg.B('Help')]
              ]
    layout_popup = [[sg.Text(
        "Insert the four values, and click the three buttons. Example above:\nInitial x = 0 \nq =-0.1 \nInitial r=0 \nEnd of r=1\nSteps=10000")], [sg.Button("OK")]]
    window = sg.Window('Bifurcation diagram',  layout, size=(
        500, 500), resizable=True, finalize=True, grab_anywhere=True)
    #window.bind('<Configure>', "Configure")
    window_help = sg.Window("Help", layout_popup)

    while True:
        window, event, values = sg.read_all_windows()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Help":
            window_help.read()
            if event == "Cancel" or event == sg.WIN_CLOSED:
                window_help.close()
                continue
        if event == "open":
            extralogistic_window()
            continue
        if event == "open1":
            le_window()
            continue
        if event == "open2":
            combined_window()
            continue
        
        # y[i+1] = r * y[i] * (1 - y[i])) % logistic
        #x(i)=r(j)*x(i-1)*(1-x(i-1))*(2+x(i-1))%cubic logistic
        #x(i)=r(j)*x(i-1)*(1-x(i-1)^2) %cubic
        # x(i)=mod(k*x(i-1),1); %renyi
        # x(i)=cos(k*acos(x(i-1))); %cheb       
        #x(i)=r(j)*sin(pi*x(i-1)) %sine
        # x(i)=k*sin(pi*sinh(pi*sin(pi*x(i-1)))); %sine-sinh

        #math . cos ( k **q * math . acos ( q*x [ i − 1 ] ) ) parallagh cheb
        #x[i] = k * math.sin(k * math.sinh(q * math.sin(2 * x[i - 1]))) parallagh np.sine - np.sinh
        window.close()


if __name__ == "__main__":
    main()
