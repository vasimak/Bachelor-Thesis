# PlotGui
import re 
from asyncio.windows_events import CONNECT_PIPE_INIT_DELAY
import matplotlib
import PySimpleGUI as sg
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib as mpl
import time
from functools import partial
from numba import jit
import multiprocessing
from multiprocessing import Pool
multiprocessing.cpu_count()
matplotlib.use('TkAgg')


sg.theme('DarkTeal')

X = []
Y = []
X1 = []
Y1 = []


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


def leplot():
    plt.style.use('dark_background')
    plt.plot(X1, Y1, ".r", alpha=1, ms=1.2)

    plt.axhline(0)
    plt.xlabel("k")
    plt.ylabel("LE")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(1920 / 40, 1080 / 40)
    plt.show()
    return plt


def bifplot():
    plt.style.use('dark_background')
    plt.plot(X, Y, ".w", alpha=1, ms=1.2)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(1920 / 40, 1080 / 40)
    plt.show()
    return plt


def combined():
    plt.style.use('dark_background')
    figure, ax = plt.subplots()
    plt.figure(1)
    plt.subplot(211)
    plt.plot(X, Y, ".w", alpha=1, ms=1.2)
    plt.subplot(212)
    plt.plot(X1, Y1, ".r", alpha=1, ms=1.2)
    plt.axhline(0)
    plt.xlabel("k")
    plt.ylabel("x,LE")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(1920 / 40, 1080 / 40)
    #plt.rcParams.update({"text.usetex": True})
    #print("--- %s seconds ---" % (time.time() - start_time))
    plt.show()
    return plt


exit_button = b'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'


def bif_window():

    layout = [
        [sg.Text('Give Initial Values for Plot', key="new")],
        [sg.Text('Initial x',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Parameter q',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Initial r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('End of r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Steps',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Button('Enter')],
        [sg.Button(image_data=exit_button, border_width=0, image_size=(100, 100),
                   button_color=(sg.theme_background_color(),
                                 sg.theme_background_color()),
                   key='Exit'), ], ]
    window = sg.Window("Bifurcation Plot", layout, modal=True)
    choice = None
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        print(values)
        if event == 'Enter':

            o=all((bool(re.search("((\+|-)?([0-9]+)(\.[0-9]+)?)|((\+|-)?\.?[0-9]+)",str(i)))) for i in values.values())
            print(o)
            if not o:
                sg.popup(
                    "Insert only numbers and characters like '+', '-', '.', '*' ")
                continue
            

     
        r = np.arange(float(values[2]), float(values[3]), float(values[4]))
        x0 = float(values[0])
        q0 = float(values[1])
        bif1 = partial(bif, q0, x0)
        le1 = partial(le, q0, x0)

        #start_time = time.time()

        if event == 'Enter':

            if __name__ == '__main__':
                # create and configure the process pool
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)
                for i, key in enumerate(values):
                    window[i].update("")
                bifplot()
                # window.FindElement().Update('')
                window.refresh()
                continue

    window.close()


def le_window():

    layout = [
        [sg.Text('Give Initial Values for Plot', key="new")],
        [sg.Text('Initial x',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Parameter q',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Initial r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('End of r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Steps',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Button('Enter')],
        [sg.Button(image_data=exit_button, border_width=0, image_size=(100, 100),
                   button_color=(sg.theme_background_color(),
                                 sg.theme_background_color()),
                   key='Exit'), ], ]
    window = sg.Window("Lyapunov Plot", layout, modal=True)
    choice = None
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event == 'Enter':

            if values[0] < chr(42) or values[0] > chr(57) or values[0] == chr(32) or values[0] == chr(0):
                sg.popup(
                    "Insert only numbers and characters like '+', '-', '.', '*' ")
                window[0].update(values[0])
                continue
            elif values[1] < chr(42) or values[1] > chr(57) or values[1] == chr(32) or values[1] == chr(0):
                sg.popup(
                    "Insert only numbers and characters like '+', '-', '.', '*' ")
                window[1].update(values[1])
                continue
            elif values[2] < chr(42) or values[2] > chr(57) or values[2] == chr(32) or values[2] == chr(0):
                sg.popup(
                    "Insert only numbers and characters like '+', '-', '.', '*' ")
                window[2].update(values[2])
                continue
            elif values[3] < chr(42) or values[3] > chr(57) or values[3] == chr(32) or values[3] == chr(0):
                sg.popup(
                    "Insert only numbers and characters like '+', '-', '.', '*' ")
                window[3].update(values[3])
                continue
            elif values[4] < chr(42) or values[4] > chr(57) or values[4] == chr(32) or values[4] == chr(0):
                sg.popup(
                    "Insert only numbers and characters like '+', '-', '.', '*' ")
                window[4].update(values[4])
                continue

        r = np.arange(float(values[2]), float(values[3]), float(values[4]))
        x0 = float(values[0])
        q0 = float(values[1])
        bif1 = partial(bif, q0, x0)
        le1 = partial(le, q0, x0)

        #start_time = time.time()

        if event == 'Enter':

            if __name__ == '__main__':
                for i, ch in enumerate(map(le1, r)):
                    # x1=np.ones(len(str((ch))))*r[i]
                    X1.append(r[i])
                    Y1.append(ch)
                for i, key in enumerate(values):
                    window[i].update("")
                leplot()
                continue

    window.close()


def combined_window():

    layout = [
        [sg.Text('Give Initial Values for Plot', key="new")],
        [sg.Text('Initial x',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Parameter q',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Initial r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('End of r', size=(15, 1), key='Status'), sg.InputText()],
        [sg.Text('Steps',  size=(15, 1), key='Status'), sg.InputText()],
        [sg.Button('Enter')],
        [sg.Button(image_data=exit_button, border_width=0, image_size=(100, 100),
                   button_color=(sg.theme_background_color(),
                                 sg.theme_background_color()),
                   key='Exit'), ], ]
    window = sg.Window("Combined Plots", layout, modal=True)
    choice = None
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event == 'Enter':

            if values[0] < chr(42) or values[0] > chr(57) or values[0] == chr(32) or values[0] == chr(0):
                sg.popup(
                    "Insert only numbers and characters like '+', '-', '.', '*' ")
                window[0].update(values[0])
                continue
            elif values[1] < chr(42) or values[1] > chr(57) or values[1] == chr(32) or values[1] == chr(0):
                sg.popup(
                    "Insert only numbers and characters like '+', '-', '.', '*' ")
                window[1].update(values[1])
                continue
            elif values[2] < chr(42) or values[2] > chr(57) or values[2] == chr(32) or values[2] == chr(0):
                sg.popup(
                    "Insert only numbers and characters like '+', '-', '.', '*' ")
                window[2].update(values[2])
                continue
            elif values[3] < chr(42) or values[3] > chr(57) or values[3] == chr(32) or values[3] == chr(0):
                sg.popup(
                    "Insert only numbers and characters like '+', '-', '.', '*' ")
                window[3].update(values[3])
                continue
            elif values[4] < chr(42) or values[4] > chr(57) or values[4] == chr(32) or values[4] == chr(0):
                sg.popup(
                    "Insert only numbers and characters like '+', '-', '.', '*' ")
                window[4].update(values[4])
                continue

        r = np.arange(float(values[2]), float(values[3]), float(values[4]))
        x0 = float(values[0])
        q0 = float(values[1])
        bif1 = partial(bif, q0, x0)
        le1 = partial(le, q0, x0)

        #start_time = time.time()

        if event == 'Enter':

            if __name__ == '__main__':
                # create and configure the process pool
                for i, ch in enumerate(map(bif1, r)):
                    x1 = np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)

                for i, ch in enumerate(map(le1, r)):
                    X1.append(r[i])
                    Y1.append(ch)
                for i, key in enumerate(values):
                    window[i].update("")
                combined()
                continue

    window.close()


def main():
    layout = [[sg.Text('Choose the Plot you want to run')],
              [sg.Button('Bifurcation Plot', key="open")], [sg.Button(
                  'Lyapunov Plot', key="open1"), sg.Button('Combined Plots', key="open2")],
              [sg.Button(image_data=exit_button, border_width=0, size=(3, 1),
                         button_color=(sg.theme_background_color(),
                                       sg.theme_background_color()),
                         key='Exit'), ]
              ]
    layout_popup = [[sg.Text(
        "Insert the four values, and click the three buttons.\n Initial x = 0 \n q =-0.1 \n Initial r=0 \n End of r=1\n Steps=10000'")], [sg.Button("OK")]]
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
            bif_window()
            continue
        if event == "open1":
            le_window()
            continue
        if event == "open2":
            combined_window()
            continue
        #print("--- %s seconds ---" % (time.time() - start_time))

        #start_time = time.time()

        # if event == 'Lyapunov Plot':
        #     if __name__ == '__main__':
        #             for i,ch in enumerate(map(le1,r)) :
        #                 # x1=np.ones(len(str((ch))))*r[i]
        #                 X1.append(r[i])
        #                 Y1.append(ch)
        #             leplot()
        #             continue

        # #print("--- %s seconds ---" % (time.time() - start_time))

        # if event == 'Combined Plots':

        #     if __name__ == '__main__':
        #         # create and configure the process pool
        #             for i,ch in enumerate(map(bif1,r)) :
        #                 x1=np.ones(len(ch))*r[i]
        #                 X.append(x1)
        #                 Y.append(ch)

        #             for i,ch in enumerate(map(le1,r)) :
        #                 X1.append(r[i])
        #                 Y1.append(ch)
        #             combined()
        #             continue

        # print("--- %s seconds ---" % (time.time() - start_time))

        window.close()


if __name__ == "__main__":
    main()
