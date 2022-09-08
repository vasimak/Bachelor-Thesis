# Bifurcation diagram


from cmath import inf
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
from functools import reduce
import PySimpleGUI as sg
import matplotlib
matplotlib.use('TkAgg')


sg.theme('DarkGrey 3')




mpl.use('qtagg')
mpl.rcParams['path.simplify_threshold'] = 1.0

  

while True:  
# Very basic window.
# Return values using
# automatic-numbered keys
      
    X=[]
    Y=[]
    X1=[]
    Y1=[]
                
    @jit(nopython=True)
    def bif(q0,x0,r):
        N=1000  
        x = np.zeros(len(range(0, N)))
        x[0]=x0
        q=q0
        for i in range(1,N):
            x[i]= r *(1 + x[i - 1]) * (1 + x[i- 1]) * (2 - x[i - 1]) + q
        return (x[-130:]) 

    @jit(nopython=True)
    def le(q0,x0,r):
        N=1000
        lyapunov =0
        l1= 0
        x=x0
        q=q0
        for i in range(1,N):
            x = r *(1 + x) * (1 + x) * (2 - x) + q  
            lyapunov += np.log(np.abs(-3*r*(x**2-1))) #derivative of the equation you calculate 
            l1=lyapunov/N
        return (l1) 
           
    def leplot():
        plt.style.use('dark_background')
        plt.plot(X1,Y1, ".r", alpha=1, ms=1.2)
        
        plt.axhline(0)
        plt.xlabel("k")
        plt.ylabel("LE")
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(1920 / 40, 1080 / 40)
        plt.show()
        return plt
    
    def bifplot():
        plt.style.use('dark_background')      
        plt.plot(X,Y, ".w", alpha=1, ms=1.2)
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(1920 / 40, 1080 / 40)
        plt.show()
        return plt

    def combined():
        plt.style.use('dark_background')    
        figure, ax = plt.subplots()
        plt.figure(1)
        plt.subplot(211)
        plt.plot(X,Y, ".w", alpha=1, ms=1.2)
        plt.subplot(212)
        plt.plot(X1,Y1, ".r", alpha=1, ms=1.2)
        plt.axhline(0)
        plt.xlabel("k")
        plt.ylabel("x,LE")
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(1920 / 40, 1080 / 40)
        #plt.rcParams.update({"text.usetex": True})
        #print("--- %s seconds ---" % (time.time() - start_time))
        plt.show()
        return plt
    
    layout = [
        [sg.Text('Give Initial Values for Plot')],
                [sg.Text('Initial x', size =(15, 1)), sg.InputText()],
                [sg.Text('q', size =(15, 1)), sg.InputText()],
                [sg.Text('Initial r', size =(15, 1)), sg.InputText()],
                [sg.Text('End of r', size =(15, 1)), sg.InputText()],
                [sg.Text('steps', size =(15, 1)), sg.InputText()],
                [sg.Button('Bifurcation plot')], [sg.Button('Lyapunov Plot') , sg.Button('Combined Plots')],
                [sg.Button('Exit')],
        #[sg.ButtonMenu('ButtonMenu',  right_click_menu, key='-BMENU-'), sg.Button('Plain Button')],
        #[sg.Output(size=(60, 20))],
    
    ]

    window = sg.Window('Bifurcation diagram', layout, resizable=True,
    finalize=True,
    element_justification="center")

   

    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break 
    
    if values[0]<chr(42) or values[0]>chr(57) or values[0]==chr(32) or values[0]==chr(0):
        sg.popup("Insert only numbers and characters like '+', '-', '.','*' ")
        continue
    elif values[1]<chr(42) or values[1]>chr(57) or values[1]==chr(32) or values[1]==chr(0):
        sg.popup("Insert only numbers and characters like '+', '-', '.', '*' ")
        continue 
    elif  values[2]<chr(42) or values[2]>chr(57) or values[2]==chr(32) or values[2]==chr(0):
        sg.popup("Insert only numbers and characters like '+', '-', '.', '*' ")
        continue
    elif values[3]<chr(42) or values[3]>chr(57) or values[3]==chr(32) or values[3]==chr(0):
        sg.popup("Insert only numbers and characters like '+', '-', '.', '*' ")
        continue     
    elif values[4]<chr(42) or values[4]>chr(57) or values[4]==chr(32) or values[4]==chr(0):
        sg.popup("Insert only numbers and characters like '+', '-', '.', '*' ")
        continue          

    r=np.arange(float(values[2]),float(values[3]),float(values[4])) 
    x0=float(values[0])
    q0=float(values[1])
    bif1= partial(bif,q0,x0)
    le1= partial(le,q0,x0)


    start_time = time.time()

    if event =='Bifurcation plot':
        if __name__ == '__main__':
        # create and configure the process pool
            with Pool(4) as p:
                    for i,ch in enumerate(p.map(bif1,r,chunksize=2500)) :
                        x1=np.ones(len(ch))*r[i]
                        X.append(x1)
                        Y.append(ch)
            bifplot()
            #plt.plot(X,Y, ".w", alpha=1, ms=1.2)
            # canvas size
            # plt.style.use('dark_background')      
            # plt.plot(X,Y, ".w", alpha=1, ms=1.2)
            # figure = plt.gcf()  # get current figure
            # figure.set_size_inches(1920 / 40, 1080 / 40)
            # print("--- %s seconds ---" % (time.time() - start_time))
            # plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))
    if event == 'Lyapunov Plot':
        if __name__ == '__main__':
            with Pool(4) as p:
                    for i,ch in enumerate(p.map(le1,r,chunksize=2500)) :
                        # x1=np.ones(len(str((ch))))*r[i]
                        X1.append(r[i])
                        Y1.append(ch)
                    
            leplot()

    if event == 'Combined Plots':
        if __name__ == '__main__':
            # create and configure the process pool
            with Pool(8) as p:
                
                for i,ch in enumerate(p.map(bif1,r,chunksize=2500)) :
                    x1=np.ones(len(ch))*r[i]
                    X.append(x1)
                    Y.append(ch)

            with Pool(8) as p:

                for i,ch in enumerate(p.map(le1,r,chunksize=25000)) :
                    X1.append(r[i])
                    Y1.append(ch)
            
            combined()


    

  
    window.close() 

   
    
    