import PySimpleGUI as sg
import base64

layout = [  [sg.Listbox(list(range(10)), size=(10,5), key='-LBOX-')],
            [sg.T('Name'), sg.In()],
            [sg.T('Address'), sg.In()],
            [sg.Button('Go'), sg.Button('Exit')]  ]

window = sg.Window('Window Title', layout,auto_size_text=False, default_element_size=(12,1))

while True:             # Event Loop
    event, values = window.read()
    print(event, values)
    if event in (None, 'Exit'):
        break
window.close()

exit()