import re
import PySimpleGUI as sg

def select(element):
    element.Widget.select_range(0, 'end')
    element.Widget.icursor('end')

def validate(text):
    result = re.match(regex, text)
    return False if result is None or result.group() != text else True

regex = "^[+-]?([0-5](\.(\d{0,2}))?)?$"
old = {'IN1':'0.00', 'IN2':'0.00'}
validate_inputs = ('IN1', 'IN2')

layout = [
    [sg.Input('0.00', enable_events=True, key='IN1')],
    [sg.Input('0.00', enable_events=True, key='IN2')],
    [sg.Button('Exit')],
]

window = sg.Window('Title', layout, finalize=True)
select(window['IN1'])
for key in validate_inputs:
    window[key].bind('<FocusIn>',  ' IN')
    window[key].bind('<FocusOut>', ' OUT')

while True:
    event, values = window.read()
    if event in ['Exit', sg.WIN_CLOSED]:
        break
    elif event in validate_inputs:
        element, text = window[event], values[event]
        if validate(text):
            try:
                v = float(text)
                if v > 5:
                    element.update(old[event])
                    continue
            except ValueError as e:
                pass
            old[event] = text
        else:
            element.update(old[event])
    elif event.endswith(' IN'):
        key = event.split()[0]
        element, text = window[key], values[key]
        select(element)
    elif event.endswith(' OUT'):
        key = event.split()[0]
        element, text = window[key], values[key]
        try:
            v = float(text)
            element.update(f'{v:.4f}')
        except ValueError as e:
            element.update('0.00')

window.close()