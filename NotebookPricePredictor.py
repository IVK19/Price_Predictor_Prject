import tkinter as tk
from tkinter import ttk
import numpy as np
import re
from regex import graphics_card as gc, display_type as dt, processor_name as pn, processor_class as pc, operational_system as os 
import tensorflow as tf
from tensorflow import keras
import torch

HEIGHT = 700
WIDTH = 800

root = tk.Tk()
root.title('ОЦЕНКА НОУТБУКА')
root.geometry(f'{HEIGHT}x{WIDTH}')
root['bg'] = '#b6fcd5'

root.iconphoto(False, tk.PhotoImage(file='bg1.png'))

def create_frame(label_text):
    frame = tk.Frame(root).place()
    label = tk.Label(frame, text=label_text, background='#b6fcd5')
    label.pack(anchor='w')
    return frame

def selected(event):
    print('Параметр выбран.')

def get_array():
    ntb_lst = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    disp = combobox_display.get()
    disp_lst = re.split('"| |x', disp)
    disp_lst[0] = disp_lst[0].replace(',', '.')
    ntb_lst[0] = float(disp_lst[0])
    ntb_lst[1] = int(disp_lst[2])
    ntb_lst[2] = int(disp_lst[3])
    ntb_lst[3] = dt[combobox_display_type.get()]
    proc = combobox_processor.get()
    proc_lst = proc.split()
    ntb_lst[4] = pn[proc_lst[0]]
    if len(proc_lst) == 2:
        ntb_lst[5] = pc[proc_lst[1]]
    else:
        ntb_lst[5] = pc[proc_lst[1] + ' ' + proc_lst[2]]
    graph = combobox_graphics_cards.get()
    graph_lst = graph.split()
    if graph_lst[0] == 'Nvidia':
        ntb_lst[6] = gc[graph.replace('Nvidia ', '')]
    if graph_lst[0] == 'AMD':
        ntb_lst[6] = gc[graph.replace('AMD ', '')]
    if graph_lst[0] == 'Intel':
        ntb_lst[6] = gc[graph]
    ramm = combobox_ram.get()
    ramm_lst = ramm.split()
    ntb_lst[7] = int(ramm_lst[0])
    hd = combobox_hard_disk.get()
    hd_lst = hd.split()
    if hd_lst[1] == 'ГБ':
        ntb_lst[8] = int(hd_lst[0])
    else:
        ntb_lst[8] = 1024 * int(hd_lst[0])
    ntb_lst[9] = os[combobox_os.get()]
    ntb_lst[10] = int(combobox_cores.get())
    cache = combobox_cache.get()
    cache_lst = cache.split()
    ntb_lst[11] = int(cache_lst[0])
    freq = combobox_frequency.get()
    freq_lst = freq.split()
    freq_lst[0] = freq_lst[0].replace(',', '.')
    ntb_lst[12] = float(freq_lst[0])
    ntb_lst = np.array(ntb_lst)
    return ntb_lst

graphics_cards = ['Nvidia GeForce RTX 3080 Ti 16GB', 'Nvidia GeForce RTX 3080 Ti 8GB', 'Nvidia GeForce RTX 3070 Ti 8GB', 'Nvidia GeForce RTX 3060 6GB', 'Nvidia GeForce RTX 3050 4GB', 'Nvidia GeForce GTX 1650 4GB', 'Nvidia GeForce MX450 2ГБ', 'Nvidia GeForce MX350 2ГБ', 'AMD Radeon Graphics', 'AMD Radeon RX Vega 8', 'AMD Radeon Vega 3', 'Intel Iris Xe Graphics', 'Intel Iris Graphics', 'Intel UHD Graphics 600', 'Intel HD Graphics 5500', 'Intel HD Graphics 500']
display = ['11,6" 1366x768 pxls', '13,4" 1920x1200 pxls', '13,5" 3000x2000 pxls', '14" 1366x768 pxls', '14" 1920x1080 pxls', '14" 2160x1440 pxls', '14,1" 1920x1080 pxls', '14,2" 3120x2080 pxls', '15,6" 1366x768 pxls', '15,6" 1920x1080 pxls', '15,6" 3200x1800 pxls', '17,3" 1920x1080 pxls', '17,3" 2560x1440 pxls', '17,3" 3840x2160 pxls']
display_type = ['LTPS', 'TFT', 'IPS', 'AHVA', 'TN']
processor = ['Intel Core i9', 'Intel Core i7', 'Intel Core i5', 'Intel Core i3', 'Intel Pentium', 'Intel Celeron', 'AMD Ryzen 7', 'AMD Ryzen 5', 'AMD Ryzen 3']
operational_system = ['MacOS', 'Windows 11 Pro 64', 'Windows 11 Pro', 'Windows 11 Домашняя 64', 'Windows 11 Домашняя S-режим', 'Windows 11 Домашняя', 'Windows 10 Pro', 'Windows 10 Домашняя 64', 'Windows 10 Домашняя', 'Linux', 'FreeDOS', 'DOS', 'не установлена']
ram = ['4 ГБ', '8 ГБ', '16 ГБ', '32 ГБ']
hard_disk = ['64 ГБ', '128 ГБ', '256 ГБ', '512 ГБ', '1 ТБ', '2 ТБ']
cores = [2, 4, 6, 8, 10, 12, 14]
cache_memory = ['2 МБ', '4 МБ', '8 МБ', '12 МБ', '16 МБ', '18 МБ', '24 МБ', '30 МБ']
max_frequency = ['2,4 ГГц', '2,6 ГГц', '2,64 ГГц', '2,8 ГГц', '3,4 ГГц', '3,5 ГГц', '3,7 ГГц', '4 ГГц', '4,1 ГГц', '4,2 ГГц', '4,4 ГГц', '4,5 ГГц', '4,7 ГГц', '5 ГГц']

frame_d = create_frame('Выберите диагональ и разрешение экрана:')
combobox_display = ttk.Combobox(frame_d, values=display, width=40, state='readonly')
combobox_display.pack(anchor='w', padx=6, pady=6)
combobox_display.bind('<<ComboboxSelected>>', selected)
frame_dt = create_frame('Выберите тип экрана:')
combobox_display_type = ttk.Combobox(frame_dt, values=display_type, width=40, state='readonly')
combobox_display_type.pack(anchor='w', padx=6, pady=6)
combobox_display_type.bind('<<ComboboxSelected>>', selected)
frame_p = create_frame('Выберите процессор:')
combobox_processor = ttk.Combobox(frame_p, values=processor, width=40, state='readonly')
combobox_processor.pack(anchor='w', padx=6, pady=6)
combobox_processor.bind('<<ComboboxSelected>>', selected)
frame_c = create_frame('Выберите количество ядер процессора:')
combobox_cores = ttk.Combobox(frame_c, values=cores, width=40, state='readonly')
combobox_cores.pack(anchor='w', padx=6, pady=6)
combobox_cores.bind('<<ComboboxSelected>>', selected)
frame_f = create_frame('Выберите максимальную тактовую частоту процессора:')
combobox_frequency = ttk.Combobox(frame_f, values=max_frequency, width=40, state='readonly')
combobox_frequency.pack(anchor='w', padx=6, pady=6)
combobox_frequency.bind('<<ComboboxSelected>>', selected)
frame_gc = create_frame('Выберите видеокарту:')
combobox_graphics_cards = ttk.Combobox(frame_gc, values=graphics_cards, width=40, state='readonly')
combobox_graphics_cards.pack(anchor='w', padx=6, pady=6)
combobox_graphics_cards.bind('<<ComboboxSelected>>', selected)
frame_ram = create_frame('Выберите объём оперативной памяти:')
combobox_ram = ttk.Combobox(frame_ram, values=ram, width=40, state='readonly')
combobox_ram.pack(anchor='w', padx=6, pady=6)
combobox_ram.bind('<<ComboboxSelected>>', selected)
frame_hd = create_frame('Выберите объём памяти жёсткого диска:')
combobox_hard_disk = ttk.Combobox(frame_hd, values=hard_disk, width=40, state='readonly')
combobox_hard_disk.pack(anchor='w', padx=6, pady=6)
combobox_hard_disk.bind('<<ComboboxSelected>>', selected)
frame_cache = create_frame('Выберите объём кэш-памяти:')
combobox_cache = ttk.Combobox(frame_cache, values=cache_memory, width=40, state='readonly')
combobox_cache.pack(anchor='w', padx=6, pady=6)
combobox_cache.bind('<<ComboboxSelected>>', selected)
frame_os = create_frame('Выберите операционную систему:')
combobox_os = ttk.Combobox(frame_os, values=operational_system, width=40, state='readonly')
combobox_os.pack(anchor='w', padx=6, pady=6)
combobox_os.bind('<<ComboboxSelected>>', selected)

label_price = tk.Label(root, width=40, font='Arial 16')
label_price.place(relx=0.1, rely=0.8, relheight=0.1, relwidth=0.8)

def button_tf_click():
    global label_price
    arr = get_array()
    npp_model4 = keras.models.load_model('npp_model_4_1')
    label_price['text'] = f'Цена ноутбука составит: {(int(npp_model4(tf.cast(arr, dtype=float)).numpy()[0][0]) // 1000) * 1000} ₽.'
                              
def button_pt_click():
    global label_price
    arr = get_array()
    npp_model5 = torch.jit.load('model5_1_scripted.pt')
    npp_model5.eval()
    label_price['text'] = f'Цена ноутбука составит: {(int(npp_model5(torch.from_numpy(arr).type(torch.FloatTensor)).item()) // 1000) * 1000} ₽.'

def button_avg_click():
    global label_price
    arr = get_array()
    npp_model4 = keras.models.load_model('npp_model_4_1')
    npp_model5 = torch.jit.load('model5_1_scripted.pt')
    npp_model5.eval()
    avg_price = ((int(npp_model4(tf.cast(arr, dtype=float)).numpy()[0][0]) // 1000) * 1000 + (int(npp_model5(torch.from_numpy(arr).type(torch.FloatTensor)).item()) // 1000) * 1000) // 2
    label_price['text'] = f'Цена ноутбука составит: {avg_price} ₽.'

button_tf = tk.Button(root, text='Определить цену (TF модель)', command=button_tf_click, bg='gray')
button_tf.pack(side=tk.LEFT, expand=1, anchor='n', padx=6, pady=6)
button_pt = tk.Button(root, text='Определить цену (PT модель)', command=button_pt_click, bg='gray')
button_pt.pack(side=tk.LEFT, expand=1, anchor='n', padx=6, pady=6)
button_avg = tk.Button(root, text='Определить цену (в среднем)', command=button_avg_click, bg='gray')
button_avg.pack(side=tk.LEFT, expand=1, anchor='n', padx=6, pady=6)

root.mainloop()