import re
import pandas as pd

data = pd.read_csv('selected_notebooks_data_4.csv')

df = {'диагональ': [], 'ширина': [], 'высота': [], 'дисплей': [], 'марка_процессора': [], 'семейство_процессора': [], 'видеокарта': [], 'оперативная_память': [], 'память_жесткого_диска': [], 'о_с': [], 'ядра': [], 'кэш_память': [], 'макс_такт_частота': [], 'цена': []}

graphics_card = {'GeForce RTX 3080 Ti 16GB': 17, 'GeForce RTX 3080 Ti для ноутбуков 8GB': 16, 'GeForce RTX 3080 Ti 8GB': 16, 'GeForce RTX 3070 Ti для ноутбуков 8GB': 15, 'GeForce RTX 3070 Ti 8GB': 15, 'GeForce RTX 3070 для ноутбуков 8GB': 15, 'GeForce RTX 3060 для ноутбуков 6GB': 14, 'GeForce RTX 3060 6GB': 14, 'GeForce RTX 3050 для ноутбуков 6GB': 13, 'GeForce RTX 3050 Ti для ноутбуков 4GB': 12, 'GeForce RTX 3050 Ti 4GB': 12, 'GeForce RTX 3050 для ноутбуков 4GB': 12, 'GeForce RTX 3050 4GB': 12, 'Radeon Graphics': 11,'GeForce GTX 1650 4GB': 10, 'GeForce MX450 2ГБ': 9, 'GeForce MX350 2ГБ': 9, 'Intel Iris Xe Graphics': 8, 'Radeon RX Vega 8': 8, 'Radeon': 7, 'Intel UHD Graphics': 6, 'Intel HD Graphics': 5, 'Radeon Vega 3': 4, 'Intel Iris Graphics': 4, 'Intel UHD Graphics 600': 3, 'UHD Graphics 600': 3, 'Intel HD Graphics 5500': 2, 'Intel HD Graphics 500': 1}
display_type = {'LTPS': 4, 'TFT': 3, 'IPS': 3, 'IPS-level': 3, 'AHVA': 2, 'TN': 1}
processor_name = {'Intel': 2, 'AMD': 1}
processor_class = {'Core i9': 9, 'Core i7': 8, 'Core i5': 7, 'Ryzen 7': 6, 'Ryzen 5': 5, 'Ryzen 3': 4, 'Core i3': 3, 'Pentium': 2, 'Celeron': 1}
operational_system = {'MacOS': 7, 'Windows 11 Pro 64': 6, 'Windows 11 Pro': 6, 'Windows 11 Домашняя 64': 5, 'Windows 11 Домашняя S-режим': 5, 'Windows 11 Домашняя': 5, 'Windows 11': 5, 'Windows 10 Pro': 4, 'Windows 10 Домашняя 64': 3, 'Windows 10 Домашняя': 2, 'Linux': 1, 'FreeDOS': 1, 'DOS': 1, 'не установлена': 0}


# for n in range(23, 102):
#     if n == 55 or n == 63 or n == 64 or n == 74 or n == 75 or n == 88 or n == 92 or n == 96 or n == 97 or n == 99:
#         continue
    
#     info_lst = re.split(';|:|/', data['information'][n])
#     print(info_lst)
#     if len(info_lst) > 1:
#         info_lst.pop()
#     if info_lst[13] == '1 ТБ':
#         info_lst[13] = '1024 ГБ'
#     if info_lst[13] == '2 ТБ':
#         info_lst[13] = '2048 ГБ'
                
#     df['диагональ'].append(float(info_lst[6].replace(' ', '').replace('"', '')))
#     df['ширина'].append(int(info_lst[7][:4]))
#     df['высота'].append(int(info_lst[7][5:9]))
#     df['дисплей'].append(display_type[info_lst[8]])
#     df['марка_процессора'].append(processor_name[info_lst[9]])
#     df['семейство_процессора'].append(processor_class[info_lst[10]])
#     df['видеокарта'].append(graphics_card[info_lst[11]])
#     df['оперативная_память'].append(int(info_lst[12].replace(' ГБ', '')))
#     df['память_жесткого_диска'].append(int(info_lst[13].replace(' ГБ', '')))
#     df['о_с'].append(operational_system[info_lst[14]])
#     df['ядра'].append(int(info_lst[16].replace(' ', '')))
#     df['кэш_память'].append(int(info_lst[18].replace(' ', '').replace('МБ', '')))
#     df['макс_такт_частота'].append(float(info_lst[20].replace(' ', '').replace('ГГц', '')))
#     df['цена'].append(data['price'][n])
    
# df1 = pd.DataFrame(df)
# df1.sort_values(by='цена', ascending=True)
# df1.reset_index(drop=True , inplace=True)
# print(df1)
# df1.to_excel('modified_notebooks_data_frame_6.xlsx', index=False)
           