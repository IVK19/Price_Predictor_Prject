import re

ntb_lst = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

string = '13,4" 1366x768 pxls'

disp_lst = re.split('"| |x', string)

disp_lst[0] = disp_lst[0].replace(',', '.')

ntb_lst[0] = float(disp_lst[0])
ntb_lst[1] = int(disp_lst[2])
ntb_lst[2] = int(disp_lst[3])

print(ntb_lst)