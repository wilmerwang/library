import xlrd

book = xlrd.open_workbook("../TMA_dot_position.xlsx")
sheet1 = book.sheets()[0]
sheet2 = book.sheets()[1]

dotNane_row = []
for i in range(sheet2.nrows):
    if str(sheet2.cell(i,0).value).startswith("875"):
        dotNane_row.append(i)

for i, dotNane_r in enumerate(dotNane_row):
    text_name = str(int(sheet2.cell(dotNane_r, 0).value))
    file = open(text_name + ".txt", "a")
    for j in range(dotNane_r+1, sheet2.nrows):
        for k in range(sheet2.ncols):
            a = sheet2.cell(j,k).value
            if str(a).startswith("875"):
                break
            if a != '':
                if a == 42:
                    a = "#N/A"
                if sheet2.cell(j,k).ctype in (2,3) and int(a) == a:
                    a = int(a)
                file.writelines(str(a) + "\n")
        if str(a).startswith("875"):
            break



