import os
import xlwt
import xlrd
from scipy.stats import hypergeom

def main():
    inputfile = os.getcwd() + '/n_k_N_K.xls'
    outputWd = xlwt.Workbook() # write files
    wd = xlrd.open_workbook(inputfile)
    sheetNames = wd.sheet_names()
    for sheetName in sheetNames:
        sheet_out = outputWd.add_sheet(sheetName) # add a sheet
        sheet = wd.sheet_by_name(sheetName)
        N_values = sheet.col_values(3)
        K_values = sheet.col_values(4)
        n_values = sheet.col_values(1)
        k_values = sheet.col_values(2)

        if len(N_values) == len(K_values) == len(n_values) == len(k_values):
            for i in range(len(N_values)):
                prb = hypergeom.cdf(float(k_values[i]), float(N_values[i]),
                        float(K_values[i]), float(n_values[i]))
                pValue = 1 - prb
                sheet_out.write(i, 5, pValue)
        else:
            print('len(N),len(n),len(k),len(K) not matched!')

    outputWd.save(os.getcwd() + '/Final_p_mRNA.xlsx')


if __name__ == '__main__':
    main()

