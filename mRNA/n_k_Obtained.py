import requests
import xlwt
import xlrd
import re
import os
from fake_useragent import UserAgent

def geturl(url, headers):
    try:
        with requests.get(url, timeout=20, headers=headers) as r:
	    #r = requests.get(url, headers={'Connection': 'close'})
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            return r.text
    except:
        print('connect fault')

def parserinfo(html, keyword):
    infolist = []
    html = str(html)
    infonums = re.findall(r'\"resultcount\" value=\"[\d\.]*', html)
    #(r'Items: 1 to [\d\.]* of [\d\.]*', html)
    for infonum in infonums:
        info = infonum.split("value=\"")[-1]
        infolist.append([keyword, info])
    return infolist


# 读取excel数据并返回[[sheetName,[cols]]]
def read_data(inputfile):
    mRNA = []
    wb = xlrd.open_workbook(inputfile)
    sheetNames = wb.sheet_names()
    for sheetName in sheetNames:
        sheet = wb.sheet_by_name(sheetName)
        cols = sheet.col_values(0)
        mRNA.append([sheetName, cols])
    return mRNA

def main():
    starturl = "https://www.ncbi.nlm.nih.gov/pubmed?term=" #网址接口
    path = os.getcwd()
    inputfile = path + "/miRNAs.xlsx" #输入表格名字
    outputfile = path + "/output.xls"

    outputFile = xlwt.Workbook()
    # 抓取数据
    mRNA = read_data(inputfile) # [['lung cancer', [mrna,mrna,rna]], ['ssss', [sddfdf,fdfd,fdff]]]

    for i in range(len(mRNA)):
        sheetName = mRNA[i][0] # 'name'
        cols = mRNA[i][1] # '[m-1, m-2, m-3]'

        sheet = outputFile.add_sheet(sheetName) # 增加一个sheet

        for i, col in enumerate(cols):
            url1 = starturl + col # search n
            url2 = starturl + col + " AND " + "'" + sheetName + "'"  # search k
            print(url1)
            print(url2)
            ua = UserAgent()
            headers = {'User-Agent': ua.random}
            html_n = geturl(url1, headers)
            html_k = geturl(url2, headers)
            info_n = parserinfo(html_n, col) # [col, n]
            info_k = parserinfo(html_k, col) # [col, k]

            if len(info_n) !=  0 and len(info_k) != 0:
                info_n = info_n[0]
                info_k = info_k[0]

		# 写入
                info_n.append(info_k[1])
                print(info_n, i, len(cols))
                for j in range(len(info_n)):
                    sheet.write(i, j, info_n[j])

    outputFile.save(outputfile)


if __name__ == "__main__":
    main()
