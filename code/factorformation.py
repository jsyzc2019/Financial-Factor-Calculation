# -*- coding:utf-8 -*-
'''
 @author: yueyang li
 @last edition:2020-02-20
'''
# %%这里是数据截止的年份，比如做到2019年12月数据这里就写2019,2020年1月数据就写2020
GLOBALDate = '2019'
# %% base package
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from importlib import reload
import gc, re, warnings, datetime

# %% base function
def readingwhole_Data(path, encoding='gb18030', dayind=False, txtind='csv'):
    '''
    读取整个数据，并按照股票顺序排列
    :param path: 路径
    :param encoding: 编码
    :param dayind:按日期排序的名字，默认False
    :return: 合并好的文件框
    '''
    filelist = os.listdir(path)
    tempfile = []
    [tempfile.append(pd.read_csv(path + x, encoding=encoding)) for x in filelist if x[-3:] == txtind]
    dataframe = pd.concat(tempfile)
    if dayind:
        dataframe.sort_values(['Stkcd', dayind], inplace=True)
    else:
        dataframe.sort_values(['Stkcd'], inplace=True)
    dataframe = dataframe.reset_index(drop=True)
    gc.collect()
    return dataframe

def readingwholetxt_Data(path, encoding='UTF-8', dayind=False, txtind='csv'):
    '''
    读取整个数据，并按照股票顺序排列
    :param path: 路径
    :param encoding: 编码
    :param dayind:按日期排序的名字，默认False
    :return: 合并好的文件框
    '''
    filelist = os.listdir(path)
    tempfile = []
    [tempfile.append(pd.read_csv(path + x, encoding=encoding, sep='\t')) for x in filelist if x[-3:] == txtind]
    dataframe = pd.concat(tempfile)
    if dayind:
        dataframe.sort_values(['Stkcd', dayind], inplace=True)
    else:
        dataframe.sort_values(['Stkcd'], inplace=True)
    dataframe = dataframe.reset_index(drop=True)
    gc.collect()
    return dataframe

def formdate(x):
    rel = str(datetime.datetime.strptime(x+'-5','%Y-%W-%w'))[0:10]
    x = rel[0:4]+'/'+rel[5:7]+'/'+rel[-2:]
    return x

def addmonth(date):
    '''
    月份+1
    :param date: 日期
    :return: 返回月份
    '''
    year = int(date[:4])
    month = int(date[5:7])
    if month == 12:
        newmonth = str(year + 1) + '/01'
    else:
        mon = '0' + str(month + 1)
        newmonth = str(year) + '/' + mon[-2:]
    return newmonth


def changeSheetdate(sheet, acceperid, annoucement):
    '''
    把会计日期转化为报表发布日期
    :param sheet: 报表
    :param acceperid: 会计期间在第几列
    :param annoucementdate: 会计报表发布日期
    :return: 增加完真实日期后的报表
    '''

    def rowChange(datarow, acceperid, annoucement):
        '''
        内置函数，转化单一日期
        :param datarow:一行数据
        :param acceperid: 会计报表期
        :param annoucement: 会计报表和公布日期
        :return: 返回月份
        '''
        try:
            return addmonth(annoucement[(annoucement['Stkcd'] == datarow.iloc[0]) & (
                        annoucement['Accper'] == datarow.iloc[acceperid])]['Actudt'].iloc[0])
        except:
            return addmonth(datarow.iloc[acceperid])

    Actudtlist = [rowChange(sheet.iloc[x, :], acceperid, annoucement) for x in range(sheet.shape[0])]
    sheet['Actudt'] = Actudtlist
    return sheet


def calna(datalist, controlp=120):
    '''
    计算日度数据na指标数目是否符合标准
    :param datalist: 数据列表
    :param controlp: 最低可用数值
    :return:
    '''
    lsnan = np.isnan(datalist)
    controlpercent = len(datalist) - np.count_nonzero(lsnan)
    if controlpercent < controlp:
        return True
    else:
        return False


def calnam(datalist, controlp=10):
    '''
    计算月度数据na指标数是否符合标准
    :param datalist: 数据列表
    :param controlp: 最低可用数值
    :return:
    '''
    lsnan = np.isnan(datalist)
    controlpercent = len(datalist) - np.count_nonzero(lsnan)
    if controlpercent < controlp:
        return True
    else:
        return False


def cal12(x):
    uselist = [np.nan] * 11 + [x.iloc[i - 11: i + 1].mean() for i in range(11, len(x))]
    x = uselist
    return x


def fillna(stockdata, fillzero=False):
    '''
    填充na指标, 其实可以用列表生成器改简单。
    :param stockdata:被填充数据，列数据/行数据
    :return: 返回填充后的数据
    '''
    if fillzero:
        for i in range(len(stockdata)):
            if np.isnan(stockdata[i]):
                if i == 0:
                    stockdata[i] = 0
                else:
                    stockdata[i] = stockdata[i - 1]
    else:
        for i in range(1, len(stockdata)):
            if np.isnan(stockdata[i]):
                stockdata[i] = stockdata[i - 1]
    return stockdata


def fillsheet(sheet, fillzero=False):
    '''
    填充整张表格
    :param sheet: 被填充表格
    :return: 无返回直接在表格内部修改
    '''
    sheet.apply(lambda x: fillna(x, fillzero))


def checkdivision(stockdata):
    '''
    将被除数为0的填充na指标
    :param stockdata:被填充数据，列数据/行数据
    :return: 返回填充后的数据
    '''
    return [np.nan if x == 0 else x for x in stockdata]


def cal_monthend(date):
    '''
    计算日度数据月初指标
    :param date: 日期数
    :return: 返回数值值，即每月月初日期
    '''
    newind = []
    for i in date:
        newind.append(int(i[-2:]))
    outlist = [0]
    outlist.extend(list(np.array(np.where((np.array(newind[1:]) - np.array(newind[:-1]) < 0))[0]) + 1))
    return outlist


def fillIntheBlank(returnData, blankFrame, stocksid, datanum=2):
    '''
    用于填充数据
    :param returnData:收益数据或是其他数据。
    :param blankFrame:空白dataframe
    :param stocksid:股票代码
    :param datanum:要填充的数据
    :return:
    '''
    for i in stocksid:
        stockreturn = returnData[returnData['Stkcd'] == i]
        blankFrame.loc[stockreturn.iloc[:, 1], i] = list(stockreturn.iloc[:, datanum])
        print('stock ' + str(i) + ' finished')
    return 'mission finished'

def fillIntheBlank2(returnData, blankFrame, stocksid, datanum=2):
    '''
    用于填充数据
    :param returnData:收益数据或是其他数据。
    :param blankFrame:空白dataframe
    :param stocksid:股票代码
    :param datanum:要填充的数据
    :return:
    '''
    for i in stocksid:
        stockreturn = returnData[returnData['Stkcd'] == i]
        blankFrame.loc[stockreturn.iloc[:, -1], i] = list(stockreturn.iloc[:, datanum])
        print('stock ' + str(i) + ' finished')
    return 'mission finished'


def changedateExpression(date, index=True):
    '''
    转换时间格式成'yyyy/mm/dd'格式，以便统一。
    :param date:原时间
    :return:
    '''
    if index:
        return date[:4] + '/' + date[5:7] + '/' + date[-2:]
    else:
        return date[:4] + '-' + date[5:7]


def checkid(x):
    '''
    去掉截止日期之后的上市公告日期的数据所用
    :param x:
    :return:
    '''
    if x[:4] == str(int(GLOBALDate) + 1):
        x = GLOBALDate + '-12'
    return x


def checkstep2(frame):
    '''
    针对数据框处理
    :param frame:
    :return:
    '''
    temp = [checkid(x) for x in frame.iloc[:, -1]]
    frame.iloc[:, -1] = temp
    return frame


def combinetax(tax_a, tax_b):
    '''
    计算税金合并
    :param tax_a:税种a
    :param tax_b: 税种b
    :return:
    '''
    tax_combine = tax_a + tax_b
    if np.isnan(tax_combine):
        if np.isnan(tax_a):
            return tax_b
        else:
            return tax_a
    return tax_combine


# %% factor function
def size_calculator(monthdata, sizeposition, stockid, dateid, save_path='../output/01_size.csv'):
    '''
    计算size
    :param monthdata:全部月度数据
    :param sizeposition: size在的位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    sizeframe = pd.DataFrame(index=dateid, columns=stockid)
    fillIntheBlank(monthdata, sizeframe, stockid, datanum=sizeposition)
    sizeframe.to_csv(save_path)
    return sizeframe


def size_ia_calculator(size, type, save_path='../output/02_size_ia.csv'):
    '''
    计算行业调整size
    :param size: 市值, 按照stkcd为index
    :param type: 类型按照第一列股票id第二列类型
    :param save_path: 存储数据
    :return:
    '''
    typelist = set(type.iloc[:, 1])

    def onerow(x):
        for i in typelist:
            stocklist = type[type['type'] == i].iloc[:, 0]
            indusmean = np.nanmean(list(x.loc[stocklist]))
            x.loc[stocklist] -= indusmean
        return x

    size_ia = size.T.apply(lambda x: onerow(x))
    size_ia.T.to_csv(save_path)
    return size_ia.T


# 以下需要日期X股票的数据框
def beta_calculator(stockdata, marketdata, stockid, dateid, monthtopindex, rf, save_path='../output/03_beta.csv'):
    '''
    计算beta
    :param stockdata: 股票数据
    :param marketdata: 市场收益数据
    :param stockid: 股票id
    :param dateid: 日期id'yyyy-mm-dd'格式
    :param monthtopindex: 月初或月末指数
    :param rf: 无风险收益率
    :param save_path: 存储路径
    :return:
    '''

    def cal_beta(stockdata, marketdata, monthind, rf):
        betalist = []
        for i in range(len(monthind)):
            if i < 12:
                betalist.append(np.nan)
            else:
                datatemp = list(stockdata.iloc[monthind[i - 12]:monthind[i]])
                if calna(datatemp):
                    betalist.append(np.nan)
                else:
                    non_nan_index = list(np.where(~np.isnan(datatemp))[0])
                    datatemp = np.array(datatemp) - np.array(list(rf.iloc[monthind[i - 12]:monthind[i], 0]))
                    mktdata = np.array(
                        list(marketdata.iloc[monthind[i - 12]:monthind[i], 0] - rf.iloc[monthind[i - 12]:monthind[i],
                                                                                0]))
                    mktdata = mktdata[non_nan_index]
                    datatempnew = list(np.array(datatemp)[non_nan_index])
                    mktdata = mktdata.reshape(-1, 1)
                    lir = LinearRegression()
                    lir.fit(mktdata, datatempnew)
                    betalist.append(lir.coef_[0])
        return betalist

    timelist = np.array(dateid)[monthtopindex]
    timeindex = [x[:-3] for x in timelist]
    betaframe = pd.DataFrame(columns=stockid, index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_beta(inputdata, marketdata, monthtopindex, rf)
        betaframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s beta finished')
    print('all finished')
    betaframe.to_csv(save_path)
    return betaframe


def betasq_calculator(betasheet, save_path='../output/04_betasq.csv'):
    '''
    计算beta的平方
    :param betasheet: beta表格
    :return: 返回betasq
    '''
    betasq = betasheet ** 2
    betasq.to_csv(save_path)
    return betasq


def betad_calculator(stockdata, marketdata, stockid, dateid, monthtopindex, rf, save_path='../output/05_betad.csv'):
    '''
    计算dimson-beta
    :param stockdata: 股票数据
    :param marketdata: 市场收益数据
    :param stockid: 股票id
    :param dateid: 日期id'yyyy-mm-dd'格式
    :param monthtopindex: 月初或月末指数
    :param rf: 无风险收益率
    :param save_path: 存储路径
    :return:
    '''

    def cal_betad(stockdata, marketdata, monthind, rf):
        betadlist = []
        for i in range(len(monthind)):
            if i < 12:
                betadlist.append(np.nan)
            else:
                datatemp = list(stockdata.iloc[monthind[i - 12]:monthind[i]])
                if calna(datatemp):
                    betadlist.append(np.nan)
                else:
                    non_nan_index = list(np.where(~np.isnan(datatemp))[0])
                    datatemp = np.array(datatemp) - np.array(list(rf.iloc[monthind[i - 12]:monthind[i], 0]))
                    mktdata0 = list(
                        marketdata.iloc[monthind[i - 12]:monthind[i], 0] - rf.iloc[monthind[i - 12]:monthind[i], 0])
                    Msf = marketdata.shift(1)
                    Msf.iloc[0] = 0
                    rsf = rf.shift(1)
                    rsf.iloc[0, 0] = 0
                    mktdata1 = list(
                        Msf.iloc[monthind[i - 12]:monthind[i], 0] - rsf.iloc[monthind[i - 12]:monthind[i], 0])
                    Mof = marketdata.shift(-1)
                    Mof.iloc[-1] = 0
                    rof = rf.shift(-1)
                    rof.iloc[-1, 0] = 0
                    mktdata01 = list(
                        Mof.iloc[monthind[i - 12]:monthind[i], 0] - rof.iloc[monthind[i - 12]:monthind[i], 0])
                    mktdata = [mktdata0, mktdata1, mktdata01]
                    mktdata = pd.DataFrame(mktdata)
                    mktdata = mktdata.T
                    mktdata = mktdata.iloc[non_nan_index, :]
                    datatempnew = list(np.array(datatemp)[non_nan_index])
                    lir = LinearRegression()
                    lir.fit(mktdata, datatempnew)
                    betadlist.append(sum(lir.coef_))
        return betadlist

    timelist = np.array(dateid)[monthtopindex]
    timeindex = [x[:-3] for x in timelist]
    betaframe = pd.DataFrame(columns=stockid, index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_betad(inputdata, marketdata, monthtopindex, rf)
        betaframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s betad finished')
    print('all finished')
    betaframe.to_csv(save_path)
    return betaframe


def idvol_calculator(stockdata, marketdata, stockid, dateid, monthtopindex, rf, save_path='../output/06_idvol.csv'):
    '''
    计算异质波动率
    :param stockdata: 股票数据
    :param marketdata: 市场收益数据
    :param stockid: 股票id
    :param dateid: 日期id'yyyy-mm-dd'格式
    :param monthtopindex: 月初或月末指数
    :param rf: 无风险收益率
    :param save_path: 存储路径
    :return:
    '''

    def cal_idovol(stockdata, marketdata, monthind, rf):
        idovollist = []
        for i in range(len(monthind)):
            if i < 11:
                idovollist.append(np.nan)
            else:
                datatemp = list(stockdata.iloc[monthind[i - 12]:monthind[i]])
                if calna(datatemp):
                    idovollist.append(np.nan)
                else:
                    non_nan_index = list(np.where(~np.isnan(datatemp))[0])
                    datatemp = np.array(datatemp) - np.array(list(rf.iloc[monthind[i - 12]:monthind[i], 0]))
                    mktdata = np.array(
                        list(marketdata.iloc[monthind[i - 12]:monthind[i], 0] - rf.iloc[monthind[i - 12]:monthind[i],
                                                                                0]))
                    mktdata = mktdata[non_nan_index]
                    datatempnew = list(np.array(datatemp)[non_nan_index])
                    mktdata = mktdata.reshape(-1, 1)
                    lir = LinearRegression()
                    lir.fit(mktdata, datatempnew)
                    fittedvalue = lir.predict(mktdata)
                    residuals = fittedvalue - datatempnew
                    idovollist.append(np.std(residuals))
        return idovollist

    timelist = np.array(dateid)[monthtopindex]
    timeindex = [x[:-3] for x in timelist]
    idovolframe = pd.DataFrame(columns=stockid, index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_idovol(inputdata, marketdata, monthtopindex, rf)
        idovolframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s idiovol finished')
    idovolframe.to_csv(save_path)
    print('all finished')
    return idovolframe


def retvol_calculator(stockdata, stockid, dateid, monthtopindex, save_path='../output/07_retvol.csv'):
    '''
    计算收益率波动率
    :param stockdata: 股票数据
    :param stockid: 股票id
    :param dateid: 日期id'yyyy-mm-dd'格式
    :param monthtopindex: 月初或月末指数
    :param save_path: 存储路径
    :return:
    '''

    def cal_retvol(stockdata, monthind):
        retvollist = []
        for i in range(len(monthind)):
            if i < 2:
                retvollist.append(np.nan)
            else:
                datatemp = list(stockdata.iloc[monthind[i - 1]:monthind[i]])
                if calnam(datatemp):
                    retvollist.append(np.nan)
                else:
                    retvollist.append(np.nanstd(datatemp))
        return retvollist

    timelist = np.array(dateid)[monthtopindex]
    timeindex = [x[:-3] for x in timelist]
    retvolframe = pd.DataFrame(columns=stockid, index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_retvol(inputdata, monthtopindex)
        retvolframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s retvol finished')
    retvolframe.to_csv(save_path)
    return retvolframe


def idskewness_calculator(stockdata, marketdata, stockid, dateid, monthtopindex, rf,
                          save_path='../output/08_idskewness.csv'):
    '''
    计算异质偏度
    :param stockdata: 股票数据
    :param marketdata: 市场收益数据
    :param stockid: 股票id
    :param dateid: 日期id'yyyy-mm-dd'格式
    :param monthtopindex: 月初或月末指数
    :param rf: 无风险收益率
    :param save_path: 存储路径
    :return:
    '''

    def cal_idskewness(stockdata, marketdata, monthind, rf):
        idskewnesslist = []
        for i in range(len(monthind)):
            if i < 12:
                idskewnesslist.append(np.nan)
            else:
                datatemp = list(stockdata.iloc[monthind[i - 12]:monthind[i]])
                if calna(datatemp):
                    idskewnesslist.append(np.nan)
                else:
                    non_nan_index = list(np.where(~np.isnan(datatemp))[0])
                    datatemp = np.array(datatemp) - np.array(list(rf.iloc[monthind[i - 12]:monthind[i], 0]))
                    mktdata = np.array(
                        list(marketdata.iloc[monthind[i - 12]:monthind[i], 0] - rf.iloc[monthind[i - 12]:monthind[i],
                                                                                0]))
                    mktdata = mktdata[non_nan_index]
                    datatempnew = list(np.array(datatemp)[non_nan_index])
                    mktdata = mktdata.reshape(-1, 1)
                    lir = LinearRegression()
                    lir.fit(mktdata, datatempnew)
                    fittedvalue = lir.predict(mktdata)
                    residuals = fittedvalue - datatempnew
                    residualsnew = (residuals - np.mean(residuals)) ** 3
                    skew = np.mean(residualsnew) / (np.std(residuals) ** 3)
                    idskewnesslist.append(skew)
        return idskewnesslist

    timelist = np.array(dateid)[monthtopindex]
    timeindex = [x[:-3] for x in timelist]
    idskewnessframe = pd.DataFrame(columns=stockid, index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_idskewness(inputdata, marketdata, monthtopindex, rf)
        idskewnessframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s idskewness finished')
    print('all finished')
    idskewnessframe.to_csv(save_path)
    return idskewnessframe


def skewness_calculator(stockdata, stockid, dateid, monthtopindex, save_path='../output/09_skewness.csv'):
    '''
    计算偏度
    :param stockdata: 股票数据
    :param stockid: 股票id
    :param dateid: 日期id'yyyy-mm-dd‘格式
    :param monthtopindex: 月初或月末指数
    :param save_path: 存储路径
    :return:
    '''

    def cal_skewness(stockdata, monthind):
        skewnesslist = []
        for i in range(len(monthind)):
            if i < 12:
                skewnesslist.append(np.nan)
            else:
                datatemp = list(stockdata.iloc[monthind[i - 12]:monthind[i]])
                if calna(datatemp):
                    skewnesslist.append(np.nan)
                else:
                    datatemp = np.array(datatemp) - np.nanmean(datatemp)
                    skew = np.nanmean(datatemp ** 3) / (np.nanstd(datatemp)) ** 3
                    skewnesslist.append(skew)
        return skewnesslist

    timelist = np.array(dateid)[monthtopindex]
    timeindex = [x[:-3] for x in timelist]
    skewness = pd.DataFrame(columns=stockid, index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_skewness(inputdata, monthtopindex)
        skewness.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s skewness finished')
    print('all finished')
    skewness.to_csv(save_path)
    return skewness


def coskewness_calculator(stockdata, marketdata, stockid, dateid, monthtopindex, rf,
                          save_path='../output/10_coskewness.csv'):
    '''
    计算共同偏度
    :param stockdata: 股票数据
    :param marketdata: 市场收益数据
    :param stockid: 股票id
    :param dateid: 日期id'yyyy-mm-dd'格式
    :param monthtopindex: 月初或月末指数
    :param rf: 无风险收益率
    :param save_path: 存储路径
    :return:
    '''

    def cal_coskewness(stockdata, marketdata, monthind, rf):
        coskewnesslist = []
        for i in range(len(monthind)):
            if i < 12:
                coskewnesslist.append(np.nan)
            else:
                datatemp = list(stockdata.iloc[monthind[i - 12]:monthind[i]])
                if calna(datatemp):
                    coskewnesslist.append(np.nan)
                else:
                    non_nan_index = list(np.where(~np.isnan(datatemp))[0])
                    datatemp = np.array(datatemp) - np.array(list(rf.iloc[monthind[i - 12]:monthind[i], 0]))
                    mktdata0 = list(
                        marketdata.iloc[monthind[i - 12]:monthind[i], 0] - rf.iloc[monthind[i - 12]:monthind[i], 0])
                    mktdatasq = list(np.array(mktdata0) ** 2)
                    mktdata = [mktdata0, mktdatasq]
                    mktdata = pd.DataFrame(mktdata)
                    mktdata = mktdata.T
                    mktdata = mktdata.iloc[non_nan_index, :]
                    datatempnew = list(np.array(datatemp)[non_nan_index])
                    lir = LinearRegression()
                    lir.fit(mktdata, datatempnew)
                    coskewnesslist.append(lir.coef_[1])
        return coskewnesslist

    timelist = np.array(dateid)[monthtopindex]
    timeindex = [x[:-3] for x in timelist]
    coskewnessframe = pd.DataFrame(columns=stockid, index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_coskewness(inputdata, marketdata, monthtopindex, rf)
        coskewnessframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s coskewness finished')
    print('all finished')
    coskewnessframe.to_csv(save_path)
    return coskewnessframe


# 日期X换手率数据框
def turn_calculator(stockdata, stockid, monthtopindex, dateid, save_path='../output/11_turn.csv'):
    '''
    计算换手率平均值
    :param stockdata: 换手率数据
    :param stockid: 股票id
    :param monthtopindex: 月初或月末指标
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''

    def cal_turn(stockdata, monthind):
        turnlist = []
        for i in range(len(monthind)):
            if i < 12:
                turnlist.append(np.nan)
            else:
                datatemp = list(stockdata.iloc[monthind[i - 12]:monthind[i]])
                if calna(datatemp):
                    turnlist.append(np.nan)
                else:
                    turnlist.append(np.nanmean(datatemp))
        return turnlist

    timelist = np.array(dateid)[monthtopindex]
    timeindex = [x[:-3] for x in timelist]
    turnframe = pd.DataFrame(columns=stockid, index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_turn(inputdata, monthtopindex)
        turnframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s turn finished')
    turnframe.to_csv(save_path)
    return turnframe


def turnsd_calculator(stockdata, stockid, monthtopindex, dateid, save_path='../output/12_turnsd.csv'):
    '''
    计算换手率波动率
    :param stockdata: 换手率数据
    :param stockid: 股票id
    :param monthtopindex: 月初或月末指标
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''

    def cal_turnsd(stockdata, monthind):
        turnsdlist = []
        for i in range(len(monthind)):
            if i < 12:
                turnsdlist.append(np.nan)
            else:
                datatemp = list(stockdata.iloc[monthind[i - 1]:monthind[i]])
                if calnam(datatemp):
                    turnsdlist.append(np.nan)
                else:
                    turnsdlist.append(np.nanstd(datatemp) + np.random.randn(1)[0] / 100)
        return turnsdlist

    timelist = np.array(dateid)[monthtopindex]
    timeindex = [x[:-3] for x in timelist]
    turnsdframe = pd.DataFrame(columns=stockid, index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_turnsd(inputdata, monthtopindex)
        turnsdframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s turnsd finished')
    turnsdframe.to_csv(save_path)
    return turnsdframe


# 此处开始需要日期X个股交易额数据框
def vold_calculator(stockdata, stockid, monthtopindex, dateid, save_path='../output/13_vold.csv'):
    '''
    计算交易额均值
    :param stockdata: 交易量数据
    :param stockid: 股票id
    :param monthtopindex: 月初或月末指标
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''

    def cal_vold(stockdata, monthind):
        voldlist = []
        for i in range(len(monthind)):
            if i < 12:
                voldlist.append(np.nan)
            else:
                datatemp = list(stockdata.iloc[monthind[i - 12]:monthind[i]])
                if calna(datatemp):
                    voldlist.append(np.nan)
                else:
                    voldlist.append(np.nanmean(datatemp))
        return voldlist

    timelist = np.array(dateid)[monthtopindex]
    voldframe = pd.DataFrame(columns=stockid, index=timelist)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_vold(inputdata, monthtopindex)
        voldframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s vold finished')
    voldframe.to_csv(save_path)
    return voldframe


def stdvold_calculator(stockdata, stockid, monthtopindex, dateid, save_path='../output/14_stdvold.csv'):
    '''
    计算交易量的波动率
    :param stockdata: 交易量数据
    :param stockid: 股票id
    :param monthtopindex: 月初或月末指标
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''

    def cal_voldsd(stockdata, monthind):
        voldsdlist = []
        for i in range(len(monthind)):
            if i < 12:
                voldsdlist.append(np.nan)
            else:
                datatemp = list(stockdata.iloc[monthind[i - 1]:monthind[i]])
                if calnam(datatemp):
                    voldsdlist.append(np.nan)
                else:
                    voldsdlist.append(np.nanstd(datatemp))
        return voldsdlist

    timelist = np.array(dateid)[monthtopindex]
    timeindex = [x[:-3] for x in timelist]
    voldsdframe = pd.DataFrame(columns=stockid, index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_voldsd(inputdata, monthtopindex)
        voldsdframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s voldsd finished')
    voldsdframe.to_csv(save_path)
    return voldsdframe


# 收益率数据
def retmax_calculator(stockdata, stockid, dateid, monthtopindex, save_path='../output/15_retmax.csv'):
    '''
    最大收益率
    :param stockdata: 收益率数据
    :param stockid: 股票id
    :param dateid: 日期id
    :param monthtopindex: 月初月末数据
    :param save_path: 存储路径
    :return:
    '''

    def cal_retmax(stockdata, monthind):
        retmaxlist = []
        for i in range(len(monthind)):
            if i < 2:
                retmaxlist.append(np.nan)
            else:
                datatemp = list(stockdata.iloc[monthind[i - 1]:monthind[i]])
                if calnam(datatemp):
                    retmaxlist.append(np.nan)
                else:
                    retmaxlist.append(np.nanmax(datatemp))
        return retmaxlist

    timelist = np.array(dateid)[monthtopindex]
    timeindex = [x[:-3] for x in timelist]
    retmaxframe = pd.DataFrame(columns=stockid, index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_retmax(inputdata, monthtopindex)
        retmaxframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s retmax finished')
    retmaxframe.to_csv(save_path)
    return retmaxframe


# illq数据-已达标
# wd['illq'] = np.abs(wd['Dretwd'])/wd['Dnvaltrd']*1000000
def illq_calculator(stockdata, stockid, monthtopindex, dateid, save_path='../output/16_illq.csv'):
    '''
    计算非流动性风险
    :param stockdata:
    :param stockid:
    :param monthtopindex:
    :param dateid:
    :param save_path:
    :return:
    '''

    def cal_illq(stockdata, monthind):
        illqlist = []
        for i in range(len(monthind)):
            if i < 12:
                illqlist.append(np.nan)
            else:
                datatemp = list(stockdata.iloc[monthind[i - 12]:monthind[i]])
                if calna(datatemp):
                    illqlist.append(np.nan)
                else:
                    illqlist.append(np.nanmean(datatemp))
        return illqlist

    timelist = np.array(dateid)[monthtopindex]
    timeindex = [x[:-3] for x in timelist]
    illqframe = pd.DataFrame(columns=stockid, index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_illq(inputdata, monthtopindex)
        illqframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s illq finished')
    print('mission finished')
    illqframe.to_csv(save_path)
    return illqframe


# 需要换手率frame- 已达标
def LM_calculator(stockdata, stockid, monthtopindex, dateid, save_path='../output/17_LM.csv'):
    '''
    LM计算
    :param stockdata:换手率
    :param stockid:股票id
    :param monthtopindex:月初月末id
    :param dateid:日期id
    :param save_path:存储路径
    :return:
    '''

    def cal_LM(stockdata, monthind, deflator=480000):
        LMlist = []
        for i in range(len(monthind)):
            if i < 12:
                LMlist.append(np.nan)
            else:
                datatemp = list(stockdata.iloc[monthind[i - 1]:monthind[i]])
                if calnam(datatemp):
                    LMlist.append(np.nan)
                else:
                    numtrade = len(datatemp)
                    numofzero = np.count_nonzero(np.array(datatemp) == 0)  # 计算交易为0的数目
                    turnoversum = np.nansum(datatemp)
                    deflatoruse = 1 / deflator
                    LM = (numofzero + deflatoruse / turnoversum) * (21 / numtrade)
                    LMlist.append(LM)
        return LMlist

    timelist = np.array(dateid)[monthtopindex]
    timeindex = [x[:-3] for x in timelist]
    LMframe = pd.DataFrame(columns=stockid, index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_LM(inputdata, monthtopindex)
        LMframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s LM finished')
    LMframe.to_csv(save_path)
    return LMframe


# 以下需要流通股本的frame 月度
# eg: monthlyshare['shares'] = monthlyshare['Msmvosd']/monthlyshare['Mclsprc']
def sharechg_calculator(stockdata, stockid, dateid, save_path='../output/18_sharechg.csv'):
    '''
    计算股本变化率
    :param stockdata: 流通股本数据
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 保存路径
    :return:
    '''

    def cal_sharechange(stockdata):
        sharechangelist = []
        for i in range(len(stockdata)):
            if i < 12:
                sharechangelist.append(np.nan)
            else:
                t_1 = stockdata.iloc[i - 1]
                t_12 = stockdata.iloc[i - 12]
                if np.isnan(t_1) | np.isnan(t_12):
                    sharechangelist.append(np.nan)
                else:
                    sharechangelist.append(t_1 / t_12 - 1)
        return sharechangelist

    timelist = dateid
    sharechangeframe = pd.DataFrame(columns=stockid, index=timelist)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_sharechange(inputdata)
        sharechangeframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s sharechange finished')
    sharechangeframe.to_csv(save_path)
    print('mission finished')
    return sharechangeframe


def age_calculator(ipodate, ipouse, stockid, save_path='../output/19_age.csv'):
    '''
    计算年龄
    :param ipodate:IPO上市时间
    :param ipouse: 股票收益数据
    :param stockid: 股票id
    :param save_path: 存储路径
    :return:
    '''

    def ipodevide(x, pattern, anothertype=False):
        if anothertype:
            return x[:4], x[-2:]
        rel = re.split(pattern, x)
        ipoyear = rel[1]
        ipomonth = rel[2]
        return ipoyear, ipomonth

    stockreturn = ipouse
    pattern = re.compile('(.*)/(.*)/(.*)')
    for i in stockid:
        useindex = stockreturn.loc[:, i][~np.isnan(stockreturn.loc[:, i])].index
        res = ipodevide(ipodate[ipodate['Stkcd'] == i]['Listdt'].values[0], pattern)
        ipoyear = int(res[0])
        ipomonth = int(res[1])
        for j in range(len(useindex)):
            rer = ipodevide(useindex[j], pattern, True)
            firstyear = int(rer[0])
            firstmonth = int(rer[1])
            yearchar = firstyear - ipoyear
            monthchar = firstmonth - ipomonth
            month = yearchar * 12 + monthchar
            stockreturn.loc[useindex[j], i] = month
        print('stock ' + str(i) + '\'s age finished')
    stockreturn = stockreturn // 12
    print('mission finished')
    stockreturn.to_csv(save_path)
    return stockreturn


def aeavol_calculator(stockdata, ann, stockid, dateid, fsreturn, save_path='../output/20_aeavol.csv'):
    '''
    计算公告日异常交易量
    :param stockdata: 交易额数据
    :param ann: 公告日数据
    :param stockid:股票id
    :param dateid:日期id
    :param fsreturn: 收益率数据
    :param save_path:存储路径
    :return:
    '''

    def cal_aeavol(voludata, eadatea):
        n = len(eadatea)
        indexaea = voludata.index
        aeavol = []
        for i in range(n):
            tempdate = eadatea.iloc[i]
            if type(tempdate) != str:
                aeavol.append(np.nan)
            else:
                year = int(tempdate[0:4])
                month = int(tempdate[5:7])
                if month == 1:
                    yearbegin = year - 1
                    month = 12
                else:
                    yearbegin = year
                monthbegin = month - 1
                if monthbegin < 10:
                    monthbstr = '0' + str(monthbegin)
                else:
                    monthbstr = str(monthbegin)
                if month < 10:
                    monthstr = '0' + str(month)
                else:
                    monthstr = str(month)
                beginind = str(yearbegin) + '/' + monthbstr + '/15'
                endind = str(year) + '/' + monthstr + '/15'
                monthind = indexaea[indexaea < endind]
                monthind = monthind[monthind > beginind]
                dateind = indexaea[indexaea < tempdate]
                dateind = dateind[-3:]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    datevol = np.nanmean(voludata.loc[dateind])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    monthvol = np.nanmean(voludata.loc[monthind])
                aeavol.append(datevol / monthvol - 1)
        return aeavol

    def fillaeavol(base, aeaind):
        baseindex = base.index
        useindex = aeaind.index
        outlist = []
        for i in range(len(baseindex)):
            if baseindex[i] in useindex:
                outlist.append(aeaind.loc[baseindex[i]])
            else:
                outlist.append(np.nan)
        outlist[0] = np.nanmean(aeaind.iloc[0:5])
        outlist = fillna(outlist)
        return outlist

    def takesamenan(aim, base):
        for i in range(len(base.index)):
            if np.isnan(base.iloc[i]):
                aim[i] = np.nan
        return aim

    def fillallaeavol(base, aeaind):
        stocks = base.columns
        time = base.index
        newframe = pd.DataFrame(index=time, columns=stocks)
        for i in stocks:
            temp = fillaeavol(base.loc[:, i], aeaind.loc[:, i])
            temp = takesamenan(temp, base.loc[:, i])
            newframe.loc[:, i] = temp
            print('stock ' + str(i) + ' finished')
        print('all finished')
        return newframe

    aeavolframe = pd.DataFrame(columns=stockid, index=dateid)
    aeavolframe.index = [x[:4] + '-' + x[-2:] for x in aeavolframe.index]
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        announceuse = ann.iloc[:, i]
        tempresult = cal_aeavol(inputdata, announceuse)
        aeavolframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s aeavol finished')
    print('1/2 finished')
    newframe = fillallaeavol(fsreturn, aeavolframe)
    newframe.to_csv(save_path)
    return newframe

# 需求周度数据
def pricedelay_calculator(stockdata, marketdata, stockid, dateid, monthtopindex,
                          save_path='../output/21_pricedelay.csv'):
    '''
    计算pricedelay
    :param stockdata:周度数据
    :param marketdata: 周度数据市场收益
    :param stockid: 股票id
    :param dateid: 日期id
    :param monthtopindex:月初月末数据
    :param save_path: 保存路径
    :return:
    '''

    def cal_pricedelay(stockdata, marketdata, monthind):
        pricedelaylist = []
        for i in range(len(monthind)):
            if monthind[i] < 39:
                pricedelaylist.append(np.nan)
            else:
                datazero = list(stockdata.iloc[monthind[i] - 36:monthind[i]])
                if calnam(datazero):
                    pricedelaylist.append(np.nan)
                else:
                    weekofmonth = monthind[i] - monthind[i - 1]
                    D1 = []
                    for j in range(weekofmonth):
                        uid = monthind[i] - j
                        datatemp = list(stockdata.iloc[(uid - 36):uid])
                        datatemp = fillna(datatemp, True)
                        datatemp = np.array(datatemp[4:])
                        marketuse = marketdata.iloc[(uid - 36):uid]
                        m1 = marketuse.shift(4)
                        m2 = marketuse.shift(3)
                        m3 = marketuse.shift(2)
                        m4 = marketuse.shift(1)
                        mktdata0 = list(marketuse.iloc[:, 0])[4:]
                        mktdata00 = np.array(mktdata0)
                        mktdata00 = mktdata00.reshape(-1, 1)
                        mktdata1 = list(m1.iloc[:, 0])[4:]
                        mktdata2 = list(m2.iloc[:, 0])[4:]
                        mktdata3 = list(m3.iloc[:, 0])[4:]
                        mktdata4 = list(m4.iloc[:, 0])[4:]
                        mktdata = [mktdata0, mktdata1, mktdata2, mktdata3, mktdata4]
                        mktdata = pd.DataFrame(mktdata)
                        mktdata = mktdata.T
                        lir = LinearRegression()
                        lir.fit(mktdata, datatemp)
                        r2base = lir.score(mktdata, datatemp)
                        lir.fit(mktdata00, datatemp)
                        r2up = lir.score(mktdata00, datatemp)
                        D1.append(1 - r2up / r2base)
                    pricedelaylist.append(np.nanmean(D1))
        return pricedelaylist

    timelist = np.array(dateid)[monthtopindex]
    timeindex = [x[:-3] for x in timelist]
    pricedelayframe = pd.DataFrame(columns=stockid, index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_pricedelay(inputdata, marketdata, monthtopindex)
        pricedelayframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s pricedelay finished')
    print('all finished')
    pricedelayframe.to_csv(save_path)
    return pricedelayframe


# 需要收益frame-已达标
def mom12_calculator(stockdata, stockid, dateid, monthtopindex, save_path='../output/22_mom12.csv'):
    def cal_mom12(stockdata, monthind):
        momlist = []
        for i in range(len(monthind)):
            if i < 12:
                momlist.append(np.nan)
            else:
                momlist.append(np.nansum(list(stockdata.iloc[monthind[i - 12]:monthind[i - 1]])))
        return momlist

    timelist = np.array(dateid)[monthtopindex]
    timeindex = [x[:-3] for x in timelist]
    momframe = pd.DataFrame(columns=stockid, index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_mom12(inputdata, monthtopindex)
        momframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s mom finished')
    momframe.to_csv(save_path)
    print('mission finished')
    return momframe


def mom6_calculator(stockdata, stockid, dateid, monthtopindex, save_path='../output/23_mom6.csv'):
    def cal_mom6(stockdata, monthind):
        momlist = []
        for i in range(len(monthind)):
            if i < 6:
                momlist.append(np.nan)
            else:
                momlist.append(np.nansum(list(stockdata.iloc[monthind[i - 6]:monthind[i - 1]])))
        return momlist

    timelist = np.array(dateid)[monthtopindex]
    timeindex = [x[:-3] for x in timelist]
    momframe = pd.DataFrame(columns=stockid, index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_mom6(inputdata, monthtopindex)
        momframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s mom finished')
    momframe.to_csv(save_path)
    print('mission finished')
    return momframe


def mom36_calculator(stockdata, stockid, dateid, monthtopindex, save_path='../output/24_mom36.csv'):
    def cal_mom36(stockdata, monthind):
        momlist = []
        for i in range(len(monthind)):
            if i < 36:
                momlist.append(np.nan)
            else:
                momlist.append(np.nansum(list(stockdata.iloc[monthind[i - 36]:monthind[i - 13]])))
        return momlist

    timelist = np.array(dateid)[monthtopindex]
    timeindex = [x[:-3] for x in timelist]
    momframe = pd.DataFrame(columns=stockid, index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_mom36(inputdata, monthtopindex)
        momframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s mom finished')
    momframe.to_csv(save_path)
    print('mission finished')
    return momframe


def momchg_calculator(stockdata, stockid, dateid, monthtopindex, save_path='../output/25_momchg.csv'):
    def cal_momchg(stockdata, monthind):
        momlist = []
        for i in range(len(monthind)):
            if i < 12:
                momlist.append(np.nan)
            else:
                momlist.append(np.nansum(list(stockdata.iloc[monthind[i - 7]:monthind[i - 1]]))
                               - np.nansum(list(stockdata.iloc[monthind[i - 12]:monthind[i - 7]])))
        return momlist

    timelist = np.array(dateid)[monthtopindex]
    timeindex = [x[:-3] for x in timelist]
    momframe = pd.DataFrame(columns=stockid, index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_momchg(inputdata, monthtopindex)
        momframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s mom finished')
    momframe.to_csv(save_path)
    print('mission finished')
    return momframe


def imom_calculator(stockdata, marketdata, stockid, dateid, monthtopindex, rf, save_path='../output/26_imom.csv'):
    def cal_imom(stockdata, marketdata, monthind, rf):
        imomlist = []
        for i in range(len(monthind)):
            if i < 11:
                imomlist.append(np.nan)
            else:
                datatemp = list(stockdata.iloc[monthind[i - 12]:monthind[i - 1]])
                if calna(datatemp):
                    imomlist.append(np.nan)
                else:
                    non_nan_index = list(np.where(~np.isnan(datatemp))[0])
                    datatemp = np.array(datatemp) - np.array(list(rf.iloc[monthind[i - 12]:monthind[i - 1], 0]))
                    mktdata = np.array(list(
                        marketdata.iloc[monthind[i - 12]:monthind[i - 1], 0] - rf.iloc[monthind[i - 12]:monthind[i - 1],
                                                                               0]))
                    mktdata = mktdata[non_nan_index]
                    datatempnew = list(np.array(datatemp)[non_nan_index])
                    mktdata = mktdata.reshape(-1, 1)
                    lir = LinearRegression()
                    lir.fit(mktdata, datatempnew)
                    fittedvalue = lir.predict(mktdata)
                    residuals = fittedvalue - datatempnew
                    imomlist.append(np.sum(residuals))
        return imomlist

    timelist = np.array(dateid)[monthtopindex]
    timeindex = [x[:-3] for x in timelist]
    imomframe = pd.DataFrame(columns=stockid, index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_imom(inputdata, marketdata, monthtopindex, rf)
        imomframe.iloc[:, i] = tempresult
        print('stock ' + str(stockid[i]) + '\'s imom finished')
    imomframe.to_csv(save_path)
    print('all finished')
    return imomframe


def largretn_calculator(monthdata, largretnposition, stockid, dateid, save_path='../output/27_largretn.csv'):
    '''
    计算短期反转largretn
    :param monthdata: 全部月度收益率
    :param largretnposition: 收益率所在位置
    :param stockid: 股票代码
    :param dateid: 日期代码
    :param save_path: 存储位置
    :return: 返回数据框
    '''
    largretnframe = pd.DataFrame(index=dateid, columns=stockid)
    fillIntheBlank(monthdata, largretnframe, stockid, datanum=largretnposition)
    largretnframe.iloc[1:, :] = largretnframe.iloc[:-1, :]
    largretnframe.iloc[0, :] = np.nan
    largretnframe.to_csv(save_path)
    return largretnframe


def BM_calculator(balancesheet, size, balanceid, stockid, dateid, save_path='../output/28_BM.csv'):
    '''
    计算book_to_market, 所有者权益合计/A股流通市值
    :param balancesheet: 资产负债表
    :param size: 流通市值表
    :param balanceid: 所有者权益所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    BMframe = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        BMframe.loc[list(balanceStock.iloc[:, -1]), i] = list(
            np.array(balanceStock.iloc[:, balanceid]))  # numpy 有他自己判断标准，直接除
        print('stock ' + str(i) + ' finished')
    fillsheet(BMframe)
    BMframe.columns = list(size.columns)
    BMframe = BMframe / size
    BMframe.columns = stockid
    BMframe.to_csv(save_path)
    return BMframe


def BM_ia_calculator(BM, type, save_path='../output/29_BM_ia.csv'):
    '''
      计算行业调整BM
      :param BM: BMframe
      :param type: 类型按照第一列股票id第二列类型
      :param save_path: 存储数据
      :return:
      '''
    return size_ia_calculator(BM, type, save_path)


def AM_calculator(balancesheet, size, balanceid, stockid, dateid, save_path='../output/30_AM.csv'):
    '''
    计算asset_to_market, 总资产/A股流通市值
    :param balancesheet: 资产负债表
    :param size: A股流通市值
    :param balanceid: 总资产所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    return BM_calculator(balancesheet, size, balanceid, stockid, dateid, save_path)


def LEV_calculator(balancesheet, size, balanceid, stockid, dateid, save_path='../output/31_LEV.csv'):
    '''
    计算liabilities_to_market, 总负债/A股流通市值
    :param balancesheet: 资产负债表
    :param size: A股流通市值
    :param balanceid: 总负债所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    return BM_calculator(balancesheet, size, balanceid, stockid, dateid, save_path)


def EP_calculator(balancesheet, size, balanceid, stockid, dateid, save_path='../output/32_EP.csv'):
    '''
    计算earnings-to-price, 净利润/A股流通市值
    :param balancesheet: 资产负债表
    :param size: A股流通市值
    :param balanceid: 净利润所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    return BM_calculator(balancesheet, size, balanceid, stockid, dateid, save_path)


def CFP_calculator(balancesheet, size, balanceid, stockid, dateid, save_path='../output/33_CFP.csv'):
    '''
    计算cashflow-to-price, 净现金流（现金及现金等价物增加值）/A股流通市值
    :param balancesheet: 利润表
    :param size: A股流通市值
    :param balanceid: 净现金流所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    return BM_calculator(balancesheet, size, balanceid, stockid, dateid, save_path)


def CFP_ia_calculator(CFP, type, save_path='../output/34_CFP_ia.csv'):
    '''
      计算行业调整CFP
      :param CFP: CFPframe
      :param type: 类型按照第一列股票id第二列类型
      :param save_path: 存储数据
      :return:
      '''
    return size_ia_calculator(CFP, type, save_path)


def OCFP_calculator(balancesheet, size, balanceid, stockid, dateid, save_path='../output/35_OCFP.csv'):
    '''
    计算operating cashflow-to-price, 营业现金流/A股流通市值
    :param balancesheet: 现金流量表
    :param size: A股流通市值
    :param balanceid: 营业现金流所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    return BM_calculator(balancesheet, size, balanceid, stockid, dateid, save_path)


def DP_calculator(balancesheet, size, balanceid, stockid, dateid, save_path='../output/36_DP.csv'):
    '''
    计算dividend-to-price, 应付股利/A股流通市值
    :param balancesheet: 资产负债表
    :param size: A股流通市值
    :param balanceid: 应付股利所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    return BM_calculator(balancesheet, size, balanceid, stockid, dateid, save_path)


def SP_calculator(balancesheet, size, balanceid, stockid, dateid, save_path='../output/37_SP.csv'):
    '''
    计算sales-to-price, 营业收入/A股流通市值
    :param balancesheet: 资产负债表
    :param size: A股流通市值
    :param balanceid: 营业收入所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    return BM_calculator(balancesheet, size, balanceid, stockid, dateid, save_path)


def growthbath(balancesheet, balanceid, stockid, dateid, save_path):
    '''
    增长率系列计算基础函数
    :param balancesheet: 财务报表
    :param balanceid: 对应id
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    frame = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        frame.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame)
    newframe = frame / frame.shift(12).apply(lambda x: checkdivision(x)) - 1
    newframe.to_csv(save_path)
    return newframe


def AG_calculator(balancesheet, balanceid, stockid, dateid, save_path='../output/38_AG.csv'):
    '''
    计算Asset-grows-ratio, 总资产增长率
    :param balancesheet: 资产负债表
    :param balanceid: 总资产所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    AGnew = growthbath(balancesheet, balanceid, stockid, dateid, save_path)
    return AGnew


def LG_calculator(balancesheet, balanceid, stockid, dateid, save_path='../output/39_LG.csv'):
    '''
    计算liabilites-grows-ratio, 总负债增长率
    :param balancesheet: 资产负债表
    :param balanceid: 总负债所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    LGnew = growthbath(balancesheet, balanceid, stockid, dateid, save_path)
    return LGnew


def BVEG_calculator(balancesheet, balanceid, stockid, dateid, save_path='../output/40_BVEG.csv'):
    '''
    计算Book market values-grows-ratio, 净资产增长率, 直接用所有者权益合计来做
    :param balancesheet: 资产负债表
    :param balanceid: 所有者权益合计所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    return growthbath(balancesheet, balanceid, stockid, dateid, save_path)


def SG_calculator(balancesheet, balanceid, stockid, dateid, save_path='../output/41_SG.csv'):
    '''
    计算Sales-grows-ratio, 营业收入增长率
    :param balancesheet: 利润表
    :param balanceid: 营业收入所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    SGnew = growthbath(balancesheet, balanceid, stockid, dateid, save_path)
    return SGnew


def PMG_calculator(balancesheet, balanceid, stockid, dateid, save_path='../output/42_PMG.csv'):
    '''
    计算Profitmargin-grows-ratio, 营业利润增长率
    :param balancesheet: 利润表
    :param balanceid: 营业利润所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    PMGnew = growthbath(balancesheet, balanceid, stockid, dateid, save_path)
    return PMGnew


def INVG_calculator(balancesheet, balanceid, stockid, dateid, save_path='../output/43_INVG.csv'):
    '''
    计算inventory-grows-ratio, 存货增长率
    :param balancesheet: 资产负债表
    :param balanceid: 存货所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    INVGnew = growthbath(balancesheet, balanceid, stockid, dateid, save_path)
    return INVGnew


def INVGchg_calculator(balancesheet, balanceid1, balanceid2, stockid, dateid, save_path='../output/44_INVGchg.csv'):
    '''
    计算inventory-grows-ratio, 存货增长率
    :param balancesheet: 资产负债表
    :param balanceid1: 存货所在位置
    :param balanceid2: 资产所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    INVTframe = pd.DataFrame(index=dateid, columns=stockid)
    Assetframe = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        INVTframe.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid1])
        Assetframe.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid2])
        print('stock ' + str(i) + ' finished')
    fillsheet(INVTframe)
    fillsheet(Assetframe)
    Assetframe.apply(lambda x: cal12(x))
    INVGchg = INVTframe - INVTframe.shift(12)
    INVGchgnew = INVGchg / Assetframe.apply(lambda x: checkdivision(x))
    INVGchgnew.to_csv(save_path)
    return INVGchgnew


def SgINVg_calculator(Sgframe, INVGframe, save_path='../output/45_SGINVG.csv'):
    '''
    计算营业收入增长率和存货增长率之间的差
    :param Sgframe: 营业收入增长率frame
    :param INVGframe: 存货增长率
    :return:
    '''
    SGINVGframe = Sgframe - INVGframe
    SGINVGframe.to_csv(save_path)
    return SGINVGframe


def TAXchg_calculator(balancesheet, balanceid, stockid, dateid, save_path='../output/46_TAXchg.csv'):
    '''
    计算TAXchg-grows-ratio, 税收增长率
    :param balancesheet: 利润表
    :param balanceid: 税收所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    TAXchgnew = growthbath(balancesheet, balanceid, stockid, dateid, save_path)
    return TAXchgnew


def acc_calculator(balancesheet, profitsheet, cashflowsheet, profitid, cashflowid, assetid, dateid, stockid,
                   save_path='../output/47_ACC.csv'):
    '''
    计算acc
    :param balancesheet: 资产负债表
    :param profitsheet: 利润表
    :param cashflowsheet: 现金流量表
    :param profitid: 利润id
    :param cashflowid: 现金流id
    :param assetid: 资产id
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path:存储路径
    :return:
    '''
    profitframe = pd.DataFrame(index=dateid, columns=stockid)
    flowframe = pd.DataFrame(index=dateid, columns=stockid)
    Assetframe = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        profitStock = profitsheet[profitsheet['Stkcd'] == i]
        cashflowStock = cashflowsheet[cashflowsheet['Stkcd'] == i]
        profitframe.loc[list(profitStock.iloc[:, -1]), i] = list(profitStock.iloc[:, profitid])
        Assetframe.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, assetid])
        flowframe.loc[list(cashflowStock.iloc[:, -1]), i] = list(cashflowStock.iloc[:, cashflowid])
        print('stock ' + str(i) + ' finished')
    fillsheet(profitframe)
    fillsheet(flowframe)
    fillsheet(Assetframe)
    Assetframe.apply(lambda x: cal12(x))
    accural = profitframe - flowframe
    accframe = accural / Assetframe.apply(lambda x: checkdivision(x))
    accframe.to_csv(save_path)
    print('acc finished!!')
    return accframe


def abacc_calculator(accframe, save_path='../output/48_abacc.csv'):
    '''
    计算增值因子绝对值
    :param accframe:增值因子数据框
    :param save_path:保存路径
    :return:
    '''
    abacc = abs(accframe)
    abacc.to_csv(save_path)
    return abacc


def stdacc_calculator(accframe, save_path='../output/49_stdacc.csv'):
    '''
    计算增值因子标准差
    :param accframe:增值因子数据框
    :param save_path:保存路径
    :return:
    '''

    def accstdcal(x):
        uselist = [np.nan] * 47 + [np.nanstd(x.iloc[i - 47: i + 1]) for i in range(47, len(x))]
        x = uselist
        return x

    accframe = accframe.apply(lambda x: checkdivision(x))
    stdacc = accframe.apply(lambda x: accstdcal(x))
    stdacc.to_csv(save_path)
    return stdacc


def accp_calculator(profitsheet, cashflowsheet, profitid, cashflowid, netid, dateid, stockid,
                    save_path='../output/50_ACCP.csv'):
    '''
    计算accp
    :param profitsheet: 利润表
    :param cashflowsheet: 现金流量表
    :param profitid: 利润id
    :param cashflowid: 现金流id
    :param netid: 净利润id
    :param save_path:存储路径
    :return:
    '''
    profitframe = pd.DataFrame(index=dateid, columns=stockid)
    flowframe = pd.DataFrame(index=dateid, columns=stockid)
    netframe = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        profitStock = profitsheet[profitsheet['Stkcd'] == i]
        cashflowStock = cashflowsheet[cashflowsheet['Stkcd'] == i]
        profitframe.loc[list(profitStock.iloc[:, -1]), i] = list(profitStock.iloc[:, profitid])
        netframe.loc[list(profitStock.iloc[:, -1]), i] = list(profitStock.iloc[:, netid])
        flowframe.loc[list(cashflowStock.iloc[:, -1]), i] = list(cashflowStock.iloc[:, cashflowid])
        print('stock ' + str(i) + ' finished')
    fillsheet(profitframe)
    fillsheet(flowframe)
    fillsheet(netframe)
    accural = profitframe - flowframe
    netframe = netframe.apply(lambda x: checkdivision(x))
    accpframe = accural / netframe
    accpframe.to_csv(save_path)
    print('accp finished!!')
    return accpframe


def cinvest_calculator(balancesheet, profitsheet, balanceid, returnid, stockid, dateid,
                       save_path='../output/51_cinvest.csv'):
    '''
    计算资本投资
    :param balancesheet: 资产负债表
    :param profitsheet: 利润表
    :param balanceid: 固定资产净值
    :param returnid: 营业总收入id
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    frame1 = pd.DataFrame(index=dateid, columns=stockid)
    frame2 = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        profitStock = profitsheet[profitsheet['Stkcd'] == i]
        frame1.loc[list(balanceStock.iloc[:, -1]), i] = list(
            balanceStock.iloc[:, balanceid])  # pandas有自己的nan判断标准，可以不用管直接除
        frame2.loc[list(profitStock.iloc[:, -1]), i] = list(profitStock.iloc[:, returnid])  # pandas有自己的nan判断标准，可以不用管直接除
        print('stock ' + str(i) + ' finished')

    def cinmeancal(x):
        uselist = [np.nan] * 8 + [x.iloc[i - 8: i + 1].mean() for i in range(8, len(x))]
        x = uselist
        return x

    fillsheet(frame1)
    fillsheet(frame2)
    cinvestframe = frame1 / frame2.apply(lambda x: checkdivision(x))
    cinvestnew = cinvestframe.apply(lambda x: cinmeancal(x))
    cinvestnew.to_csv(save_path)
    return cinvestnew


def depr_calculator(balancesheet, profitsheet, balanceid, returnid, stockid, dateid, save_path='../output/52_depr.csv'):
    '''
    计算资本投资
    :param balancesheet: 资产负债表
    :param profitid: 专题研究表
    :param balanceid: 固定资产净值
    :param returnid: 固定资产折旧
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    deprframe = pd.DataFrame(index=dateid, columns=stockid)
    fixedframe = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        profitStock = profitsheet[profitsheet['Stkcd'] == i]
        deprframe.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid])
        fixedframe.loc[list(profitStock.iloc[:, -1]), i] = list(profitStock.iloc[:, returnid])
        print('stock ' + str(i) + ' finished')
    fillsheet(deprframe)
    fillsheet(fixedframe)
    fixedframe = fixedframe.apply(lambda x: checkdivision(x))
    deprrframe = deprframe / fixedframe
    deprrframe.to_csv(save_path)
    return deprrframe


def pchdepr_calculator(deprframe, save_path='../output/53_pchdepr.csv'):
    '''
    计算增值因子绝对值
    :param deprframe:增值因子数据框
    :param save_path:保存路径
    :return:
    '''
    pchdepr = deprframe / deprframe.shift(1).apply(lambda x: checkdivision(x)) - 1
    pchdepr.to_csv(save_path)
    return pchdepr


def egr_calculator(balancesheet, balanceid, stockid, dateid, save_path='../output/54_egr.csv'):
    '''
    计算egr-grows-ratio, 股东权益增长率
    :param balancesheet: 资产负债表
    :param balanceid: 股东权益所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    egrnew = growthbath(balancesheet, balanceid, stockid, dateid, save_path)
    return egrnew

def calprereturn(analyst, analystid, stockid, dateid):
    '''
    整理eps
    :param analyst: 分析师文件
    :param analystid: 分析师预测净利润id
    :param stockid: 股票id
    :param dateid: 日期id
    :return:
    '''
    tempframe = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        anaStock = analyst[analyst['Stkcd'] == i]
        anadate = list(set(anaStock['Rptdt']))
        acumnumber = []
        for x in anadate:
            acumnumber.append(anaStock[anaStock['Rptdt'] == x].iloc[:, analystid].mean())
        anadate = [x[:4] + '-' + x[5:7] for x in anadate]
        tempframe.loc[anadate, i] = acumnumber
        print('stock ' + str(i) + ' finished')
    fillsheet(tempframe)
    return tempframe

def fgr5yr_calculator(analyst, analystid, stockid, dateid, save_path='../output/55_fgr5yr.csv'):
    '''
       计算预期EPS增长率
       :param analyst: 分析师文件
       :param analystid: 分析师预测收益EPS
       :param stockid: 股票id
       :param dateid: 日期id
       :param save_path: 存储路径
       :return:
       '''
    fgr5 = calprereturn(analyst, analystid, stockid, dateid) # EPS
    fgr5yr = (fgr5 - fgr5.shift(12)) / fgr5.shift(12).apply(lambda x: checkdivision(x))
    fgr5yr.to_csv(save_path)
    return fgr5yr


def grCAPX_calculator(balancesheet, balanceid, stockid, dateid, save_path='../output/56_grCAPX.csv'):
    '''
    计算grCAPX-grows-ratio, 资本支出增长率
    :param balancesheet: 资产负债表
    :param balanceid: 资本支出所在位置（用固定资产净额代替）
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    grCAPXnew = growthbath(balancesheet, balanceid, stockid, dateid, save_path)
    return grCAPXnew


def pchcapx_ia_calculator(grCAPX, type, save_path='../output/57_pchcapx_ia.csv'):
    '''
      计算行业调整pchcapx
      :param pchcapx: pchcapxframe
      :param type: 类型按照第一列股票id第二列类型
      :param save_path: 存储数据
      :return:
      '''
    return size_ia_calculator(grCAPX, type, save_path)


def grltnoa(noaframe, save_path='../output/58_grltnoa.csv'):
    '''
    计算noa增长率，这个需要先计算noa才行
    :param noaframe:noa表格
    :param save_path: 保存路径
    :return:
    '''
    grltfram = noaframe / noaframe.shift(12).apply(lambda x: checkdivision(x)) - 1
    grltfram.to_csv(save_path)
    return grltfram


def invest_calculator(CAPX, INVchg, save_path='../output/59_invest.csv'):
    '''
    计算投资变化
    :param CAPX:资本支出变化
    :param INVchg:存货变化
    :param save_path:存储路径
    :return:
    '''
    invest = CAPX + INVchg
    invest.to_csv(save_path)
    return invest


def pchsale_pchinvt_calculator(SGr, INVchg, save_path='../output/60_pchsale_pchinvt.csv'):
    '''
    计算投资变化
    :param SGr:营业总收入变化
    :param INVchg:存货变化
    :param save_path:存储路径
    :return:
    '''
    pchsale_pchinvt = SGr + INVchg
    pchsale_pchinvt.to_csv(save_path)
    return pchsale_pchinvt


def pchsale_pchrect_calculator(SGr, balancesheet, balanceid, stockid, dateid,
                               save_path='../output/61_pchsale_pchrect.csv'):
    '''
    计算
    :param SGr:营业总收入变化率
    :param balancesheet: 资产负债表
    :param balanceid: 应收账款id
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 保存路径
    :return:
    '''
    yszk = growthbath(balancesheet, balanceid, stockid, dateid, '../temp.csv')
    pchsale_pchrect = SGr - yszk
    pchsale_pchrect.to_csv(save_path)
    return pchsale_pchrect


def pchsale_pchxsga_calculator(SGr, balancesheet, balanceid, stockid, dateid,
                               save_path='../output/62_pchsale_pchxsga.csv'):
    '''
    计算营收变化-三费变化
    :param SGr:营业总收入变化率
    :param balancesheet: 利润表
    :param balanceid: 三费id[销售，管理，财务]格式
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 保存路径
    :return:
    '''
    frame1 = pd.DataFrame(index=dateid, columns=stockid)
    frame2 = pd.DataFrame(index=dateid, columns=stockid)
    frame3 = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        frame1.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid[0]])
        frame2.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid[1]])
        frame3.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid[2]])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    fillsheet(frame3)
    newframe = frame1 + frame2 + frame3
    newframechange = newframe / newframe.shift(12).apply(lambda x: checkdivision(x)) - 1
    pchsale_pchxsga = SGr - newframechange
    pchsale_pchxsga.to_csv(save_path)
    return pchsale_pchxsga


def realestate_calculator(balancesheet, balanceid, stockid, dateid, save_path='../output/63_realestate.csv'):
    '''
    计算不动产，
    :param balancesheet:表格
    :param balanceid:固定资产总额id
    :param stockid:股票id
    :param dateid:日期id
    :param save_path:存储路径
    :return:
    '''
    realframe = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        realframe.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid])
        print('stock ' + str(i) + ' finished')
    fillsheet(realframe)
    realframe.to_csv(save_path)
    return realframe


def sgr_calculator(balancesheet, balanceid, stockid, dateid, save_path='../output/64_sgr.csv'):
    '''
    计算营业总收入增长率
    :param balancesheet: 利润表
    :param balanceid: 营业总收入所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    sgrnew = growthbath(balancesheet, balanceid, stockid, dateid, save_path)
    return sgrnew


def NOA_calculator(balancesheet, balanceid, debetid, stockid, dateid, save_path='../output/65_NOA.csv'):
    '''
    计算净经营资产, 虽然蠢
    :param balancesheet:资产负债表
    :param balanceid: [总资产, 货币资金, 短期投资额]
    :param debetid:[短期借款，长期借款，归属于母公司权益，少数股东权益]
    :param stockid:股票id
    :param dateid:日期id
    :param save_path:存储路径
    :return:
    '''
    frame1 = pd.DataFrame(index=dateid, columns=stockid)
    frame2 = pd.DataFrame(index=dateid, columns=stockid)
    frame3 = pd.DataFrame(index=dateid, columns=stockid)
    frame4 = pd.DataFrame(index=dateid, columns=stockid)
    frame5 = pd.DataFrame(index=dateid, columns=stockid)
    frame6 = pd.DataFrame(index=dateid, columns=stockid)
    frame7 = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        frame1.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid[0]])
        frame2.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid[1]])
        frame3.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid[2]])
        frame4.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, debetid[0]])
        frame5.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, debetid[1]])
        frame6.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, debetid[2]])
        frame7.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, debetid[3]])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2, True)
    fillsheet(frame3, True)
    fillsheet(frame4, True)
    fillsheet(frame5, True)
    fillsheet(frame6, True)
    fillsheet(frame7, True)
    noaframe = (frame1 - frame2 - frame3) - (frame1 - frame4 - frame5 - frame6 - frame7)
    noaframe2 = noaframe / frame1.apply(lambda x: checkdivision(x)) - 1
    noaframe2.to_csv(save_path)
    return [noaframe, noaframe2]


# 记得把员工数据格式改成yyyy-mm
def hire_calculator(hiresheet, hireid, stockid, dateid, save_path='../output/66_hire.csv'):
    '''
    计算员工数
    :param hire: 表格
    :param hireid: 员工id
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path:
    :return:
    '''
    hire = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        hireStock = hiresheet[hiresheet['Stkcd'] == i]
        hire.loc[hireStock.iloc[:, -1], i] = list(hireStock.iloc[:, hireid])
        print('stock ' + str(i) + 'finished')
    fillsheet(hire)
    hire.to_csv(save_path)
    return hire


def chempia_calculator(hire, type, save_path='../output/67_chempia.csv'):
    '''
      计算行业调整hire
      :param hire: hireframe
      :param type: 类型按照第一列股票id第二列类型
      :param save_path: 存储数据
      :return:
      '''
    return size_ia_calculator(hire, type, save_path)


def RD_calculator(balancesheet, balanceid, stockid, dateid, save_path='../output/68_RD.csv'):
    '''
    计算研发支出
    :param balancesheet: 利润表
    :param balanceid: 管理费用id
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path:
    :return:
    '''
    RD = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        RDStock = balancesheet[balancesheet['Stkcd'] == i]
        RD.loc[list(RDStock.iloc[:, -1]), i] = list(RDStock.iloc[:, balanceid])
    fillsheet(RD)
    RD.to_csv(save_path)
    return RD


def RD_mve_calculator(RD, size, save_path='../output/69_RD_mve.csv'):
    '''
    计算研发支出/市值
    :param RD: 研发支出
    :param size: 市值
    :param save_path:
    :return:
    '''
    RD_mve = RD / size
    RD_mve.to_csv(save_path)
    return RD_mve


def RDsale_calculator(RD, balancesheet, balanceid, stockid, dateid, save_path='../output/70_RDsale.csv'):
    '''
    计算研发支出/营业收入
    :param RD: 研发支出
    :param balancesheet: 利润表
    :param balanceid: 营业收入id
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    S = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        SStock = balancesheet[balancesheet['Stkcd'] == i]
        S.loc[list(SStock.iloc[:, -1]), i] = list(SStock.iloc[:, balanceid])
    fillsheet(S)
    RDsale = RD / S.apply(lambda x: checkdivision(x))
    RDsale.to_csv(save_path)
    return RDsale


## 其实可以提取公共代码的但是我实在是提取不动了
def ROE_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid, save_path='../output/71_ROE.csv'):
    '''
    计算ROE
    :param profit: 利润表
    :param profitid: 净利润id
    :param balancesheet: 资产负债表
    :param balanceid: 所有者权益合计
    :param stockid:股票id
    :param dateid:日期id
    :param save_path:存储路径
    :return:
    '''
    frame1 = pd.DataFrame(index=dateid, columns=stockid)
    frame2 = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        profitStock = profit[profit['Stkcd'] == i]
        frame1.loc[list(profitStock.iloc[:, -1]), i] = list(profitStock.iloc[:, profitid])
        frame2.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    ROEframe = frame1 / frame2.apply(lambda x: checkdivision(x))
    ROEframe.to_csv(save_path)
    return ROEframe


def ROA_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid, save_path='../output/72_ROA.csv'):
    '''
    计算ROE
    :param profit: 利润表
    :param profitid: 净利润id
    :param balancesheet: 资产负债表
    :param balanceid: 总资产
    :param stockid:股票id
    :param dateid:日期id
    :param save_path:存储路径
    :return:
    '''
    return ROE_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid, save_path)


def CT_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid, save_path='../output/73_CT.csv',
                  shift=True):
    '''
    计算CT， 以及用于相同类因子的计算
    :param profit: 利润表
    :param profitid: 营业总收入id
    :param balancesheet:资产负债表
    :param balanceid:总资产id
    :param stockid:股票id
    :param dateid:日期id
    :param save_path:存储路径
    :return:
    '''
    frame1 = pd.DataFrame(index=dateid, columns=stockid)
    frame2 = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        profitStock = profit[profit['Stkcd'] == i]
        frame1.loc[list(profitStock.iloc[:, -1]), i] = list(profitStock.iloc[:, profitid])
        frame2.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    if shift:
        CTframe = frame1 / frame2.shift(12).apply(lambda x: checkdivision(x))
    else:
        CTframe = frame1 / frame2.apply(lambda x: checkdivision(x))
    CTframe.to_csv(save_path)
    return CTframe


def PA_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid, save_path='../output/74_PT.csv'):
    '''
    计算PA
    :param profit: 利润表
    :param profitid: 总利润id
    :param balancesheet:资产负债表
    :param balanceid:总资产id
    :param stockid:股票id
    :param dateid:日期id
    :param save_path:存储路径
    :return:
    '''
    return CT_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid, save_path)


def cashpr_calculator(size, balancesheet, balanceid, stockid, dateid, save_path='../output/75_cashpr.csv'):
    '''
    计算现金生厂力
    :param size:A股流通市值
    :param balancesheet:资产负债表
    :param balanceid:[长期负债，总资产，货币资金]
    :param stockid:股票id
    :param dateid:日期id
    :param save_path:存储路径
    :return:
    '''
    frame1 = pd.DataFrame(index=dateid, columns=stockid)
    frame2 = pd.DataFrame(index=dateid, columns=stockid)
    frame3 = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        frame1.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid[0]])
        frame2.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid[1]])
        frame3.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid[2]])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    fillsheet(frame3)
    cashpr = (size + frame1 - frame2) / frame3.apply(lambda x: checkdivision(x))
    cashpr.to_csv(save_path)
    return cashpr


def cash_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid, save_path='../output/76_cash.csv'):
    '''
    计算cash
    :param profit: 现金流量表
    :param profitid: 现金及现金等价物id
    :param balancesheet:资产负债表
    :param balanceid:总资产id
    :param stockid:股票id
    :param dateid:日期id
    :param save_path:存储路径
    :return:
    '''
    return CT_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid, save_path, False)


# 为了以后还要计算个股本的frame供以后用
def operprof_calculator(share, balancesheet, balanceid, stockid, dateid, save_path='../output/77_operprof.csv'):
    '''
    计算营业利润率
    :param share: 股本
    :param balancesheet:利润表
    :param balanceid: 营业利润位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    operprofframe = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        operprofframe.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid])
        print('stock ' + str(i) + ' finished')
    fillsheet(operprofframe)
    operprofframeuse = operprofframe / share.apply(lambda x: checkdivision(x))
    operprofframeuse.to_csv(save_path)
    return operprofframeuse


def pchgm_pchsale_calculator(sg, profit, profitid, stockid, dateid, save_path='../output/78_pchgm_pchsale.csv'):
    '''
    毛利变化率-营收变化率
    :param profit:利润表
    :param profitid:[营业收入id, 营业成本id]
    :param sg:营收变化率
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path:存储路径
    :return:
    '''
    frame1 = pd.DataFrame(index=dateid, columns=stockid)
    frame2 = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = profit[profit['Stkcd'] == i]
        frame1.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, profitid[0]])
        frame2.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, profitid[1]])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    mpro = frame1 - frame2

    def mprochange(x):
        uselist = [np.nan] * 23 + [(x.iloc[i] - x.iloc[i - 23: i - 11].mean() )/ x.iloc[i - 23: i - 11].mean()
                                   for i in range(23, len(x))]
        x = uselist
        return x

    mprochg = mpro.apply(lambda x: mprochange(x))
    pchgm_pchsale = mprochg - sg
    pchgm_pchsale.to_csv(save_path)
    return pchgm_pchsale


def ATO_calculator(NOA, profit, profitid, stockid, dateid, save_path='../output/79_ATO.csv'):
    '''
    计算总资产周转率
    :param NOA:
    :param profit:利润表
    :param profitid: 营业收入id
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path:存储路径
    :return:
    '''
    frame1 = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = profit[profit['Stkcd'] == i]
        frame1.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, profitid])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    ATO = frame1 / NOA.apply(lambda x: checkdivision(x))
    ATO.to_csv(save_path)
    return ATO


# 这里需要预期earnings frame
def chfeps_calculator(predictef, profit, profitid, stockid, dateid, save_path='../output/80_cheps.csv'):
    '''
    计算eps变化
    :param predictef:预测EPSframe
    :param profit: 利润表
    :param profitid: 基本每股收益id
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    frame1 = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = profit[profit['Stkcd'] == i]
        frame1.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, profitid])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    epschg = frame1 - predictef
    epschg.to_csv(save_path)
    return epschg


def nincr_calculator(profit, profitid, stockid, dateid, save_path='../output/81_nincr.csv'):
    '''
    计算净利润增加期数，最大为8
    :param profit: 利润表
    :param profitid: 净利润id
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    frame1 = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = profit[profit['Stkcd'] == i]
        frame1.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, profitid])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    nin = frame1 - frame1.shift(1)

    def calagrenum(x):  # 估计慢死，但我懒得改了
        rsultlist = [np.nan] * 11
        for i in range(11, len(x)):
            countnum = 0
            bascount = list(x.iloc[i - 11: i + 1])[::-1]
            index = np.where(np.array(bascount) > 0)[0]
            try:
                if index[0] != 0:
                    rsultlist.append(countnum)
                    continue
                temp = 0
                for j in index:
                    if j - temp > 1:
                        break
                    temp = j
                    countnum += 1
            except:
                pass
            rsultlist.append(min(8, countnum))
        return rsultlist


    nincr = nin.apply(lambda x: calagrenum(x))
    nincr.to_csv(save_path)
    return nincr


def roic_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid, save_path='../output/82_roic.csv'):
    '''
    投入资本回报
    :param profit:利润表
    :param profitid:营业收入id
    :param balancesheet:资产负债表
    :param balanceid:[负债，所有者权益，货币资金id】
    :param stockid:股票id
    :param dateid:日期id
    :param save_path:存储路径
    :return:
    '''
    frame1 = pd.DataFrame(index=dateid, columns=stockid)
    frame2 = pd.DataFrame(index=dateid, columns=stockid)
    frame3 = pd.DataFrame(index=dateid, columns=stockid)
    frame4 = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        profitStock = profit[profit['Stkcd'] == i]
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        frame1.loc[profitStock.iloc[:, -1], i] = list(profitStock.iloc[:, profitid])
        frame2.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid[0]])
        frame3.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid[1]])
        frame4.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid[2]])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    fillsheet(frame3)
    fillsheet(frame4)
    roic = frame1 / (frame2 + frame3 - frame4).apply(lambda x: checkdivision(x))
    roic.to_csv(save_path)
    return roic


def rusp_calculator(size, profit, profitid, stockid, dateid, save_path='../output/83_rusp.csv'):
    '''
    计算eps变化
    :param size:市值
    :param profit: 利润表
    :param profitid: 营业收入id
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    frame1 = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = profit[profit['Stkcd'] == i]
        frame1.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, profitid])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    sc = frame1 - frame1.shift(4)
    rusp = sc / size
    rusp.to_csv(save_path)
    return rusp


# 这里需要分析师预测利润均值
def sfe_calculator(price, prereturn, save_path='../output/84_sfe.csv'):
    '''
    计算收益预测
    :param price:股价
    :param prereturn: 预测利润
    :param save_path: 存储路径
    :return:
    '''
    sfe = prereturn / price.apply(lambda x: checkdivision(x))
    sfe.to_csv(save_path)
    return sfe


def CR_calculator(balancesheet, balanceid, stockid, dateid, save_path='../output/85_CR.csv'):
    '''
    计算CR以及用于后续同类型计算
    :param balancesheet: 资产负债表
    :param balanceid: [流动资产id, 流动负债id]
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 保存路径
    :return: 
    '''
    frame1 = pd.DataFrame(index=dateid, columns=stockid)
    frame2 = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        frame1.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid[0]])
        frame2.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid[1]])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    CR = frame1 / frame2.apply(lambda x: checkdivision(x))
    CR.to_csv(save_path)
    return CR


def QR_calculator(balancesheet, balanceid, stockid, dateid, save_path='../output/86_QR.csv'):
    '''
    计算QR
    :param balancesheet: 资产负债表
    :param balanceid: [流动资产id, 存货净额id]
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 保存路径
    :return:
    '''
    return CR_calculator(balancesheet, balanceid, stockid, dateid, save_path)


def CFdebt_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid, save_path='../output/87_CFdebt.csv'):
    '''
    计算现金流负债比
    :param profit:利润表
    :param profitid: 净利润id
    :param balancesheet: 资产负债表
    :param balanceid: 总负债id
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 保存路径
    :return:
    '''
    frame1 = pd.DataFrame(index=dateid, columns=stockid)
    frame2 = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        profitStock = profit[profit['Stkcd'] == i]
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        frame1.loc[list(profitStock.iloc[:, -1]), i] = list(profitStock.iloc[:, profitid])
        frame2.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    frame2plus = frame2.apply(lambda x: cal12(x))
    CFdebt = frame1 / frame2plus.apply(lambda x: checkdivision(x))
    CFdebt.to_csv(save_path)
    return CFdebt


def salecash_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid,
                        save_path='../output/88_salecash.csv'):
    '''
    计算营业收入现金比
    :param profit:利润表
    :param profitid: 营业收入id
    :param balancesheet: 资产负债表
    :param balanceid: 货币资金id
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 保存路径
    :return:
    '''
    frame1 = pd.DataFrame(index=dateid, columns=stockid)
    frame2 = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        profitStock = profit[profit['Stkcd'] == i]
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        frame1.loc[list(profitStock.iloc[:, -1]), i] = list(profitStock.iloc[:, profitid])
        frame2.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    salecash = frame1 / frame2.apply(lambda x: checkdivision(x))
    salecash.to_csv(save_path)
    return salecash


def saleinv_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid,
                       save_path='../output/89_saleinv.csv'):
    '''
    计算营业收入存货比
    :param profit:利润表
    :param profitid: 营业收入id
    :param balancesheet: 资产负债表
    :param balanceid: 存货id
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 保存路径
    :return:
    '''
    return salecash_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid, save_path)


def CRG_calculator(CR, save_path='../output/90_CRG.csv'):
    '''
    计算流动比率增长率 和以后同类型计算
    :param CR:流动比率
    :param save_path:存储路径
    :return:
    '''
    CRG = (CR - CR.shift(12)) / CR.shift(12).apply(lambda x: checkdivision(x))
    CRG.to_csv(save_path)
    return CRG


def QRG_calculator(QR, save_path='../output/91_QRG.csv'):
    '''
    计算速动比率增长率
    :param QR:速动比率
    :param save_path:存储路径
    :return:
    '''
    return CRG_calculator(QR, save_path)


def pchsaleinv_calculator(saleinv, save_path='../output/92_pchsaleinv.csv'):
    '''
    营业收入存货比增长率
    :param saleinv: 营业收入存货比
    :param save_path: 存储路径
    :return:
    '''
    return CRG_calculator(saleinv, save_path)


def salerec_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid,
                       save_path='../output/93_salerec.csv'):
    '''
    计算营业收入应收账款比
    :param profit:利润表
    :param profitid: 营业收入id
    :param balancesheet: 资产负债表
    :param balanceid: 应收账款id
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 保存路径
    :return:
    '''
    return salecash_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid, save_path)


def tang_calculator(balancesheet, balanceid, stockid, dateid, save_path='../output/94_tang.csv'):
    '''
    计算偿债能力比
    :param balancesheet: 资产负债表
    :param balanceid: [货币资金，应收账款，存货，不动产，总资产id]
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    frame1 = pd.DataFrame(index=dateid, columns=stockid)
    frame2 = pd.DataFrame(index=dateid, columns=stockid)
    frame3 = pd.DataFrame(index=dateid, columns=stockid)
    frame4 = pd.DataFrame(index=dateid, columns=stockid)
    frame5 = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        frame1.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid[0]])
        frame2.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid[1]])
        frame3.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid[2]])
        frame4.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid[3]])
        frame5.loc[list(balanceStock.iloc[:, -1]), i] = list(balanceStock.iloc[:, balanceid[4]])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    fillsheet(frame3)
    fillsheet(frame4)
    fillsheet(frame5)
    tang = (frame1 + 0.715 * frame2 + 0.547 * frame3 + 0.535 * frame4) / frame5.apply(lambda x: checkdivision(x))
    tang.to_csv(save_path)
    return tang


def chnanlyst_calculator(nanlyst, save_path='../output/95_chnanlyst.csv'):
    '''
    计算分析师变化人数
    :param nanlyst:分析师人数
    :param save_path:路径
    :return:
    '''
    chnanlyst = nanlyst - nanlyst.shift(3)
    chnanlyst.to_csv(save_path)
    return chnanlyst


def nanlyst_calculator(analyst, analystid, stockid, dateid, save_path='../output/96_nanlyst.csv'):
    '''
    计算每月分析师人数
    :param analyst: 分析师文件
    :param analystid: 分析师人数累积
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    nanlyst = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        anaStock = analyst[analyst['Stkcd'] == i]
        anadate = list(set(anaStock['Rptdt']))
        acumnumber = []
        for x in anadate:
            acumnumber.append(anaStock[anaStock['Rptdt'] == x].iloc[:, analystid].sum())
        anadate = [x[:4] + '-' + x[5:7] for x in anadate]
        nanlyst.loc[anadate, i] = acumnumber
        print('stock ' + str(i) + ' finished')
    fillsheet(nanlyst)
    nanlyst.to_csv(save_path)
    return nanlyst


# %% reading data
# basedata
codesframe = pd.read_excel('../data/stockcode/allCodes.xls', header=None)
## 股票类型
industrialframe = pd.read_csv('../data/stockcode/TRD_Co.csv', encoding='gb2312')
industrialframe['type'] = list(map(lambda x: x[0], industrialframe['Nnindcd']))
typeframe = industrialframe.iloc[:, [0, -1]]
indexDaily = pd.read_csv('../data/dailydata/index/TRD_Cndalym.csv')
AshareindexDaily = indexDaily[indexDaily['Markettype'] == 5]  # deq等权重 dos加权
AshareindexDailyuse = AshareindexDaily.iloc[:, 1:3]
AshareindexDailyuse.set_index('Trddt', inplace=True)

indexWeekly = pd.read_csv('../data/weeklydata/index/TRD_Weekcm.csv')
AshareindexWeekly = indexWeekly[indexWeekly['Markettype'] == 5]  # deq等权重 dos加权
mdda = list(AshareindexWeekly['Trdwnt'])
mwdda = list(map(formdate,mdda))
AshareindexWeekly['Trdwnt'] = mwdda
AshareindexWeekly = AshareindexWeekly.iloc[:, 1:3]
AshareindexWeekly.set_index('Trdwnt', inplace=True)

stockcodes = list(codesframe.iloc[:, 0])
typelist = set(industrialframe['type'])
rf = pd.read_csv('../data/rf/TRD_Nrrate.txt', sep='\t')

# analyst
analist = os.listdir('../data/analyst')
listfile = []
[listfile.append(pd.read_csv('../data/analyst/' + x, encoding='gb18030')) for x in analist if x[-3:] == 'csv']
analistfile = pd.concat(listfile)
analistfile = analistfile.sort_values('Stkcd')
analistfile = analistfile.reset_index(drop=True)
analistfile['number'] = [len(x.split(',')) if type(x) == str else 1 for x in analistfile['Ananm']]

# quarterlydata
annoucement = pd.read_csv('../data/quarterlydata/annoucement/IAR_Forecdt.csv')
balance = pd.read_csv('../data/quarterlydata/balancesheet/FS_Combas.csv')
cash = pd.read_csv('../data/quarterlydata/cash/FS_Comscfd.csv')
profit = pd.read_csv('../data/quarterlydata/profit/FS_Comins.csv')
balance = balance[balance['Typrep'] == 'A']
cash = cash[cash['Typrep'] == 'A']
profit = profit[profit['Typrep'] == 'A']
balancenew = changeSheetdate(balance, 1, annoucement)
cashnew = changeSheetdate(cash, 1, annoucement)
cashnew = cashnew.sort_values(['Stkcd', 'Accper'])
cashnew.reset_index(drop=True, inplace=True)
profitnew = changeSheetdate(profit, 1, annoucement)
profitnew = profitnew.sort_values(['Stkcd', 'Accper'])
profitnew.reset_index(drop=True, inplace=True)

balancenew = balancenew.sort_values(['Stkcd', 'Accper'])
balancenew.reset_index(drop=True, inplace=True)

ZTSJ = pd.read_csv('../data/quarterlydata/depreciation/MNMAPR_Accruals.csv')
ZTSJnew = changeSheetdate(ZTSJ, 1, annoucement)
ZTSJnew = ZTSJnew.sort_values(['Stkcd', 'Accper'])
ZTSJnew.reset_index(drop=True, inplace=True)
ZTSJnew.to_csv('../data/quarterlydata/depreciation/MNMAPR_AccrualsNew.csv')
ZTSJ = ZTSJnew
# 二次引用
balancenew.to_csv('../data/quarterlydata/balancesheet/FS_CombasNew.csv')
cashnew.to_csv('../data/quarterlydata/cash/FS_ComscfdNew.csv')
profitnew.to_csv('../data/quarterlydata/profit/FS_CominsNew.csv')

balance = pd.read_csv('../data/quarterlydata/balancesheet/FS_CombasNew.csv')
cash = pd.read_csv('../data/quarterlydata/cash/FS_ComscfdNew.csv')
profit = pd.read_csv('../data/quarterlydata/profit/FS_CominsNew.csv')
ZTSJ = pd.read_csv('../data/quarterlydata/depreciation/MNMAPR_AccrualsNew.csv')
balance = checkstep2(balance)
profit = checkstep2(profit)
cash = checkstep2(cash)
ZTSJ = checkstep2(ZTSJ)
balance.iloc[:, -1] = [changedateExpression(x, False) for x in balance.iloc[:, -1]]
profit.iloc[:, -1] = [changedateExpression(x, False) for x in profit.iloc[:, -1]]
cash.iloc[:, -1] = [changedateExpression(x, False) for x in cash.iloc[:, -1]]
ZTSJ.iloc[:, -1] = [changedateExpression(x, False) for x in ZTSJ.iloc[:, -1]]
annoucement['month'] = [annoucement['Actudt'].iloc[t][:-3] if type(t) == str else annoucement['Accper'].iloc[t][:-3] for
                        t in range(len(list(annoucement['Actudt'])))]
del balance['Unnamed: 0']
del profit['Unnamed: 0']
del cash['Unnamed: 0']
del ZTSJ['Unnamed: 0']
TAXframe = profit.iloc[:, [0, 1, 8, 14]]
TAXframe['tax'] = [combinetax(TAXframe.iloc[i, -2], TAXframe.iloc[i, -1]) for i in range(TAXframe.shape[0])]
TAXframe['Actudt'] = profit['Actudt']

# monthlydata
# monthlydata = pd.read_csv('../data/monthlydata/stock/TRD_Mnth.csv')
monthlydata = pd.read_csv('../data/monthlydata/stock/TRD_Mnth.txt', sep='\t')
monthdate = pd.read_csv('../data/monthlydata/date.txt', sep='\t')
monthlydata['share'] = monthlydata['Msmvosd'] * 1000 / monthlydata['Mclsprc']
date = monthdate[monthdate['Markettype'] == 1]
date = list(date['Trdmnt'])
# weeklydata
weeklydata = readingwhole_Data('../data/weeklydata/stock/', dayind='Trdwnt')
wdda = list(weeklydata['Trdwnt'])
rwdda = list(map(formdate, wdda))
weeklydata['Trdwnt'] = rwdda
weeklydateid = sorted(set(weeklydata['Trdwnt']))
weeklyreturn = pd.DataFrame(columns=stockcodes, index=weeklydateid)
fillIntheBlank(weeklydata, weeklyreturn, stockcodes, datanum=3)
# dailydata
dailydata = readingwhole_Data('../data/dailydata/stock/', dayind='Trddt')
dailydate = sorted(list(set(dailydata['Trddt'])))
dailydata['changerate'] = dailydata['Dnshrtrd'] * dailydata['Clsprc'] / (dailydata['Dsmvosd'] * 1000)
dailydata['share'] = dailydata['Dsmvosd'] * 1000 / dailydata['Clsprc']
dailydata['illq'] = np.abs(dailydata['Dretwd']) / dailydata['Dnvaltrd'] * 1000000

rfnewdate = [changedateExpression(x) for x in rf['Clsdt']]
rf['Clsdt'] = rfnewdate
rf.set_index('Clsdt', inplace=True)
rf_corresponding = rf.loc[dailydate]
rfuse = rf_corresponding.iloc[:, 1:3] / 100  # 因为数据是百分数

# %% calculating factors
# aimdataframe = pd.DataFrame(columns=stockcodes, index=dailydate)
# fillIntheBlank(dailydata,aimdataframe, stockcodes, datanum=-5)# datanum这里是收益率在的位置
# aimdataframe.to_csv('../data/aimdataframe.csv')
#
# changeratedataframe = pd.DataFrame(columns=stockcodes, index=dailydate)
# fillIntheBlank(dailydata, changeratedataframe, stockcodes, datanum=-3)# datanum这里是计算完换手率的数据
# changeratedataframe.to_csv('../data/turndata.csv')
#
# volumndataframe = pd.DataFrame(columns=stockcodes,index=dailydate)
# fillIntheBlank(dailydata, volumndataframe, stockcodes, datanum=4)# datanum这里是交易金额的数据
# volumndataframe.to_csv('../data/volumndata.csv')
#
# illiqdataframe = pd.DataFrame(columns=stockcodes,index=dailydate)
# fillIntheBlank(dailydata, illiqdataframe,stockcodes, datanum=-1)# datanum这里是异质性波动率数据id
# illiqdataframe.to_csv('../data/illiqdataframe.csv')


shareframedataframe = pd.DataFrame(columns=stockcodes, index=date)
fillIntheBlank(monthlydata, shareframedataframe, stockcodes, datanum=-1)  # 月度数据： datanum这里是股本数据id
shareframedataframe.to_csv('../data/sharedataframe.csv')

# 二次引用
aimdataframe = pd.read_csv('../data/aimdataframe.csv')
aimdataframe.set_index('Unnamed: 0', inplace=True)

changeratedataframe = pd.read_csv('../data/turndata.csv')
changeratedataframe.set_index('Unnamed: 0', inplace=True)

volumndataframe = pd.read_csv('../data/volumndata.csv')
volumndataframe.set_index('Unnamed: 0', inplace=True)

illqdataframe = pd.read_csv('../data/illiqdataframe.csv')
illqdataframe.set_index('Unnamed: 0', inplace=True)

shareframedataframe = pd.read_csv('../data/sharedataframe.csv')
shareframedataframe.set_index('Unnamed: 0', inplace=True)

monthtopindex = cal_monthend(dailydate)
AshareindexDailyuse = AshareindexDailyuse.loc[dailydate]
##
# monthlyreturn = pd.DataFrame(index=date, columns=stockcodes)
# fillIntheBlank(monthlydata, monthlyreturn, stockcodes, datanum=-2) # datanum是收益率数据
# monthlyreturn.to_csv('../output/monthlyreturn.csv')
# 二次引用
monthlyreturn = pd.read_csv('../output/monthlyreturn.csv')
mtdate = [changedateExpression(x, False) for x in monthlyreturn['Unnamed: 0']]
monthlyreturn['Unnamed: 0'] = mtdate
monthlyreturn.set_index('Unnamed: 0', inplace=True)

# 01 size
size = size_calculator(monthdata=monthlydata, sizeposition=5, stockid=stockcodes, dateid=date)
# 02 size_ia
size_ia = size_ia_calculator(size, typeframe)
# 03 beta
beta = beta_calculator(aimdataframe, AshareindexDailyuse, stockcodes, dailydate, monthtopindex, rfuse)
# 04_betasq
betasq = betasq_calculator(beta)
# 05 betaad
betad = betad_calculator(aimdataframe, AshareindexDailyuse, stockcodes, dailydate, monthtopindex, rfuse)
# 06 idvol
idvol = idvol_calculator(aimdataframe, AshareindexDailyuse, stockcodes, dailydate, monthtopindex, rfuse)
# 07 retvol
retvol = retvol_calculator(aimdataframe, stockcodes, dailydate, monthtopindex)
# 08 idskewness
idskewness = idskewness_calculator(aimdataframe, AshareindexDailyuse, stockcodes, dailydate, monthtopindex, rfuse)
# 09 skewness
skewness = skewness_calculator(aimdataframe, stockcodes, dailydate, monthtopindex)
# 10 coskewness
coskewness = coskewness_calculator(aimdataframe, AshareindexDailyuse, stockcodes, dailydate, monthtopindex, rfuse)
# 11 turn
turn = turn_calculator(changeratedataframe, stockcodes, monthtopindex, dailydate)
# 12 turnsd
turnsd = turnsd_calculator(changeratedataframe, stockcodes, monthtopindex, dailydate)
# 13 vold
vold = vold_calculator(volumndataframe, stockcodes, monthtopindex, dailydate)
# 14 stdvold
stdvold = stdvold_calculator(volumndataframe, stockcodes, monthtopindex, dailydate)
# 15 retmax
retmax = retmax_calculator(aimdataframe, stockcodes, dailydate, monthtopindex)
# 16 illq
illq = illq_calculator(illqdataframe, stockcodes, monthtopindex, dailydate)
# 17 LM
LM = LM_calculator(changeratedataframe, stockcodes, monthtopindex, dailydate)
# 18 sharechg
sharechg = sharechg_calculator(shareframedataframe, stockcodes, date)
# 19 age
age = age_calculator(industrialframe, monthlyreturn, stockcodes)
# 20 aeavol
announcedate = sorted(set(annoucement['month']))
eaframe = pd.DataFrame(columns=stockcodes, index=announcedate)
fillIntheBlank2(annoucement, eaframe, stockcodes, datanum=2)
aeavol = aeavol_calculator(volumndataframe, eaframe, stockcodes, announcedate, monthlyreturn)
# 21 pricedelay
weektopindex = cal_monthend(weeklydateid)
pricedelay = pricedelay_calculator(weeklyreturn, AshareindexWeekly, stockcodes, weeklydateid, weektopindex)
# 22 mom12
mom12 = mom12_calculator(aimdataframe, stockcodes, dailydate, monthtopindex)
# 23 mom6
mom6 = mom6_calculator(aimdataframe, stockcodes, dailydate, monthtopindex)
# 24 mom36
mom36 = mom36_calculator(aimdataframe, stockcodes, dailydate, monthtopindex)
# 25 momchg
momchg = momchg_calculator(aimdataframe, stockcodes, dailydate, monthtopindex)
# 26 imom
imom = imom_calculator(aimdataframe, AshareindexDailyuse, stockcodes, dailydate, monthtopindex, rfuse)
# 27largretn
largretn = largretn_calculator(monthdata=monthlydata, sizeposition=-1, stockid=stockcodes, dateid=date)
# 28BM
size = pd.read_csv('../output/01_size.csv')
size.set_index('Unnamed: 0', inplace=True)
size = size * 1000
BM = BM_calculator(balance, size, 18, stockcodes, date)
# 29 BM_ia
BM_ia = BM_ia_calculator(BM, typeframe)
# 30 AM
AM = AM_calculator(balance, size, 9, stockcodes, date)
# 31 LEV
LEV = LEV_calculator(balance, size, 15, stockcodes, date)
# 32 EP
EP = EP_calculator(profit, size, 15, stockcodes, date)
# 33 CFP
CFP = CFP_calculator(cash, size, 6, stockcodes, date)
# 34 CFP_ia
CFP_ia = CFP_ia_calculator(CFP, typeframe)
# 35 OCFP
OCFP = OCFP_calculator(cash, size, 3, stockcodes, date)
# 36 DP
DP = DP_calculator(balance, size, 11, stockcodes, date)
# 37 SP
SP = SP_calculator(profit, size, 4, stockcodes, date)
# 38 AG
AG = AG_calculator(balance, 9, stockcodes, date)
# 39 LG
LG = LG_calculator(balance, 15, stockcodes, date)
# 40 BVEG
BVEG = BVEG_calculator(balance, 18, stockcodes, date)
# 41 SG
SG = SG_calculator(profit, 4, stockcodes, date)
# 42 PMG
PMG = PMG_calculator(profit, 12, stockcodes, date)
# 43 INVG
INVG = INVG_calculator(balance, 6, stockcodes, date)
# 44 INVGchg
INVGchg = INVGchg_calculator(balance, 6, 9, stockcodes, date)
# 45 SGINVG
SgINVg = SgINVg_calculator(SG, INVG)
# 46 TAXchg
TAXchg = TAXchg_calculator(TAXframe, 4, stockcodes, date)
# 47 ACC
ACC = acc_calculator(balance, profit, cash, 13, 6, 9, date, stockcodes)
# 48 abacc
abacc = abacc_calculator(ACC)
# 49 stdacc
stdacc = stdacc_calculator(ACC)
# 50 ACCP
ACCP = accp_calculator(profit, cash, 13, 3, 15, date, stockcodes)  # 利润总额， 经营活动现金流， 净利润id
# 51 cinvest
cinvest = cinvest_calculator(balance, profit, 8, 3, stockcodes, date)
# 52 depr
depr = depr_calculator(balance, ZTSJ, 8, 2, stockcodes, date)
# 53 pchdepr
pchdepr = pchdepr_calculator(depr)
# 54 egr
egr = egr_calculator(balance, 18, stockcodes, date)
# 55 fgr5yr 记得转换日期格式
fgr5yr = fgr5yr_calculator(analistfile, 5, stockcodes, date)
# 56 grCAPX
grCAPX = grCAPX_calculator(balance, 8, stockcodes, date)
# 57 pchcapx_ia
pchcapx_ia = pchcapx_ia_calculator(grCAPX, typeframe)
# 65 NOA
[NOA, NOArate] = NOA_calculator(balance, [9, 3, 4], [10, 13, 16, 17], stockcodes, date)
# 58 grithnoa
grithnoa = grltnoa(NOA)
# 59 invest
invset = invest_calculator(grCAPX, INVGchg)
# 64 sgr
SGr = sgr_calculator(profit, 3, stockcodes, date)
# 60 pchsale_pchinvt
pchsale_pchinvt = pchsale_pchinvt_calculator(SGr, INVGchg)
# 61 pchsale_pchrect
pchsale_pchrect = pchsale_pchrect_calculator(SGr, balance, 5, stockcodes, date)
# 62 pchsale_pchxsga
pchsale_pchxsga = pchsale_pchxsga_calculator(SGr, profit, [9, 10, 11], stockcodes, date)
# 63 realstate
realestate = realestate_calculator(balance, 8, stockcodes, date)
# 66 hire
hireframe = readingwholetxt_Data('../data/employees/', dayind='Annodt', txtind='txt')
hireframe['month'] = [x[:-3] for x in hireframe['Annodt']]
hireframe = checkstep2(hireframe)
hire = hire_calculator(hireframe, 2, stockcodes, date)
# 67 chepmia
chepmia = chempia_calculator(hire, typeframe)
# 68 RD
RD = RD_calculator(profit, 10, stockcodes, date)
# 69 RD_MVE
RD_mve = RD_mve_calculator(RD, size)
# 70 RDsale
RDsale = RDsale_calculator(RD, profit, 4, stockcodes, date)
# 71 ROE
ROE = ROE_calculator(profit, 15, balance, 18, stockcodes, date)
# 72 ROA
ROA = ROA_calculator(profit, 15, balance, 9, stockcodes, date)
# 73 CT
CT = CT_calculator(profit, 3, balance, 9, stockcodes, date)
# 74 PA
PA = PA_calculator(profit, 13, balance, 9, stockcodes, date)
# 75 cashpr
cashpr = cashpr_calculator(size, balance, [14, 9, 3], stockcodes, date)
# 76 cash 这里需要现金及现金等价物期末额，靠！
cashf = cash_calculator(cash, 7, balance, 9, stockcodes, date)
# 77 operprof
share = size / monthlyreturn
operprof = operprof_calculator(share, profit, 12, stockcodes, date)
# 78 pchgm_pchsale
pchgm_pchsale = pchgm_pchsale_calculator(SG, profit, [4, 7], stockcodes, date)
# 79 ATO
ATO = ATO_calculator(NOA, profit, 4, stockcodes, date)
# 80 chfeps 需要EPS
anaprereturn = calprereturn(analistfile, 5, stockcodes, date)
chfeps = chfeps_calculator(anaprereturn, profit, 16, stockcodes, date)
# 81 nincr
nincr = nincr_calculator(profit, 15, stockcodes, date)
# 82 roic
roic = roic_calculator(profit, 4, balance, [15, 18, 3], stockcodes, date)
# 83 rusp
rusp = rusp_calculator(size, profit, 4, stockcodes, date)
# 84 sfe
sfe = sfe_calculator(monthlyreturn, anaprereturn)
# 85 CR
CR = CR_calculator(balance, [7, 12], stockcodes, date)
# 86 QR
QR = QR_calculator(balance, [7, 6], stockcodes, date)
# 87 CFdebt
CFdebt = CFdebt_calculator(profit, 15, balance, 15, stockcodes, date)
# 88 salecash
salecash = salecash_calculator(profit, 4, balance, 3, stockcodes, date)
# 89 saleinv
saleinv = saleinv_calculator(profit, 4, balance, 6, stockcodes, date)
# 90 CRG
CRG = CRG_calculator(CR)
# 91 QRG
QRG = QRG_calculator(QR)
# 92 pchsaleinv
pchsaleinv = pchsaleinv_calculator(saleinv)
# 93 salerec
salerec = saleinv_calculator(profit, 4, balance, 6, stockcodes, date)
# 94 tang
tang = tang_calculator(balance, [3, 5, 6, 8, 9], stockcodes, date)
# 96 nanlyst
nanlyst = nanlyst_calculator(analistfile, 7, stockcodes, date)
# 95 chnanlyst
chnanlyst = chnanlyst_calculator(nanlyst)

