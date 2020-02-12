# -*- coding:utf-8 -*-
'''
 @author: yueyang li
 @last edition:2020-02-12
'''
#%% base package
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from importlib import reload
import gc, re, warnings

#%% base function
def readingwhole_Data(path, encoding='gb18030', dayind=False):
    '''
    读取整个数据，并按照股票顺序排列
    :param path: 路径
    :param encoding: 编码
    :param dayind:按日期排序的名字，默认False
    :return: 合并好的文件框
    '''
    filelist = os.listdir(path)
    tempfile = []
    [tempfile.append(pd.read_csv(path + x, encoding=encoding)) for x in filelist if x[-3:] == 'csv']
    dataframe = pd.concat(tempfile)
    if dayind:
        dataframe.sort_values(['Stkcd', dayind], inplace=True)
    else:
        dataframe.sort_values(['Stkcd'], inplace=True)
    dataframe = dataframe.reset_index(drop=True)
    gc.collect()
    return dataframe

def addmonth(date):
    '''
    月份+1
    :param date: 日期
    :return: 返回月份
    '''
    year = int(date[:4])
    month = int(date[5:7])
    if month == 12:
        newmonth = str(year+1) + '/01'
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
            return addmonth(annoucement[(annoucement['Stkcd'] == datarow.iloc[0]) & (annoucement['Accper'] == datarow.iloc[acceperid])]['Actudt'].iloc[0])
        except:
            return addmonth(datarow.iloc[acceperid])
    Actudtlist = [rowChange(sheet.iloc[x, :], acceperid, annoucement) for x in range(sheet.shape[0])]
    sheet['Actudt'] = Actudtlist
    return sheet

def calna(datalist,controlp = 120):
    '''
    计算日度数据na指标数目是否符合标准
    :param datalist: 数据列表
    :param controlp: 最低可用数值
    :return:
    '''
    lsnan = np.isnan(datalist)
    controlpercent = len(datalist) - np.count_nonzero(lsnan)
    if controlpercent<controlp:
        return True
    else:
        return False

def calnam(datalist,controlp = 10):
    '''
    计算月度数据na指标数是否符合标准
    :param datalist: 数据列表
    :param controlp: 最低可用数值
    :return:
    '''
    lsnan = np.isnan(datalist)
    controlpercent = len(datalist) - np.count_nonzero(lsnan)
    if controlpercent<controlp:
        return True
    else:
        return False

def cal12(x):
    uselist = [np.nan] * 11 + [np.nanmean(x.iloc[i-11: i+1]) for i in range(11, len(x))]
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
                    stockdata[i] = stockdata[i-1]
    else:
        for i in range(1, len(stockdata)):
            if np.isnan(stockdata[i]):
                stockdata[i] = stockdata[i-1]
    return stockdata

def fillsheet(sheet, fillzero=False):
    '''
    填充整张表格
    :param sheet: 被填充表格
    :return: 无返回直接在表格内部修改
    '''
    sheet.apply(lambda x: fillna(x, fillzero))


def cal_monthend(date):
    '''
    计算日度数据月末指标
    :param date: 日期数
    :return: 返回数值值，如果是负的即位月末下一天就是月初
    '''
    newind = []
    for i in date:
        newind.append(int(i[-2:]))
    outlist = [0]
    outlist.extend(list(np.array(np.where((np.array(newind[1:]) - np.array(newind[:-1]) < 0))[0])+1))
    return outlist

def fillIntheBlank(returnData,blankFrame,stocksid, datanum=2):
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
        print('stock '+str(i)+' finished')
    return 'mission finished'

def changedateExpression(date):
    '''
    转换时间格式成'yyyy-mm-dd'格式，以便统一。
    :param date:原时间
    :return:
    '''
    return date[:4]+'-'+date[5:7]+'-'+date[-2:]


#%% factor function
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
    :param size: 市值
    :param type: 类型按照第一列股票id第二列类型
    :param save_path: 存储数据
    :return:
    '''
    typelist = set(type.iloc[:, 1])
    def onerow(x):
        for i in typelist:
            stocklist = type[type['type'] == i]
            indusmean = np.nanmean(x[stocklist])
            x[stocklist] -= indusmean
    size_ia = size.apply(lambda x: onerow(x))
    size_ia.to_csv(save_path)
    return size_ia


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
                    datatemp = fillna(datatemp)
                    datatemp = np.array(datatemp) - np.array(list(rf.iloc[monthind[i - 12]:monthind[i], 0]))
                    mktdata = np.array(
                        list(marketdata.iloc[monthind[i - 12]:monthind[i]] - rf.iloc[monthind[i - 12]:monthind[i], 0]))
                    mktdata = mktdata.reshape(-1, 1)
                    lir = LinearRegression()
                    lir.fit(mktdata, datatemp)
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
    betasq = betasheet**2
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
                    datatemp = fillna(datatemp)
                    datatemp = np.array(datatemp) - np.array(list(rf.iloc[monthind[i - 12]:monthind[i], 0]))
                    mktdata0 = list(
                        marketdata.iloc[monthind[i - 12]:monthind[i]] - rf.iloc[monthind[i - 12]:monthind[i], 0])
                    Msf = marketdata.shift(1)
                    Msf.iloc[0] = 0
                    rsf = rf.shift(1)
                    rsf.iloc[0, 0] = 0
                    mktdata1 = list(Msf.iloc[monthind[i - 12]:monthind[i]] - rsf.iloc[monthind[i - 12]:monthind[i], 0])
                    Mof = marketdata.shift(-1)
                    Mof.iloc[-1] = 0
                    rof = rf.shift(-1)
                    rof.iloc[-1, 0] = 0
                    mktdata01 = list(Mof.iloc[monthind[i - 12]:monthind[i]] - rof.iloc[monthind[i - 12]:monthind[i], 0])
                    mktdata = [mktdata0, mktdata1, mktdata01]
                    mktdata = pd.DataFrame(mktdata)
                    mktdata = mktdata.T
                    lir = LinearRegression()
                    lir.fit(mktdata, datatemp)
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
                    datatemp = fillna(datatemp)
                    datatemp = np.array(datatemp) - np.array(list(rf.iloc[monthind[i - 12]:monthind[i], 0]))
                    mktdata = np.array(
                        list(marketdata.iloc[monthind[i - 12]:monthind[i]] - rf.iloc[monthind[i - 12]:monthind[i], 0]))
                    mktdata = mktdata.reshape(-1, 1)
                    lir = LinearRegression()
                    lir.fit(mktdata, datatemp)
                    fittedvalue = lir.predict(mktdata)
                    residuals = fittedvalue - datatemp
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

def idskewness_calculator(stockdata, marketdata, stockid, dateid, monthtopindex, rf, save_path='../output/08_idskewness.csv'):
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
                    datatemp = fillna(datatemp)
                    datatemp = np.array(datatemp) - np.array(list(rf.iloc[monthind[i - 12]:monthind[i], 0]))
                    mktdata = np.array(
                        list(marketdata.iloc[monthind[i - 12]:monthind[i]] - rf.iloc[monthind[i - 12]:monthind[i], 0]))
                    mktdata = mktdata.reshape(-1, 1)
                    lir = LinearRegression()
                    lir.fit(mktdata, datatemp)
                    fittedvalue = lir.predict(mktdata)
                    residuals = fittedvalue - datatemp
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
                    datatemp = fillna(datatemp)
                    datatemp = np.array(datatemp) - np.mean(datatemp)
                    skew = np.mean(datatemp ** 3) / (np.std(datatemp)) ** 3
                    skewnesslist.append(skew)
        return skewnesslist

    timelist = np.array(dateid)[monthtopindex]
    timeindex = [x[:-3] for x in timelist]
    skewness = pd.DataFrame(columns=stockid, index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:, i]
        tempresult = cal_skewness(inputdata, monthtopindex)
        skewness.iloc[:, i] = tempresult
        print('stock '+str(stockid[i])+'\'s skewness finished')
    print('all finished')
    skewness.to_csv(save_path)
    return skewness

def coskewness_calculator(stockdata, marketdata, stockid, dateid, monthtopindex, rf, save_path='../output/10_coskewness.csv'):
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
                    datatemp = fillna(datatemp)
                    datatemp = np.array(datatemp) - np.array(list(rf.iloc[monthind[i - 12]:monthind[i], 0]))
                    mktdata0 = list(
                        marketdata.iloc[monthind[i - 12]:monthind[i]] - rf.iloc[monthind[i - 12]:monthind[i], 0])
                    mktdatasq = list(np.array(mktdata0) ** 2)
                    mktdata = [mktdata0, mktdatasq]
                    mktdata = pd.DataFrame(mktdata)
                    mktdata = mktdata.T
                    lir = LinearRegression()
                    lir.fit(mktdata, datatemp)
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
# illq数据
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
# 需要换手率frame
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
                    numofzero = np.count_nonzero(np.array(datatemp) == 0)# 计算交易为0的数目
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
    def ipodevide(x, pattern):
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
            rer = ipodevide(useindex[j], pattern)
            firstyear = int(rer[0])
            firstmonth = int(rer[1])
            yearchar = firstyear - ipoyear
            monthchar = firstmonth - ipomonth
            month = yearchar * 12 + monthchar
            stockreturn.loc[useindex[j],i] = month
        print('stock '+str(i)+'\'s age finished')
    stockreturn = stockreturn//12
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
def pricedelay_calculator(stockdata, marketdata, stockid, dateid, monthtopindex, save_path='../output/21_pricedelay.csv'):
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
                        datatemp = fillna(datatemp)
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
    pricedelayframe = pd.DataFrame(columns=stockid,index=timeindex)
    for i in range(len(stockid)):
        inputdata = stockdata.iloc[:,i]
        tempresult = cal_pricedelay(inputdata,marketdata,monthtopindex)
        pricedelayframe.iloc[:,i] = tempresult
        print('stock '+str(stockid[i])+'\'s pricedelay finished')
    print('all finished')
    pricedelayframe.to_csv(save_path)
    return pricedelayframe
# 需要收益frame
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
                    datatemp = fillna(datatemp)
                    datatemp = np.array(datatemp) - np.array(list(rf.iloc[monthind[i - 12]:monthind[i - 1], 0]))
                    mktdata = np.array(list(
                        marketdata.iloc[monthind[i - 12]:monthind[i - 1]] - rf.iloc[monthind[i - 12]:monthind[i - 1],
                                                                            0]))
                    mktdata = mktdata.reshape(-1, 1)
                    lir = LinearRegression()
                    lir.fit(mktdata, datatemp)
                    fittedvalue = lir.predict(mktdata)
                    residuals = fittedvalue - datatemp
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

def BM_calculator(balancesheet, monthdata, balanceid, sizeid, stockid, dateid, save_path='../output/28_BM.csv'):
    '''
    计算book_to_market, 所有者权益合计/A股流通市值
    :param balancesheet: 资产负债表
    :param monthdata: 月度数据，包含流通市值
    :param balanceid: 所有者权益所在位置
    :param sizeid: A股流通市值所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    BMframe = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        sizeStock = monthdata[monthdata['Stkcd'] == i]
        BMframe.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid])
        BMframe.loc[balanceStock.iloc[:, -1], i] /= sizeStock[balanceStock.iloc[:, -1], sizeid]# pandas有自己的nan判断标准，可以不用管直接除
        print('stock ' + str(i) + ' finished')
    fillsheet(BMframe)
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

def AM_calculator(balancesheet, monthdata, balanceid, sizeid, stockid, dateid, save_path='../output/30_AM.csv'):
    '''
    计算asset_to_market, 总资产/A股流通市值
    :param balancesheet: 资产负债表
    :param monthdata: 月度数据，包含流通市值
    :param balanceid: 总资产所在位置
    :param sizeid: A股流通市值所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    AMframe = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        sizeStock = monthdata[monthdata['Stkcd'] == i]
        AMframe.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid])
        AMframe.loc[balanceStock.iloc[:, -1], i] /= sizeStock[balanceStock.iloc[:, -1], sizeid]# pandas有自己的nan判断标准，可以不用管直接除
        print('stock ' + str(i) + ' finished')
    fillsheet(AMframe)
    AMframe.to_csv(save_path)
    return AMframe


def LEV_calculator(balancesheet, monthdata, balanceid, sizeid, stockid, dateid, save_path='../output/31_LEV.csv'):
    '''
    计算liabilities_to_market, 总负债/A股流通市值
    :param balancesheet: 资产负债表
    :param monthdata: 月度数据，包含流通市值
    :param balanceid: 总负债所在位置
    :param sizeid: A股流通市值所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    Levframe = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        sizeStock = monthdata[monthdata['Stkcd'] == i]
        Levframe.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid])
        Levframe.loc[balanceStock.iloc[:, -1], i] /= sizeStock[balanceStock.iloc[:, -1], sizeid]# pandas有自己的nan判断标准，可以不用管直接除
        print('stock ' + str(i) + ' finished')
    fillsheet(Levframe)
    Levframe.to_csv(save_path)
    return Levframe


def EP_calculator(balancesheet, monthdata, balanceid, sizeid, stockid, dateid, save_path='../output/32_EP.csv'):
    '''
    计算earnings-to-price, 净利润/A股流通市值
    :param balancesheet: 资产负债表
    :param monthdata: 月度数据，包含流通市值
    :param balanceid: 净利润所在位置
    :param sizeid: A股流通市值所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    EPframe = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        sizeStock = monthdata[monthdata['Stkcd'] == i]
        EPframe.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid])
        EPframe.loc[balanceStock.iloc[:, -1], i] /= sizeStock[balanceStock.iloc[:, -1], sizeid]# pandas有自己的nan判断标准，可以不用管直接除
        print('stock ' + str(i) + ' finished')
    fillsheet(EPframe)
    EPframe.to_csv(save_path)
    return EPframe

def CFP_calculator(balancesheet, monthdata, balanceid, sizeid, stockid, dateid, save_path='../output/33_CFP.csv'):
    '''
    计算cashflow-to-price, 净现金流/A股流通市值
    :param balancesheet: 利润表
    :param monthdata: 月度数据，包含流通市值
    :param balanceid: 净现金流所在位置
    :param sizeid: A股流通市值所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    CFPframe = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        sizeStock = monthdata[monthdata['Stkcd'] == i]
        CFPframe.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid])
        CFPframe.loc[balanceStock.iloc[:, -1], i] /= sizeStock[balanceStock.iloc[:, -1], sizeid]# pandas有自己的nan判断标准，可以不用管直接除
        print('stock ' + str(i) + ' finished')
    fillsheet(CFPframe)
    CFPframe.to_csv(save_path)
    return CFPframe

def CFP_ia_calculator(CFP, type, save_path='../output/34_CFP_ia.csv'):
    '''
      计算行业调整CFP
      :param CFP: CFPframe
      :param type: 类型按照第一列股票id第二列类型
      :param save_path: 存储数据
      :return:
      '''
    return size_ia_calculator(CFP, type, save_path)

def OCFP_calculator(balancesheet, monthdata, balanceid, sizeid, stockid, dateid, save_path='../output/35_OCFP.csv'):
    '''
    计算operating cashflow-to-price, 营业现金流/A股流通市值
    :param balancesheet: 现金流量表
    :param monthdata: 月度数据，包含流通市值
    :param balanceid: 营业现金流所在位置
    :param sizeid: A股流通市值所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    OCFPframe = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        sizeStock = monthdata[monthdata['Stkcd'] == i]
        OCFPframe.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid])
        OCFPframe.loc[balanceStock.iloc[:, -1], i] /= sizeStock[balanceStock.iloc[:, -1], sizeid]# pandas有自己的nan判断标准，可以不用管直接除
        print('stock ' + str(i) + ' finished')
    fillsheet(OCFPframe)
    OCFPframe.to_csv(save_path)
    return OCFPframe

def DP_calculator(balancesheet, monthdata, balanceid, sizeid, stockid, dateid, save_path='../output/36_DP.csv'):
    '''
    计算dividend-to-price, 股利/A股流通市值
    :param balancesheet: 资产负债表
    :param monthdata: 月度数据，包含流通市值
    :param balanceid: 应付股利所在位置
    :param sizeid: A股流通市值所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    DPframe = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        sizeStock = monthdata[monthdata['Stkcd'] == i]
        DPframe.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid])
        DPframe.loc[balanceStock.iloc[:, -1], i] /= sizeStock[balanceStock.iloc[:, -1], sizeid]# pandas有自己的nan判断标准，可以不用管直接除
        print('stock ' + str(i) + ' finished')
    fillsheet(DPframe)
    DPframe.to_csv(save_path)
    return DPframe

def SP_calculator(balancesheet, monthdata, balanceid, sizeid, stockid, dateid, save_path='../output/37_SP.csv'):
    '''
    计算sales-to-price, 营业收入/A股流通市值
    :param balancesheet: 资产负债表
    :param monthdata: 月度数据，包含流通市值
    :param balanceid: 营业收入所在位置
    :param sizeid: A股流通市值所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    SPframe = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        sizeStock = monthdata[monthdata['Stkcd'] == i]
        SPframe.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid])
        SPframe.loc[balanceStock.iloc[:, -1], i] /= sizeStock[balanceStock.iloc[:, -1], sizeid]# pandas有自己的nan判断标准，可以不用管直接除
        print('stock ' + str(i) + ' finished')
    fillsheet(SPframe)
    SPframe.to_csv(save_path)
    return SPframe

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
        frame.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame)
    newframe = frame / frame.shift(12) - 1
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


def BVEG_calculator(balancesheet, balanceid1, balanceid2, stockid, dateid, save_path='../output/40_BVEG.csv'):
    '''
    计算Book market values-grows-ratio, 净资产增长率, 或者直接用所有者权益合计来做
    :param balancesheet: 资产负债表
    :param balanceid1: 总资产所在位置
    :param balanceid2: 总负债所在位置
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    BVEGframe = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        BVEGframe.loc[balanceStock.iloc[:, -1], i] = \
            list(balanceStock.iloc[:, balanceid1] - balanceStock.iloc[:, balanceid2])
        print('stock ' + str(i) + ' finished')
    fillsheet(BVEGframe)
    BVEGnew = BVEGframe / BVEGframe.shift(12) - 1
    BVEGnew.to_csv(save_path)
    return BVEGnew

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
        INVTframe.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid1])
        Assetframe.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid2])
        print('stock ' + str(i) + ' finished')
    fillsheet(INVTframe)
    fillsheet(Assetframe)
    Assetframe.apply(lambda x: cal12(x))
    INVGchg = INVTframe - INVTframe.shift(12)
    INVGchgnew = INVGchg / Assetframe
    INVGchgnew.to_csv(save_path)
    return INVGchgnew

def SgINVg_calculator(Sgframe, INVGframe, save_path='../45_SGINCG.csv'):
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

def acc_calculator(balancesheet, profitsheet, cashflowsheet, profitid, cashflowid, assetid, dateid, stockid, save_path='../output/47_ACC.csv'):
    '''
    计算acc
    :param balancesheet: 资产负债表
    :param profitsheet: 利润表
    :param cashflowsheet: 现金流量表
    :param profitid: 利润id
    :param cashflowid: 现金流id
    :param assetid: 资产id
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
        profitframe.loc[profitStock.iloc[:, -1], i] = list(profitStock.iloc[:, profitid])
        Assetframe.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, assetid])
        flowframe.loc[cashflowStock.iloc[:, -1], i] = list(cashflowStock.iloc[:, cashflowid])
        print('stock ' + str(i) + ' finished')
    fillsheet(profitframe)
    fillsheet(flowframe)
    fillsheet(Assetframe)
    Assetframe.apply(lambda x: cal12(x))
    accural = profitframe - flowframe
    accframe = accural / Assetframe
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

def stdacc_calculator(accframe, save_path = '../output/49_stdacc.csv'):
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
    stdacc = accframe.apply(lambda x: accstdcal(x))
    stdacc.to_csv(save_path)
    return stdacc

def accp_calculator(profitsheet, cashflowsheet, profitid, cashflowid, netid, dateid, stockid, save_path='../output/50_ACCP.csv'):
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
        profitframe.loc[profitStock.iloc[:, -1], i] = list(profitStock.iloc[:, profitid])
        netframe.loc[profitStock.iloc[:, -1], i] = list(profitStock.iloc[:, netid])
        flowframe.loc[cashflowStock.iloc[:, -1], i] = list(cashflowStock.iloc[:, cashflowid])
        print('stock ' + str(i) + ' finished')
    fillsheet(profitframe)
    fillsheet(flowframe)
    fillsheet(netframe)
    accural = profitframe - flowframe
    accpframe = accural / netframe
    accpframe.to_csv(save_path)
    print('accp finished!!')
    return accpframe

def cinvest_calculator(balancesheet, profitsheet, balanceid, returnid, stockid, dateid, save_path='../output/51_cinvest.csv'):
    '''
    计算资本投资
    :param balancesheet: 资产负债表
    :param profitid: 利润表
    :param balanceid: 固定资产净值
    :param returnid: 营业总收入id
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    cinvestframe = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        profitStock = profitsheet[profitsheet['Stkcd'] == i]
        cinvestframe.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid])
        cinvestframe.loc[balanceStock.iloc[:, -1], i] /= profitStock[balanceStock.iloc[:, -1], returnid]# pandas有自己的nan判断标准，可以不用管直接除
        print('stock ' + str(i) + ' finished')
    def cinmeancal(x):
        uselist = [np.nan] * 8 + [np.nanmean(x.iloc[i - 8: i + 1]) for i in range(8, len(x))]
        x = uselist
        return x
    cinvestnew = cinvestframe.apply(lambda x: cinmeancal(x))
    fillsheet(cinvestnew)
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
        deprframe.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid])
        fixedframe.loc[profitStock.iloc[:, -1], i] = list(profitStock.iloc[:, returnid])
        print('stock ' + str(i) + ' finished')
    fillsheet(deprframe)
    fillsheet(fixedframe)
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
    pchdepr = deprframe / deprframe.shift(1) - 1
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

def fgr5yr_calculator(share, analyst, analystid, stockid, dateid, save_path='../output/55_fgr5yr.csv'):
    '''
       计算预期EPS增长率
       :param share: 股本
       :param analyst: 分析师文件
       :param analystid: 分析师预测收益
       :param stockid: 股票id
       :param dateid: 日期id
       :param save_path: 存储路径
       :return:
       '''
    tempframe = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        anaStock = analyst[analyst['Stkcd'] == i]
        anadate = list(set(anaStock['Rptdt']))
        acumnumber = []
        for i in anadate:
            acumnumber.append(np.nanmean(anaStock[['Rptdt'] == i].iloc[:, analystid]))
        tempframe.loc[anadate, i] = acumnumber
        print('stock ' + str(i) + ' finished')
    fillsheet(tempframe)
    fgr5 = tempframe / share# EPS
    fgr5yr = (fgr5 - fgr5.shift(12)) / fgr5.shift(12)
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
    grltfram = noaframe / noaframe.shift(12) - 1
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

def pchsale_pchinvt_calculator(SGr, INVchg, save_path='../output/601_pchsale_pchinvt.csv'):
    '''
    计算投资变化
    :param SGr:销售总收入变化
    :param INVchg:存货变化
    :param save_path:存储路径
    :return:
    '''
    pchsale_pchinvt = SGr + INVchg
    pchsale_pchinvt.to_csv(save_path)
    return pchsale_pchinvt

def pchsale_pchrect_calculator(SGr, balancesheet, balanceid, stockid, dateid, save_path='../output/61_pchsale_pchrect.csv'):
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


def pchsale_pchxsga_calculator(SGr, balancesheet, balanceid, stockid, dateid, save_path='../output/62_pchsale_pchxsga.csv'):
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
        frame1.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[0]])
        frame2.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[1]])
        frame3.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[2]])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    fillsheet(frame3)
    newframe = frame1 + frame2 + frame3
    newframechange = newframe / newframe.shift(12) - 1
    pchsale_pchxsga = SGr - newframechange
    pchsale_pchxsga.to_csv(save_path)
    return pchsale_pchxsga

def realestate_calculator(balancesheet, balanceid, stockid, dateid, save_path='../63_realestate.csv'):
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
        realframe.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid])
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
        frame1.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[0]])
        frame2.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[1]])
        frame3.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[2]])
        frame4.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, debetid[0]])
        frame5.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, debetid[1]])
        frame6.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, debetid[2]])
        frame7.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, debetid[3]])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2, True)
    fillsheet(frame3, True)
    fillsheet(frame4, True)
    fillsheet(frame5, True)
    fillsheet(frame6, True)
    fillsheet(frame7, True)
    noaframe = (frame1 - frame2 - frame3) - (frame1 - frame4 - frame5 - frame6 -frame7)
    noaframe2 = noaframe / frame1 - 1
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
        RD.loc[RDStock.iloc[:, -1], i] = list(RDStock.iloc[:, balanceid])
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
        S.loc[SStock.iloc[:, -1], i] = list(SStock.iloc[:, balanceid])
    fillsheet(S)
    RDsale = RD / S
    RDsale.to_csv(save_path)
    return RDsale
## 其实可以提取公共代码的但是我实在是提取不动了
def ROE_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid, save_path='../output/71_ROE.csv'):
    '''
    计算ROE
    :param profit: 利润表
    :param profitid: 净利润id
    :param balancesheet: 资产负债表
    :param balanceid: [总资产, 总负债id]
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
        profitStock = profit[profit['Stkcd'] == i]
        frame1.loc[profitStock.iloc[:, -1], i] = list(profitStock.iloc[:, profitid])
        frame2.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[0]])
        frame3.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[1]])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    fillsheet(frame3)
    ROEframe = frame1 / (frame2 - frame3)
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
    frame1 = pd.DataFrame(index=dateid, columns=stockid)
    frame2 = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = balancesheet[balancesheet['Stkcd'] == i]
        profitStock = profit[profit['Stkcd'] == i]
        frame1.loc[profitStock.iloc[:, -1], i] = list(profitStock.iloc[:, profitid])
        frame2.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    ROAframe = frame1 / frame2
    ROAframe.to_csv(save_path)
    return ROAframe

def CT_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid, save_path='../output/73_CT.csv', shift=True):
    '''
    计算CT， 以及用于相同类因子的计算
    :param profit: 利润表
    :param profitid: 销售收入id
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
        frame1.loc[profitStock.iloc[:, -1], i] = list(profitStock.iloc[:, profitid])
        frame2.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    if shift:
        CTframe = frame1 / frame2.shift(12)
    else:
        CTframe = frame1 / frame2
    CTframe.to_csv(save_path)
    return CTframe

def PT_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid, save_path='../output/74_PT.csv'):
    '''
    计算PT
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
        frame1.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[0]])
        frame2.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[1]])
        frame3.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[2]])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    fillsheet(frame3)
    cashpr = (size + frame1 - frame2) / frame3
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
        operprofframe.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid])
        print('stock ' + str(i) + ' finished')
    fillsheet(operprofframe)
    operprofframeuse = operprofframe / share
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
        frame1.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, profitid[0]])
        frame2.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, profitid[1]])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    mpro = frame1 - frame2
    def mprochange(x):
        uselist = [np.nan] * 23 + [(x.iloc[i] - np.nanmean(x.iloc[i - 23: i - 11])) / np.nanmean(x.iloc[i - 23: i - 11]) for i in range(23, len(x))]
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
        frame1.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, profitid])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    ATO = frame1 / NOA
    ATO.to_csv(save_path)
    return ATO
# 这里需要预期earnings frame
def chfeps_calculator(share, predictef, profit, profitid, stockid, dateid, save_path='../output/80_cheps.csv'):
    '''
    计算eps变化
    :param share:股本
    :param predictef:预测净利润frame
    :param profit: 利润表
    :param profitid: 利润id
    :param stockid: 股票id
    :param dateid: 日期id
    :param save_path: 存储路径
    :return:
    '''
    frame1 = pd.DataFrame(index=dateid, columns=stockid)
    for i in stockid:
        balanceStock = profit[profit['Stkcd'] == i]
        frame1.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, profitid])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    epschg = (frame1 - predictef) / share
    epschg.to_csv(save_path)
    return epschg

def nincr(profit, profitid, stockid, dateid, save_path='../output/81_nincr.csv'):
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
        frame1.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, profitid])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    nin = frame1 - frame1.shift(1)
    def calagrenum(x): # 估计慢死，但我懒得改了
        rsultlist = [np.nan] * 11
        for i in range(11, len(x)):
            countnum = 0
            bascount = list(x.iloc[i - 11: i + 1])[::-1]
            index = np.where(np.array(bascount)>0)[0]
            if index[0] != 0:
                rsultlist.append(countnum)
                continue
            for j in range(12):
                if index[j] != 1:
                    rsultlist.append(min(8, countnum))
                    break
                countnum += 1
            rsultlist.append(min(8, countnum))
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
        frame2.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[0]])
        frame3.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[1]])
        frame4.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[2]])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    fillsheet(frame3)
    fillsheet(frame4)
    roic = frame1 / (frame2 + frame3 - frame4)
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
        frame1.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, profitid])
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
    sfe = prereturn / price
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
        frame1.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[0]])
        frame2.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[1]])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    CR = frame1 / frame2
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
        frame1.loc[profitStock.iloc[:, -1], i] = list(profitStock.iloc[:, profitid])
        frame2.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    frame2plus = frame2.apply(lambda x: cal12(x))
    CFdebt = frame1 / frame2plus
    CFdebt.to_csv(save_path)
    return CFdebt

def salecash_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid, save_path='../output/88_salecash.csv'):
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
        frame1.loc[profitStock.iloc[:, -1], i] = list(profitStock.iloc[:, profitid])
        frame2.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    salecash = frame1 / frame2
    salecash.to_csv(save_path)
    return salecash

def saleinv_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid, save_path='../output/89_saleinv.csv'):
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
    CRG = (CR - CR.shift(12)) / CR.shift(12)
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


def salerec_calculator(profit, profitid, balancesheet, balanceid, stockid, dateid, save_path='../output/93_salerec.csv'):
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
        frame1.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[0]])
        frame2.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[1]])
        frame3.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[2]])
        frame4.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[3]])
        frame5.loc[balanceStock.iloc[:, -1], i] = list(balanceStock.iloc[:, balanceid[4]])
        print('stock ' + str(i) + ' finished')
    fillsheet(frame1)
    fillsheet(frame2)
    fillsheet(frame3)
    fillsheet(frame4)
    fillsheet(frame5)
    tang = (frame1 + 0.715 * frame2 + 0.547 * frame3 + 0.535 * frame4) / frame5
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
        for i in anadate:
            acumnumber.append(np.nansum(anaStock[['Rptdt'] == i].iloc[:, analystid]))
        nanlyst.loc[anadate, i] = acumnumber
        print('stock ' + str(i) + ' finished')
    fillsheet(nanlyst)
    nanlyst.to_csv(save_path)
    return nanlyst




#%% reading data
# basedata
codesframe = pd.read_excel('../data/stockcode/allCodes.xls', header=None)
industrialframe = pd.read_csv('../data/stockcode/TRD_Co.csv', encoding='gb2312')
industrialframe['type'] = list(map(lambda x: x[0], industrialframe['Nnindcd']))
indexDaily = pd.read_csv('../data/dailydata/index/TRD_Cndalym.csv')
AshareindexDaily = indexDaily[indexDaily['Markettype'] == 5] # deq等权重 dos加权
indexWeekly = pd.read_csv('../data/weeklydata/index/TRD_Weekcm.csv')
AshareindexWeekly = indexWeekly[indexWeekly['Markettype'] == 5] # deq等权重 dos加权
stockcodes = list(codesframe.iloc[:, 0])
typelist = set(industrialframe['type'])
rf = pd.read_csv('../data/rf/TRD_Nrrate.txt', sep='\t')
rf.set_index('Clsdt', inplace=True)
# analyst
analist = os.listdir('../data/analyst')
listfile = []
[listfile.append(pd.read_csv('../data/analyst/' + x, encoding='gb18030')) for x in analist if x[-3:] == 'csv']
analistfile = pd.concat(listfile)
analistfile = analistfile.sort_values('Stkcd')
analistfile = analistfile.reset_index(drop=True)
analistfile['number'] = [len(x.split(',')) if type(x)==str else 1 for x in analistfile['Ananm']]
# yearlydata
dividend = pd.read_csv('../data/yearlydata/TRD_Cptl.csv')
# quarterlydata
balance = pd.read_csv('../data/quarterlydata/balancesheet/FS_Combas.csv')
cash = pd.read_csv('../data/quarterlydata/cash/FS_Comscfd.csv')
profit = pd.read_csv('../data/quarterlydata/profit/FS_Comins.csv')
annoucement = pd.read_csv('../data/quarterlydata/annoucement/IAR_Forecdt.csv')
balancenew = changeSheetdate(balance, 1, annoucement)
cashnew = changeSheetdate(cash, 1, annoucement)
profitnew = changeSheetdate(profit, 1, annoucement)
# 二次引用
balancenew.to_csv('../data/quarterlydata/balancesheet/FS_CombasNew.csv')
cashnew.to_csv('../data/quarterlydata/cash/FS_ComscfdNew.csv')
profitnew.to_csv('../data/quarterlydata/profit/FS_CominsNew.csv')

# monthlydata
# monthlydata = pd.read_csv('../data/monthlydata/stock/TRD_Mnth.csv')
monthlydata = pd.read_csv('../data/monthlydata/stock/TRD_Mnth.txt', sep='\t')
monthdate = pd.read_csv('../data/monthlydata/date.txt', sep='\t')
date = monthdate[monthdate['Markettype'] == 1]
date = list(date['Trdmnt'])
# weeklydata
weeklydata = readingwhole_Data('../data/weeklydata/stock/', dayind='Trdwnt')
# dailydata
dailydata = readingwhole_Data('../data/dailydata/stock/', dayind='Trddt')
dailydate = sorted(list(set(dailydata['Trddt'])))
rfnewdate = [changedateExpression(x) for x in rf['Clsdt']]
rf['Clsdt'] = rfnewdate
rf_corresponding = rf.loc[dailydate]
rfuse = rf_corresponding.iloc[:, 1]


#%% calculating factors
monthlyreturn = pd.DataFrame(index=date, columns=stockcodes)
fillIntheBlank(monthlydata, monthlyreturn, stockcodes, datanum=-1)
monthlyreturn.to_csv('../output/monthlyreturn.csv')
# 01size
size_calculator(monthdata=monthlydata, sizeposition=5, stockid=stockcodes, dateid=date)
# 02size_ia

# 03beta
beta_calculator()

# 27largretn
largretn_calculator(monthdata=monthlydata, sizeposition=-1, stockid=stockcodes, dateid=date)
# 28BM
BM_calculator(balancenew, monthlydata, )


