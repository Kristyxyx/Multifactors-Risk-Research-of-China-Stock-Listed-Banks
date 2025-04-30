#-*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码
# Prompt Claude 3.7 Sonnet-Thinking:
# 对于未定义的函数，其操作指南在JoinQunat Doc。除非你查阅了函数的具体参数，不要改变函数的调用方式，即函数传入的参数。
# 我对于这些未定义函数的返回结果进行了数据清洗。重构这些数据清洗操作，改进其相关的算法和数据结构，以尽可能加快数据清洗速度，减少算法复杂度，并保证输出和源代码相同。
# 优化注释，中文。
# 如果我有多次使用API调用同一个数据，再合并，那么一定是为了绕过单次数据量获取限制。在这种情况下，不要将多次API调用合并为一次性API调用全部数据，保留"多次API调用，再合并数据“的数据处理方式。

import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from decimal import Decimal, getcontext
from tqdm import tqdm
from jqdata import get_all_trade_days, get_valuation, macro, get_money_flow, get_concepts, get_mtss, finance, bond
from jqfactor import get_all_factors, get_factor_values, neutralize, winsorize_med#, standardlize
from sqlalchemy.sql import func
import warnings
#from sklearn.preprocessing import QuantileTransformer

# 将警告转换为异常
warnings.filterwarnings('error')

# 定义数据获取类
class FactorsDataGetter():
# 个股信息列表获取

    # 定义在日期date时，板块代码为industry_code的板块个股信息列表industry_stocks_info取得函数get_industry_stocks_info
    def __init__(self, start_date, end_date, industry_code='801780'):
        """
        初始化因子数据获取器
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            industry_code: 行业代码，默认为'801780'(银行业)
        """
        # 获取交易日期范围
        all_dates = get_all_trade_days()
        self.dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]
        
        self.__start_date = self.dates[0]
        self.__end_date = self.dates[-1]
        
        # 获取指定行业的股票列表
        industry_stocks_codes = get_industry_stocks(industry_code=industry_code, date=end_date)
        
        # 获取所有股票的基本信息
        all_stocks_info = get_all_securities(types=['stock'], date=end_date)
        # 筛选出目标行业的股票信息
        industry_stocks_info = all_stocks_info[all_stocks_info.index.isin(industry_stocks_codes)]
        
        # 获取行业股票的行业归属信息
        industry_stocks_industries = get_industry(security=industry_stocks_codes, date=end_date)
        
        # 提取一级和二级行业信息
        # 使用列表推导式并过滤无效数据
        jq_l1_data = [(stock_code, info['jq_l1']['industry_code'], info['jq_l1']['industry_name']) 
                      for stock_code, info in industry_stocks_industries.items() 
                      if info.get('jq_l1')]
        
        jq_l2_data = [(stock_code, info['jq_l2']['industry_code'], info['jq_l2']['industry_name']) 
                      for stock_code, info in industry_stocks_industries.items() 
                      if info.get('jq_l2')]
        
        # 创建行业信息DataFrame
        jq_l1_df = pd.DataFrame(jq_l1_data, columns=['stock_code', 'jq_l1_industry_code', 'jq_l1_industry_name'])
        jq_l2_df = pd.DataFrame(jq_l2_data, columns=['stock_code', 'jq_l2_industry_code', 'jq_l2_industry_name'])
        
        # 合并行业信息
        industry_stocks_industries = pd.merge(jq_l1_df, jq_l2_df, on='stock_code', how='outer')
        industry_stocks_industries.set_index('stock_code', inplace=True)
        
        # 获取概念板块信息
        concepts = get_concepts()
        concepts['start_date'] = concepts['start_date'].dt.date
        
        # 根据一级行业名称筛选相关概念
        industry_concepts_keywords = set([name for _, _, name in jq_l1_data])
        
        # 获取与行业关键词相关的概念
        concept_stocks_map = {}
        for keyword in industry_concepts_keywords:
            # 筛选包含关键词且在结束日期前创立的概念
            matched_concepts = concepts[
                concepts['name'].str.contains(keyword) & 
                (concepts['start_date'] <= end_date)
            ]
            
            # 获取每个概念的股票列表
            for concept_code, concept_name in zip(matched_concepts.index, matched_concepts['name']):
                concept_stocks = get_concept_stocks(concept_code, date=end_date)
                # 只保留行业内的股票
                concept_stocks = set(concept_stocks) & set(industry_stocks_codes)
                # 为每个股票记录概念信息
                for stock in concept_stocks:
                    if stock not in concept_stocks_map:
                        concept_stocks_map[stock] = (concept_code, concept_name)
        
        # 创建概念信息DataFrame
        concept_data = []
        for stock in industry_stocks_codes:
            if stock in concept_stocks_map:
                concept_data.append((stock, *concept_stocks_map[stock]))
            else:
                concept_data.append((stock, np.nan, np.nan))
        
        industry_stocks_concepts = pd.DataFrame(
            concept_data, 
            columns=['stock_code', 'concept_code', 'concept_name']
        ).set_index('stock_code')
        
        # 获取流通市值信息
        industry_stocks_circulating_market_cap = get_fundamentals(
            query(
                valuation.code.label('industry_stocks_codes'),
                valuation.circulating_market_cap
            ).filter(
                valuation.code.in_(industry_stocks_codes)
            ), 
            date=end_date
        ).set_index('industry_stocks_codes')
        
        # 重命名流通市值列，添加单位信息
        industry_stocks_circulating_market_cap.rename(
            columns={'circulating_market_cap': 'circulating_market_cap/billion_yuan'}, 
            inplace=True
        )
        
        # 合并所有信息
        industry_stocks_info = industry_stocks_info.join([
            industry_stocks_industries, 
            industry_stocks_concepts, 
            industry_stocks_circulating_market_cap
        ])
        
        # 设置索引名称
        industry_stocks_info.index.name = 'stock_code'
        
        # 按流通市值降序排序
        industry_stocks_info.sort_values(
            by='circulating_market_cap/billion_yuan', 
            axis=0, 
            ascending=False,  
            inplace=True, 
            na_position='first'
        )
        
        # 保存股票信息和代码列表
        self.industry_stocks_info = industry_stocks_info
        self.__stocks_codes = list(industry_stocks_info.index)

    # 获取因子数据表
        
    def get_factors_info_and_data(self):
        """
        获取所有因子的信息和数据，并对数据进行清洗处理
        
        返回:
            factors_data: 清洗后的因子数据字典，格式为 {factor_code: factor_data DataFrame}
        """
        # 获取所有因子信息，并将因子代码设为索引
        all_factors_info = get_all_factors().rename(
            columns={'factor': 'factor_code'}, inplace=False
        ).set_index(keys='factor_code', inplace=False)
        self.all_factors_info = all_factors_info
        
        # 获取因子代码列表
        factors_codes = list(all_factors_info.index)
        
        # 预分配因子数据字典
        factors_data = {}
        
        # 设置批处理大小，减少API调用次数
        batch_size = 10
        for i in range(0, len(factors_codes), batch_size):
            batch_factors = factors_codes[i:i+batch_size]
            
            try:
                # 一次获取多个因子数据
                if len(batch_factors) > 1:
                    # 显示进度信息
                    batch_desc = f'获取因子 {i+1}-{min(i+batch_size, len(factors_codes))}/{len(factors_codes)}'
                    with tqdm(total=len(batch_factors), desc=batch_desc) as pbar:
                        # 批量获取因子数据
                        batch_data = get_factor_values(
                            securities=self.__stocks_codes, 
                            factors=batch_factors, 
                            start_date=self.__start_date, 
                            end_date=self.__end_date, 
                            count=None
                        )
                        
                        # 对每个因子进行数据清洗
                        for factor_code in batch_factors:
                            factor_data = batch_data[factor_code]
                            
                            # 设置索引名称
                            factor_data.index.name = 'date'
                            
                            # 数据类型转换
                            factor_data = factor_data.astype(float)
                            
                            # 极端值处理 - 使用中位数绝对偏差法进行缩尾处理
                            # 保留原始数据分布特征，只处理明显异常值
                            factor_data = winsorize_med(
                                factor_data, 
                                scale=5,  # 设置为5倍中位数绝对偏差
                                inclusive=True, 
                                inf2nan=False, 
                                axis=1
                            )
                            
                            # 标准化处理 - 使每个横截面的均值为0，标准差为1
                            #factor_data = standardlize(factor_data, inf2nan=False, axis=1)
                            
                            # 中性化处理 - 消除行业和市值对因子的影响
                            # 对流通市值因子只进行行业中性化，避免消除自身特性
                            if factor_code == 'circulating_market_cap':
                                factor_data = neutralize(
                                    factor_data, 
                                    how=['jq_l2'],  # 仅行业中性化
                                    date=self.__end_date, 
                                    axis=1, 
                                    fillna=None, 
                                    add_constant=True
                                )
                            else:
                                factor_data = neutralize(
                                    factor_data, 
                                    how=['jq_l2', 'ln_circulating_market_cap'],  # 行业和市值双重中性化
                                    date=self.__end_date, 
                                    axis=1, 
                                    fillna=None,
                                    add_constant=True
                                )
                            
                            # 存储处理后的因子数据
                            factors_data[factor_code] = factor_data
                            
                            # 更新进度条
                            pbar.update(1)
                else:
                    # 单个因子处理
                    factor_code = batch_factors[0]
                    with tqdm(total=1, desc=f'获取因子 {factor_code}') as pbar:
                        factor_data = get_factor_values(
                            securities=self.__stocks_codes, 
                            factors=factor_code, 
                            start_date=self.__start_date, 
                            end_date=self.__end_date, 
                            count=None
                        )[factor_code]
                        
                        factor_data.index.name = 'date'
                        factor_data = factor_data.astype(float)
                        factor_data = winsorize_med(factor_data, scale=5, inclusive=True, inf2nan=False, axis=1)
                        #factor_data = standardlize(factor_data, inf2nan=False, axis=1)
                        
                        if factor_code == 'circulating_market_cap':
                            factor_data = neutralize(
                                factor_data, 
                                how=['jq_l2'], 
                                date=self.__end_date, 
                                axis=1, 
                                fillna=None, 
                                add_constant=True
                            )
                        else:
                            factor_data = neutralize(
                                factor_data, 
                                how=['jq_l2', 'ln_circulating_market_cap'],
                                date=self.__end_date, 
                                axis=1, 
                                fillna=None,
                                add_constant=True
                            )
                        
                        factors_data[factor_code] = factor_data
                        pbar.update(1)
            
            except Exception as e:
                print(f"处理因子批次 {i+1}-{min(i+batch_size, len(factors_codes))} 时出错: {str(e)}")
                # 降级为单个因子处理
                for factor_code in batch_factors:
                    try:
                        with tqdm(total=1, desc=f'单独获取因子 {factor_code}') as pbar:
                            factor_data = get_factor_values(
                                securities=self.__stocks_codes, 
                                factors=factor_code, 
                                start_date=self.__start_date, 
                                end_date=self.__end_date, 
                                count=None
                            )[factor_code]
                            
                            factor_data.index.name = 'date'
                            factor_data = factor_data.astype(float)
                            factor_data = winsorize_med(factor_data, scale=5, inclusive=True, inf2nan=False, axis=1)
                            #factor_data = standardlize(factor_data, inf2nan=False, axis=1)
                            
                            if factor_code == 'circulating_market_cap':
                                factor_data = neutralize(
                                    factor_data, 
                                    how=['jq_l2'], 
                                    date=self.__end_date, 
                                    axis=1, 
                                    fillna=None, 
                                    add_constant=True
                                )
                            else:
                                factor_data = neutralize(
                                    factor_data, 
                                    how=['jq_l2', 'ln_circulating_market_cap'],
                                    date=self.__end_date, 
                                    axis=1, 
                                    fillna=None,
                                    add_constant=True
                                )
                            
                            factors_data[factor_code] = factor_data
                            pbar.update(1)
                    except Exception as e:
                        print(f"处理因子 {factor_code} 时出错: {str(e)}")
        
        return factors_data
    
# 获取其他基本面因子数据

# 获取基本面因子数据
def get_fundamentals_data(self, tables=[
    valuation, 
    indicator, 
    bank_indicator
]):
    """
    获取指定日期范围内的基本面因子数据
    
    参数:
        tables: 基本面指标所在表对象的列表，默认为估值表(valuation)、
                财务指标表(indicator)和银行业财务指标表(bank_indicator)
    
    返回:
        fundamentals_data: 清洗后的基本面因子数据字典，格式为 {factor_name: factor_data DataFrame}
    """
    
    # 定义根据基本面指标所在表对象和属性名称返回过滤条件的函数
    def get_attributes_filters(table, attribute_name):
        """
        判断某属性是否为所需的基本面指标
        
        参数:
            table: 基本面指标所在表对象
            attribute_name: 属性名称
            
        返回:
            boolean: 是否为所需的基本面指标
        """
        # 排除表中的无关属性
        original_filter = attribute_name not in attributes_names_to_delete
        
        # 对于估值表，只选择包含_ratio的比率指标
        if table is valuation:
            extra_filter = '_ratio' in attribute_name
        else:
            extra_filter = True
            
        # 必须同时满足原始过滤条件和额外过滤条件
        attributes_filters = original_filter and extra_filter
        return attributes_filters
    
    # 定义根据表对象和日期获取查询参数的函数
    def get_query_params(table, date):
        """
        获取查询参数，根据表的不同设置不同的查询条件
        
        参数:
            table: 基本面指标所在表对象
            date: 查询日期
            
        返回:
            dict: 包含查询参数的字典
        """
        query_params = {}
        
        # 股票代码过滤条件
        stock_code_filter = (table.code.in_(self.__stocks_codes))
        columns_filter = True
        
        # 根据表类型设置不同的查询参数
        if table is bank_indicator:
            # 银行业财务指标表使用年度数据
            query_params['filter'] = stock_code_filter & columns_filter
            query_params['date'] = None
            query_params['statDate'] = date.year
        else:
            # 其他表使用日期数据
            query_params['filter'] = stock_code_filter & columns_filter
            query_params['date'] = date
            query_params['statDate'] = None
    
        return query_params
    
    # 初始化基本面数据字典
    fundamentals_data = {}
    
    # 初始化表对象和基本面指标的映射字典
    tables_and_fundamentals_dict = {}
    
    # 定义要排除的属性名称列表
    attributes_names_to_delete = [
        '__module__', 
        '__doc__', 
        '__tablename__', 
        '__dict__',
        '__weakref__',
        'id', 
        'code', 
        'day',
        'stat_date',
        'stat_month',
        'stat_year',
        'pubDate',
        'statDate', 
        '__table__', 
        '_sa_class_manager', 
        '__init__', 
        '__mapper__'
    ]
    
    # 从各表中提取基本面指标
    for table in tables:
        # 将表对象的属性转换为字典
        table_attributes_dict = dict(vars(table))
        
        # 筛选出满足条件的基本面指标
        table_and_fundamentals_dict = {
            attribute_name: {'table': table, 'fundamental': attribute} 
            for attribute_name, attribute in table_attributes_dict.items()
            if get_attributes_filters(table, attribute_name)
        }
        
        # 将当前表的基本面指标添加到总字典中
        tables_and_fundamentals_dict.update(table_and_fundamentals_dict)
    
    # 批量处理基本面指标，每次处理5个指标以减少API调用
    batch_size = 5
    fundamental_names = list(tables_and_fundamentals_dict.keys())
    
    # 分批处理基本面指标
    for i in range(0, len(fundamental_names), batch_size):
        batch_fundamentals = fundamental_names[i:i+batch_size]
        batch_desc = f'获取基本面指标 {i+1}-{min(i+batch_size, len(fundamental_names))}/{len(fundamental_names)}'
        
        # 显示进度信息
        with tqdm(total=len(batch_fundamentals), desc=batch_desc) as pbar:
            # 处理当前批次中的每个基本面指标
            for fundamental_name in batch_fundamentals:
                fundamental_info = tables_and_fundamentals_dict[fundamental_name]
                table, fundamental = fundamental_info['table'], fundamental_info['fundamental']
                
                # 初始化数据列表
                fundamental_data_list = []
                
                # 对于每个日期获取数据
                for date in self.dates:
                    query_params = get_query_params(table, date)
                    
                    # 查询基本面数据
                    fundamental_data = get_fundamentals(
                        query(
                            table.code.label('stock_code'), 
                            fundamental
                        ).filter(
                            query_params['filter']
                        ),
                        date=query_params['date'],
                        statDate=query_params['statDate']
                    )
                    
                    # 创建包含所有股票代码的DataFrame
                    stocks_codes_df = pd.DataFrame({'stock_code': self.__stocks_codes})
                    
                    # 左连接以补全可能缺失的股票数据
                    fundamental_data = pd.merge(stocks_codes_df, fundamental_data, on='stock_code', how='left')
                    
                    # 删除股票代码列，只保留基本面数据
                    fundamental_data.drop('stock_code', axis=1, inplace=True)
                    
                    # 转置数据，使每列代表一只股票
                    fundamental_data = fundamental_data.T
                    
                    # 添加到数据列表
                    fundamental_data_list.append(fundamental_data)
                
                # 合并所有日期的数据
                fundamental_data = pd.concat(fundamental_data_list, axis=0)
                
                # 设置日期索引
                fundamental_data.index = self.dates
                fundamental_data.index.name = 'date'
                
                # 设置列名为股票代码
                fundamental_data.columns = self.__stocks_codes
                
                # 数据类型转换
                fundamental_data = fundamental_data.astype(float)
                
                # 极端值处理 - 使用中位数绝对偏差法进行缩尾处理
                fundamental_data = winsorize_med(
                    fundamental_data, 
                    scale=5,  # 设置为5倍中位数绝对偏差
                    inclusive=True, 
                    inf2nan=False, 
                    axis=1
                )
                
                # 标准化处理 - 使每个横截面的均值为0，标准差为1
                #fundamental_data = standardlize(
                #    fundamental_data, 
                #    inf2nan=False, 
                #    axis=1
                #)
                
                # 中性化处理 - 消除行业和市值对因子的影响
                fundamental_data = neutralize(
                    fundamental_data, 
                    how=['jq_l2', 'ln_circulating_market_cap'], 
                    date=self.__end_date,
                    axis=1, 
                    fillna=None, 
                    add_constant=True
                )
                
                # 将处理后的数据添加到结果字典
                fundamentals_data[fundamental_name] = fundamental_data
                
                # 更新进度条
                pbar.update(1)
    
    # 删除被用于计算加权平均资本成本(WACC)的参数，防止过拟合
    if 'interest_bearing_liabilities' in fundamentals_data:
        fundamentals_data.pop('interest_bearing_liabilities')
    if 'interest_bearing_liabilities_interest_rate' in fundamentals_data:
        fundamentals_data.pop('interest_bearing_liabilities_interest_rate')
    
    return fundamentals_data

    # 获取宏观因子数据

    def get_macros_data(self, tables=[
        macro.MAC_RMB_EXCHANGE_RATE, 
        macro.MAC_LEND_RATE, 
        macro.MAC_MONEY_SUPPLY_MONTH,
        macro.MAC_MONEY_SUPPLY_YEAR,
        macro.MAC_CURRENCY_STATE_YEAR,
        macro.MAC_OTHER_DEPOSIT,
        macro.MAC_SOCIAL_SCALE_FINANCE,
        macro.MAC_STK_MARKET, 
        macro.MAC_GOLD_FOREIGN_RESERVE,
    ]):
        """
        获取宏观经济指标数据
        
        参数:
            tables: 宏观指标所在表对象的列表，默认包含汇率、拆借利率、货币供应等表
            
        返回:
            macros_data: 清洗后的宏观因子数据字典，格式为 {macro_name: macro_data DataFrame}
        """
        
        # 定义属性过滤函数 - 决定哪些宏观指标属性需要被选取
        def get_attributes_filters(table, attribute_name):
            # 排除表的内部属性、元数据等
            original_filter = attribute_name not in attributes_names_to_delete
            
            # 根据不同表选择特定的宏观指标
            if table is macro.MAC_RMB_EXCHANGE_RATE:
                extra_filter = attribute_name in ['spot_sell', 'cash_offer_prc', 'safe_prc', 'bank_reduced_prc']
            elif table is macro.MAC_LEND_RATE:
                extra_filter = attribute_name in ['interest_rate']
            elif table is macro.MAC_MONEY_SUPPLY_MONTH:
                extra_filter = attribute_name in ['m0', 'm0_yoy', 'm1', 'm1_yoy', 'm2', 'm2_yoy']
            elif table is macro.MAC_MONEY_SUPPLY_YEAR:
                extra_filter = attribute_name not in ['m0', 'm0_yoy', 'm1', 'm1_yoy', 'm2', 'm2_yoy']
            elif table is macro.MAC_CURRENCY_STATE_YEAR:
                # 选择与中央银行影响商业银行相关的指标
                extra_filter = attribute_name in [
                    'foreign_assets', 'foreign_exchange', 'money_gold', 'government_claim',
                    'bank_claim', 'other_finance_claim', 'non_finance_claim', 'reserve_money',
                    'currency_issue', 'bank_deposit', 'other_finance_deposit',
                    'non_reserve_finance_deposit', 'bond_issue', 'government_deposit', 
                    'owned_capital'
                ]
            elif table is macro.MAC_OTHER_DEPOSIT:
                # 选择与存款性公司和商业银行竞争相关的指标
                extra_filter = attribute_name in [
                    'reserve_assets', 'reserve_deposit', 'cash_in_vault', 'government_claim',
                    'central_bank_claim', 'other_claim', 'other_finance_claim', 
                    'non_finance_claim', 'other_resident_claim', 'non_finance_liability',
                    'corporate_demand_deposit', 'corporate_time_deposit', 'personal_deposit',
                    'other_non_finance_liability', 'central_bank_liability', 
                    'other_deposit_liability', 'other_finance_liability', 'include_broad_money',
                    'bond_issue', 'paid_in_capital'
                ]
            elif table is macro.MAC_SOCIAL_SCALE_FINANCE:
                extra_filter = attribute_name in [
                    'foreign_loan', 'rmb_loan', 'entrust_loan', 'trust_loan',
                    'out_fulfilled_scale', 'corporate_bond_scale', 'non_finance_scale'
                ]
            elif table is macro.MAC_STK_MARKET:
                extra_filter = attribute_name in [
                    'total_trade_amount', 'xshg_avg_pe', 'xshe_avg_pe',
                    'xshg_avg_turnover', 'xshe_avg_turnover'
                    # 以下债券发行数据截止2006年，不使用
                    # 'treasury_bond_issue', 'company_bond_issue', 'bond_amount',
                    # 'treasury_bond_spot_amount', 'treasury_bond_repurchase_amount'
                ]
            elif table is macro.MAC_GOLD_FOREIGN_RESERVE:
                extra_filter = attribute_name in ['gold', 'foreign']
            else:
                extra_filter = True
                
            # 同时满足两个条件
            return original_filter & extra_filter
        
        # 定义查询参数构建函数 - 根据表和日期生成查询条件
        def get_query_params(table, date):
            query_params = {}
            
            # 根据不同表类型设置不同的查询条件
            if table is macro.MAC_RMB_EXCHANGE_RATE:
                # 人民币汇率表 - 筛选指定日期和美元币种
                query_params['filter'] = (table.day == date) & (table.currency_id == 40)
            elif table is macro.MAC_LEND_RATE:
                # 拆借利率表 - 根据期限设置不同参数
                if macro_name == 'interest_rate_O/N':
                    term_id = '20'  # 隔夜
                elif macro_name == 'interest_rate_3m':
                    term_id = '3'   # 3个月
                elif macro_name == 'interest_rate_1y':
                    term_id = '12'  # 1年
                
                # 筛选指定日期、人民币、银行间市场和期限
                query_params['filter'] = (table.day == date) & (table.currency_id == '1') & (table.market_id == '5') & (table.term_id == term_id)
            elif table is macro.MAC_MONEY_SUPPLY_MONTH:
                # 月度货币供应表 - 按月份筛选
                query_params['filter'] = (table.stat_month == date.strftime('%Y-%m-%d')[0: 7])
            elif table is macro.MAC_GOLD_FOREIGN_RESERVE:
                # 黄金外汇储备表 - 按月份筛选
                query_params['filter'] = (table.stat_date == date.strftime('%Y-%m-%d')[0: 7])
            elif table in [
                macro.MAC_MONEY_SUPPLY_YEAR,
                macro.MAC_CURRENCY_STATE_YEAR,
                macro.MAC_OTHER_DEPOSIT,
                macro.MAC_SOCIAL_SCALE_FINANCE,
                macro.MAC_STK_MARKET
            ]:
                # 年度数据表 - 按年份筛选
                query_params['filter'] = (table.stat_year == date.year)
            else:
                # 其他表按日期筛选
                query_params['filter'] = (table.day == date)
                
            return query_params
        
        # 初始化宏观数据存储字典
        macros_data = {}
        
        # 表和宏观指标的映射字典
        tables_and_macros_dict = {}
        
        # 定义需要排除的表属性名称
        attributes_names_to_delete = [
            '__module__', '__doc__', '__tablename__', '__dict__',
            '__weakref__', 'id', 'code', 'day', 'stat_date',
            'stat_month', 'stat_year', 'pubDate', 'statDate', 
            '__table__', '_sa_class_manager', '__init__', '__mapper__'
        ]
        
        # 从表中提取宏观指标
        for table in tables:
            # 获取表的所有属性
            table_attributes_dict = dict(vars(table))
            
            # 筛选符合条件的宏观指标
            table_and_macros_dict = {
                attribute_name: {'table': table, 'macro_sql': attribute} 
                for attribute_name, attribute in table_attributes_dict.items()
                if get_attributes_filters(table, attribute_name)
            }
            
            # 合并到总指标字典中
            tables_and_macros_dict.update(table_and_macros_dict)
            
            # 利率指标特殊处理 - 拆分为不同期限的利率
            if table is macro.MAC_LEND_RATE and 'interest_rate' in table_and_macros_dict:
                tables_and_macros_dict['interest_rate_1y'] = \
                tables_and_macros_dict['interest_rate_3m'] = \
                tables_and_macros_dict['interest_rate_O/N'] = \
                tables_and_macros_dict.pop('interest_rate')
        
        # 处理每个宏观指标
        for macro_name, macro_info in tqdm(tables_and_macros_dict.items(), desc='获取宏观指标进度'):
            table, macro_sql = macro_info['table'], macro_info['macro_sql']
            
            # 存储每个日期的数据
            macro_data_list = []
            
            # 对每个日期获取数据
            for date in self.dates:
                # 构建查询参数
                query_params = get_query_params(table, date)
                
                # 查询宏观数据
                macro_data = macro.run_query(
                    query(macro_sql).filter(query_params['filter'])
                )
                
                # 转置数据，使日期成为行索引
                macro_data = macro_data.T
                
                # 添加到列表中
                macro_data_list.append(macro_data)
                
            # 合并所有日期的数据
            macro_data = pd.concat(macro_data_list, axis=0)
            
            # 设置日期索引
            macro_data.index = self.dates
            macro_data.index.name = 'date'
            
            # 债券数据特殊处理 - 用于构建非国债发行额
            if macro_name in ['bond_amount', 'treasury_bond_spot_amount', 'treasury_bond_repurchase_amount']:
                # 暂存原始数据
                macros_data[macro_name] = macro_data
                
                # 仅在处理完所有债券相关指标后计算非国债发行额
                if macro_name != 'treasury_bond_repurchase_amount':
                    continue
                else:
                    # 计算非国债发行额 = 总债券 - (国债现货 + 国债回购)
                    macro_data = macros_data.pop('bond_amount') - (
                        macros_data.pop('treasury_bond_spot_amount') + 
                        macros_data.pop('treasury_bond_repurchase_amount')
                    )
                    macro_name = 'non_treasury_bond_amount'
            
            # 数据清洗处理
            # 1. 转换为浮点型
            macro_data = macro_data.astype(float)
            
            # 2. 极端值处理 - 中位数绝对偏差法缩尾
            macro_data = winsorize_med(macro_data, scale=5, inclusive=True, inf2nan=False, axis=0)
            
            # 3. 标准化处理 - 使均值为0，标准差为1
            #macro_data = standardlize(macro_data, inf2nan=False, axis=0)
            
            # 存储处理后的宏观数据
            macros_data[macro_name] = macro_data
        
        return macros_data
    
    # 获取资金流因子

    def get_money_flows_data(self):
        """
        获取股票资金流向数据
        
        处理步骤:
        1. 获取原始资金流向数据（如涨跌幅、净额、主力净比等）
        2. 计算构造因子（如成交量因子）
        3. 对数据进行清洗（极端值处理、标准化、行业市值中性化）
        
        返回:
            money_flows_data: 字典，键为资金流向指标名称，值为对应的DataFrame数据
        """
        # 定义原始资金流向代码列表
        money_flows_raw_codes = [
            'change_pct',     # 涨跌幅
            'net_amount_main', # 主力净额
            'net_pct_main',   # 主力净比
            'net_pct_xl',     # 超大单净比
            'net_pct_l',      # 大单净比
            'net_pct_m',      # 中单净比
            'net_pct_s'       # 小单净比
        ]
        
        # 定义使用的资金流向代码列表（包含构造因子）
        money_flows_used_codes = [
            'change_pct', 
            'net_pct_main',
            'net_pct_xl',
            'net_pct_l',
            'net_pct_m',
            'net_pct_s',
            'amount'  # 构造因子：成交量
        ]
        
        # 合并两个列表，确保不重复
        money_flows_merged_codes = money_flows_raw_codes + [
            money_flow_code 
            for money_flow_code in money_flows_used_codes 
            if money_flow_code not in money_flows_raw_codes
        ]
        
        # 初始化结果字典
        money_flows_data = {}
        
        # 处理每个资金流向指标
        for money_flow_code in tqdm(money_flows_merged_codes, desc='获取资金流向数据'):
            # 处理原始资金流向数据
            if money_flow_code in money_flows_raw_codes:
                money_flow_data_list = []
                
                # 对每个交易日获取数据
                for date in self.dates:
                    try:
                        # 获取单日资金流向数据
                        money_flow_data = get_money_flow(
                            security_list=self.__stocks_codes, 
                            start_date=date, 
                            end_date=date, 
                            fields=['sec_code', money_flow_code], 
                            count=None
                        )
                        
                        # 重命名股票代码列
                        money_flow_data.rename(columns={'sec_code': 'stock_code'}, inplace=True)
                        
                        # 处理空数据情况
                        if money_flow_data.empty:
                            money_flow_data = pd.DataFrame()
                            money_flow_data['stock_code'] = self.__stocks_codes
                            money_flow_data[money_flow_code] = np.nan
                        
                        # 确保数据包含所有股票
                        stocks_codes_df = pd.DataFrame({'stock_code': self.__stocks_codes})
                        money_flow_data = pd.merge(stocks_codes_df, money_flow_data, on='stock_code', how='left')
                        
                        # 删除股票代码列，只保留指标数据
                        money_flow_data.drop('stock_code', axis=1, inplace=True)
                        
                        # 转置数据，使股票代码成为列名
                        money_flow_data = money_flow_data.T
                        
                        # 添加到数据列表
                        money_flow_data_list.append(money_flow_data)
                        
                    except Exception as e:
                        print(f"获取资金流向数据出错，日期: {date}, 指标: {money_flow_code}, 错误: {str(e)}")
                        # 创建空数据作为占位符
                        money_flow_data = pd.DataFrame(index=[0], columns=self.__stocks_codes)
                        money_flow_data.loc[0, :] = np.nan
                        money_flow_data_list.append(money_flow_data)
                
                # 合并所有日期的数据
                money_flow_data = pd.concat(money_flow_data_list, axis=0)
                
                # 设置日期索引
                money_flow_data.index = self.dates
                money_flow_data.index.name = 'date'
                
                # 确保列名为股票代码
                money_flow_data.columns = self.__stocks_codes
                
                # 存储原始数据，用于后续构造因子
                money_flows_data[money_flow_code] = money_flow_data
            
            # 处理构造因子
            if money_flow_code in money_flows_used_codes and money_flow_code == 'amount':
                # 构造成交量因子 = 主力净额 / 主力净比
                # 注意: 由于主力净比可能为0，需要处理除零情况
                net_amount_main = money_flows_data.pop('net_amount_main')
                net_pct_main = money_flows_data['net_pct_main'].replace(0, np.nan)
                money_flows_data[money_flow_code] = net_amount_main / net_pct_main
                money_flow_data = money_flows_data[money_flow_code]
            
            # 数据清洗处理
            try:
                # 1. 类型转换
                money_flow_data = money_flow_data.astype(float)
                
                # 2. 极端值处理 - 使用中位数绝对偏差法进行缩尾处理
                money_flow_data = winsorize_med(
                    money_flow_data, 
                    scale=5,          # 设置为5倍中位数绝对偏差
                    inclusive=True,   # 包含边界值
                    inf2nan=False,    # 不将无穷值转换为NaN
                    axis=1            # 按行处理（横截面）
                )
                
                # 3. 标准化处理 - 使每个横截面的均值为0，标准差为1
                #money_flow_data = standardlize(
                #    money_flow_data, 
                #    inf2nan=False, 
                #    axis=1
                #)
                
                # 4. 中性化处理 - 消除行业和市值对因子的影响
                money_flow_data = neutralize(
                    money_flow_data, 
                    how=['jq_l2', 'ln_circulating_market_cap'],  # 行业和市值双重中性化
                    date=self.__end_date, 
                    axis=1, 
                    fillna=None,      # 不填充缺失值
                    add_constant=True # 添加常数项
                )
                
                # 更新清洗后的数据
                money_flows_data[money_flow_code] = money_flow_data
                
            except Exception as e:
                print(f"数据清洗处理出错，指标: {money_flow_code}, 错误: {str(e)}")
                # 保持原有数据不变
        
        return money_flows_data
    
    # 获取股票融资融券因子数据
    def get_securities_margins_data(self):
        """
        获取股票融资融券相关因子数据并进行清洗处理
        
        处理步骤:
        1. 获取原始融资融券数据（如融资买入金额、融券卖出金额等）
        2. 构造衍生因子（如融资融券比例、还款还券比例等）
        3. 对数据进行清洗（极端值处理、标准化、行业市值中性化）
        
        返回:
            securities_margins_data: 字典，键为融资融券指标名称，值为对应的DataFrame数据
        """
        # 初始化结果字典
        securities_margins_data = {}
        
        # 定义原始融资融券数据指标列表
        securities_margins_raw_list = [
            'fin_buy_value',     # 融资买入金额
            'fin_refund_value',  # 融资偿还金额
            'sec_sell_value',    # 融券卖出金额
            'sec_refund_value'   # 融券偿还金额
        ]
        
        # 定义构造的融资融券因子列表
        securities_margins_derived_list = [
            'fin_sec_ratio',     # 融资融券比例
            'fin_refund_ratio',  # 融资偿还比例
            'sec_refund_ratio'   # 融券偿还比例
        ]
        
        # 合并两个列表，确保不重复
        all_securities_margins = securities_margins_raw_list + securities_margins_derived_list
        
        # 定义数据清洗函数，避免代码重复
        def clean_data(data):
            """
            对融资融券数据进行清洗处理
            
            参数:
                data: 需要清洗的DataFrame数据
                
            返回:
                清洗后的DataFrame数据
            """
            # 1. 类型转换
            data = data.astype(float)
            
            # 2. 极端值处理 - 使用中位数绝对偏差法进行缩尾处理
            data = winsorize_med(
                data, 
                scale=5,          # 设置为5倍中位数绝对偏差
                inclusive=True,   # 包含边界值
                inf2nan=False,    # 不将无穷值转换为NaN
                axis=1            # 按行处理（横截面）
            )
            
            # 3. 标准化处理 - 使每个横截面的均值为0，标准差为1
            #data = standardlize(
            #    data, 
            #    inf2nan=False, 
            #    axis=1
            #)
            
            # 4. 中性化处理 - 消除行业和市值对因子的影响
            data = neutralize(
                data, 
                how=['jq_l2', 'ln_circulating_market_cap'],  # 行业和市值双重中性化
                date=self.__end_date, 
                axis=1, 
                fillna=None,      # 不填充缺失值
                add_constant=True # 添加常数项
            )
            
            return data
        
        # 批量获取原始融资融券数据
        with tqdm(total=len(securities_margins_raw_list), desc='获取融资融券原始数据') as pbar:
            for margin_name in securities_margins_raw_list:
                # 存储每个日期的数据
                margin_data_list = []
                
                # 对每个交易日获取数据
                for date in self.dates:
                    try:
                        # 获取单日融资融券数据
                        margin_data = get_mtss(
                            self.__stocks_codes, 
                            start_date=date, 
                            end_date=date, 
                            fields=['sec_code', margin_name]
                        )
                        
                        # 重命名股票代码列
                        margin_data.rename(columns={'sec_code': 'stock_code'}, inplace=True)
                        
                        # 确保数据包含所有股票
                        stocks_codes_df = pd.DataFrame({'stock_code': self.__stocks_codes})
                        margin_data = pd.merge(stocks_codes_df, margin_data, on='stock_code', how='left')
                        
                        # 删除股票代码列，只保留指标数据
                        margin_data.drop('stock_code', axis=1, inplace=True)
                        
                        # 转置数据，使股票代码成为列名
                        margin_data = margin_data.T
                        
                    except Exception as e:
                        # 发生错误时创建空数据框
                        print(f"获取{margin_name}数据出错，日期: {date}, 错误: {str(e)}")
                        margin_data = pd.DataFrame(index=[0], columns=self.__stocks_codes)
                        margin_data.loc[0, :] = np.nan
                    
                    # 添加到数据列表
                    margin_data_list.append(margin_data)
                
                # 合并所有日期的数据
                margin_data = pd.concat(margin_data_list, axis=0)
                
                # 设置日期索引
                margin_data.index = self.dates
                margin_data.index.name = 'date'
                
                # 确保列名为股票代码
                margin_data.columns = self.__stocks_codes
                
                # 保存原始数据
                securities_margins_data[margin_name] = margin_data
                
                # 更新进度条
                pbar.update(1)
        
        # 构造衍生融资融券因子
        with tqdm(total=len(securities_margins_derived_list), desc='构造融资融券衍生因子') as pbar:
            # 融资融券比例 = 融资买入金额 / (融资买入金额 + 融券卖出金额)
            fin_sec_ratio = securities_margins_data['fin_buy_value'] / (
                securities_margins_data['fin_buy_value'] + securities_margins_data['sec_sell_value']
            )
            securities_margins_data['fin_sec_ratio'] = clean_data(fin_sec_ratio)
            pbar.update(1)
            
            # 融资偿还比例 = 融资偿还金额 / 融资买入金额
            fin_refund_ratio = securities_margins_data['fin_refund_value'] / securities_margins_data['fin_buy_value']
            securities_margins_data['fin_refund_ratio'] = clean_data(fin_refund_ratio)
            pbar.update(1)
            
            # 融券偿还比例 = 融券偿还金额 / 融券卖出金额
            sec_refund_ratio = securities_margins_data['sec_refund_value'] / securities_margins_data['sec_sell_value']
            securities_margins_data['sec_refund_ratio'] = clean_data(sec_refund_ratio)
            pbar.update(1)
        
        # 对原始数据也进行清洗处理
        with tqdm(total=len(securities_margins_raw_list), desc='清洗融资融券原始数据') as pbar:
            for margin_name in securities_margins_raw_list:
                securities_margins_data[margin_name] = clean_data(securities_margins_data[margin_name])
                pbar.update(1)
        
        return securities_margins_data
    
    # 获取龙虎榜因子数据
    def get_billboards_data(self):
        """
        获取龙虎榜数据并转换为因子
        
        处理步骤:
        1. 获取原始龙虎榜数据（包括成交金额比例、异常类型代码）
        2. 对异常类型代码创建哑变量（One-hot编码）
        3. 对数据进行标准化处理流程（极端值处理、标准化、行业市值中性化）
        
        返回:
            billboards_data: 清洗后的龙虎榜因子数据字典
        """
        # 需要获取的龙虎榜指标列表
        billboards_codes = [
            'amount_ratio',     # 成交金额比例
            'abnormal_code'     # 异常类型代码
        ]
        
        # 初始化结果字典
        billboards_data = {}
        
        # 遍历指标列表
        for billboard_code in billboards_codes:
            if billboard_code in ['amount_ratio', 'abnormal_code']:
                # 初始化数据列表，用于存储每个日期的数据
                billboard_data_list = []
                
                # 对每个交易日获取数据
                for date in tqdm(self.dates, desc=f'获取龙虎榜数据: {billboard_code}'):
                    try:
                        # 获取指定日期的龙虎榜数据
                        billboard_data = get_billboard_list(
                            stock_list=self.__stocks_codes, 
                            start_date=date, 
                            end_date=date
                        )
                        
                        # 龙虎榜数据不为空
                        if not billboard_data.empty:
                            # 重命名股票代码列
                            billboard_data.rename(columns={'code': 'stock_code'}, inplace=True)
                            
                            # 筛选出总榜单（rank=0）的数据，并只保留需要的列
                            billboard_data = billboard_data.loc[
                                billboard_data['rank'] == 0, 
                                ['stock_code', 'abnormal_code', 'total_value']
                            ]
                            
                            # 对相同股票代码和异常类型的样本，汇总成交金额
                            billboard_data = billboard_data.groupby(
                                ['stock_code', 'abnormal_code'], 
                                as_index=False
                            ).agg({'total_value': 'sum'}).reset_index(drop=True)
                            
                            # 保留每个股票的第一条记录（如果有多个异常类型）
                            billboard_data = billboard_data.drop_duplicates(
                                subset='stock_code', 
                                keep='first'
                            )
                            
                            # 根据需要的指标筛选列
                            if billboard_code == 'amount_ratio':
                                billboard_data.drop(['abnormal_code'], axis=1, inplace=True)
                            elif billboard_code == 'abnormal_code':
                                billboard_data.drop(['total_value'], axis=1, inplace=True)
                        else:
                            # 创建空数据框
                            billboard_data = pd.DataFrame()
                            billboard_data['stock_code'] = self.__stocks_codes
                            billboard_data[billboard_code] = np.nan
                            
                    except Exception as e:
                        print(f"获取{date}龙虎榜数据出错: {str(e)}")
                        # 创建空数据框
                        billboard_data = pd.DataFrame()
                        billboard_data['stock_code'] = self.__stocks_codes
                        billboard_data[billboard_code] = np.nan

                    # 确保数据包含所有股票代码
                    stocks_codes_df = pd.DataFrame({'stock_code': self.__stocks_codes})
                    billboard_data = pd.merge(
                        stocks_codes_df, 
                        billboard_data, 
                        on='stock_code', 
                        how='left'
                    )
                    
                    # 删除股票代码列，保留因子值
                    billboard_data.drop(['stock_code'], axis=1, inplace=True)
                    
                    # 转置数据，使股票代码成为列名
                    billboard_data = billboard_data.T
                    
                    # 添加到数据列表
                    billboard_data_list.append(billboard_data)
                
                # 合并所有日期的数据
                billboard_data = pd.concat(billboard_data_list, axis=0)
            else:
                # 获取已存在的数据
                billboard_data = billboards_data[billboard_code]
                
            # 设置日期索引
            billboard_data.index = self.dates
            billboard_data.index.name = 'date'
            
            # 确保列名为股票代码
            billboard_data.columns = self.__stocks_codes
            
            # 对异常代码特殊处理：创建哑变量
            if billboard_code == 'abnormal_code':
                # 收集所有出现的异常代码
                abnormal_codes_set = set()
                for stock_code in self.__stocks_codes:
                    stock_values = billboard_data[stock_code].dropna().unique()
                    abnormal_codes_set.update(stock_values)
                
                # 将异常代码转换为字符串格式的列名
                abnormal_codes = [f"{billboard_code}_{int(code)}" for code in abnormal_codes_set if not np.isnan(code)]
                billboards_codes.extend(abnormal_codes)
                
                # 使用向量化操作创建哑变量
                # 预分配存储空间提高效率
                dummies_data = {}
                for abnormal_code_str in abnormal_codes:
                    code_value = float(abnormal_code_str.split('_')[-1])
                    # 创建一个与原数据同形状的全零数组
                    dummy_data = pd.DataFrame(0, index=self.dates, columns=self.__stocks_codes)
                    
                    # 对每个股票代码进行处理
                    for stock_code in self.__stocks_codes:
                        # 找出该股票该异常代码出现的日期
                        match_dates = billboard_data.index[billboard_data[stock_code] == code_value]
                        if not match_dates.empty:
                            # 将对应位置设为1
                            dummy_data.loc[match_dates, stock_code] = 1
                    
                    dummies_data[abnormal_code_str] = dummy_data
                
                # 将所有哑变量添加到结果字典
                billboards_data.update(dummies_data)
                continue
            
            # 数据清洗处理
            try:
                # 类型转换
                billboard_data = billboard_data.astype(float)
                
                # 极端值处理 - 使用中位数绝对偏差法进行缩尾处理
                billboard_data = winsorize_med(
                    billboard_data, 
                    scale=5,          # 设置为5倍中位数绝对偏差
                    inclusive=True,   # 包含边界值
                    inf2nan=False,    # 不将无穷值转换为NaN
                    axis=1            # 按行处理（横截面）
                )
                
                # 标准化处理 - 使每个横截面的均值为0，标准差为1
                #billboard_data = standardlize(
                #    billboard_data, 
                #    inf2nan=False, 
                #    axis=1
                #)
                
                # 中性化处理 - 消除行业和市值对因子的影响
                billboard_data = neutralize(
                    billboard_data, 
                    how=['jq_l2', 'ln_circulating_market_cap'],  # 行业和市值双重中性化
                    date=self.__end_date, 
                    axis=1, 
                    fillna=None,      # 不填充缺失值
                    add_constant=True # 添加常数项
                )
                
                # 只保存非空数据
                if billboard_data.any().any():
                    billboards_data[billboard_code] = billboard_data
                    
            except Exception as e:
                print(f"处理龙虎榜数据{billboard_code}时出错: {str(e)}")
        
        return billboards_data

    # 获取行业因子数据

    def get_industries_data(self):
        """
        获取个股所在行业因子数据
        
        处理步骤:
        1. 获取全部个股的行业归属信息
        2. 为每个行业创建哑变量矩阵(1表示属于该行业，0表示不属于)
        3. 对数据进行清洗处理(极端值处理、标准化、行业市值中性化)
        
        返回:
            jq_l2_industries_data: 字典，键为行业代码，值为对应的行业数据DataFrame
        """
        # 初始化结果字典
        jq_l2_industries_data = {}
        
        # 获取结束日期时所有股票的行业归属信息
        industry_stocks_industries = get_industry(security=self.__stocks_codes, date=self.__end_date)
        
        # 提取所有聚宽二级行业代码和名称(过滤掉没有二级行业归属的股票)
        jq_l2_industries_codes = {
            industry_stocks_industries[stock_code]['jq_l2']['industry_code'] 
            for stock_code in industry_stocks_industries 
            if industry_stocks_industries[stock_code].get('jq_l2')
        }
        
        # 初始化行业数据DataFrame(初始值为0表示不属于该行业)
        for jq_l2_industry_code in jq_l2_industries_codes:
            jq_l2_industries_data[jq_l2_industry_code] = pd.DataFrame(0, columns=self.__stocks_codes, index=self.dates)
        
        # 逐日获取行业归属数据
        for date in tqdm(self.dates, desc='行业数据获取进度'):
            # 获取当日所有股票的行业归属
            date_industries_info = get_industry(security=self.__stocks_codes, date=date)
            
            # 更新行业归属矩阵
            for stock_code, stock_industries_info in date_industries_info.items():
                try:
                    # 获取股票所属的二级行业代码
                    stock_jq_l2_industry_code = stock_industries_info['jq_l2']['industry_code']
                    
                    # 行业代码变更处理(2021年12月13日的行业代码调整)
                    if date < datetime.date(2021, 12, 13) < self.__end_date:
                        if stock_jq_l2_industry_code == 'HY07001':
                            stock_jq_l2_industry_code = 'HY07101'
                        elif stock_jq_l2_industry_code == 'HY07002':
                            stock_jq_l2_industry_code = 'HY07102'
                    
                    # 标记股票在当日属于该行业
                    if stock_jq_l2_industry_code in jq_l2_industries_data:
                        jq_l2_industries_data[stock_jq_l2_industry_code].loc[date, stock_code] = 1
                
                except KeyError:
                    # 股票在该日期可能未上市或没有行业信息，跳过处理
                    continue
        
        # 批量进行数据清洗处理
        for jq_l2_industry_code, jq_l2_industry_data in tqdm(list(jq_l2_industries_data.items()), desc='行业数据清洗进度'):
            # 1. 类型转换
            jq_l2_industry_data = jq_l2_industry_data.astype(float)
            
            # 2. 极端值处理 - 使用中位数绝对偏差法进行缩尾处理
            jq_l2_industry_data = winsorize_med(
                jq_l2_industry_data, 
                scale=5,          # 设置为5倍中位数绝对偏差
                inclusive=True,   # 包含边界值
                inf2nan=False,    # 不将无穷值转换为NaN
                axis=1            # 按行处理（横截面）
            )
            
            # 3. 标准化处理 - 使每个横截面的均值为0，标准差为1
            #jq_l2_industry_data = standardlize(
            #    jq_l2_industry_data, 
            #    inf2nan=False, 
            #    axis=1
            #)
            
            # 4. 中性化处理 - 消除行业和市值对因子的影响
            jq_l2_industry_data = neutralize(
                jq_l2_industry_data, 
                how=['jq_l2', 'ln_circulating_market_cap'],  # 行业和市值双重中性化
                date=self.__end_date, 
                axis=1, 
                fillna=None,      # 不填充缺失值
                add_constant=True # 添加常数项
            )
            
            # 更新清洗后的数据
            jq_l2_industries_data[jq_l2_industry_code] = jq_l2_industry_data
        
        return jq_l2_industries_data

    # 获取概念因子数据
    def get_concepts_data(self):
        """
        获取个股所在概念板块的因子数据
        
        处理步骤:
        1. 获取行业关键词，用于筛选相关概念
        2. 筛选出符合条件的概念板块代码
        3. 构建个股-概念矩阵(1表示属于该概念，0表示不属于)
        4. 对数据进行清洗(极端值处理、标准化、行业市值中性化)
        
        返回:
            concepts_data: 字典，键为概念板块代码，值为对应的概念数据DataFrame
        """
        # 获取所有概念板块的信息
        concepts = get_concepts()
        concepts['start_date'] = concepts['start_date'].dt.date
        
        # 获取个股所在行业信息
        stocks_industries = get_industry(security=self.__stocks_codes, date=self.__end_date)
        
        # 提取行业关键词集合(用于筛选相关概念)
        industry_keywords = {
            stocks_industries[stock_code]['jq_l1']['industry_name'] 
            for stock_code in stocks_industries 
            if stocks_industries[stock_code].get('jq_l1')
        }
        
        # 根据行业关键词筛选相关概念板块代码
        concepts_codes_list = []
        for keyword in industry_keywords:
            # 筛选包含关键词且在结束日期前创立的概念
            matched_concepts = pd.DataFrame(
                concepts[
                    concepts['name'].str.contains(keyword) & 
                    (concepts['start_date'] <= self.__end_date)
                ].index
            )
            if not matched_concepts.empty:
                concepts_codes_list.append(matched_concepts)
        
        # 合并所有相关概念板块代码
        if concepts_codes_list:
            concepts_codes = pd.concat(concepts_codes_list, axis=0)
            concepts_codes = list(concepts_codes.loc[:, 0])
        else:
            concepts_codes = []
        
        # 初始化概念数据字典
        concepts_data = {}
        for concept_code in concepts_codes:
            # 预分配DataFrame提高性能
            concepts_data[concept_code] = pd.DataFrame(0, columns=self.__stocks_codes, index=self.dates)
        
        # 批量获取概念板块因子数据
        for date in tqdm(self.dates, desc='概念板块数据获取进度'):
            for concept_code in concepts_codes:
                # 获取当日概念板块的所有股票
                concept_stocks_codes = get_concept_stocks(concept_code, date=date)
                
                # 使用集合运算找出同时存在于概念股和目标股票列表中的股票
                common_stocks = set(concept_stocks_codes) & set(self.__stocks_codes)
                
                # 向量化赋值(比循环逐个赋值更高效)
                if common_stocks:
                    concepts_data[concept_code].loc[date, list(common_stocks)] = 1
        
        # 过滤并处理有效概念数据
        valid_concepts = {}
        for concept_code, concept_data in tqdm(list(concepts_data.items()), desc='概念板块数据清洗'):
            # 检查是否包含有效信息(至少有一个股票属于该概念)
            if concept_data.any().any():
                # 1. 类型转换
                concept_data = concept_data.astype(float)
                
                # 2. 极端值处理 - 使用中位数绝对偏差法进行缩尾处理
                concept_data = winsorize_med(
                    concept_data, 
                    scale=5,          # 设置为5倍中位数绝对偏差
                    inclusive=True,   # 包含边界值
                    inf2nan=False,    # 不将无穷值转换为NaN
                    axis=1            # 按行处理（横截面）
                )
                
                # 3. 标准化处理 - 使每个横截面的均值为0，标准差为1
                #concept_data = standardlize(
                #    concept_data, 
                #    inf2nan=False, 
                #    axis=1
                #)
                
                # 4. 中性化处理 - 消除行业和市值对因子的影响
                concept_data = neutralize(
                    concept_data, 
                    how=['jq_l2', 'ln_circulating_market_cap'],  # 行业和市值双重中性化
                    date=self.__end_date, 
                    axis=1, 
                    fillna=None,      # 不填充缺失值
                    add_constant=True # 添加常数项
                )
                
                # 保存清洗后的数据
                valid_concepts[concept_code] = concept_data
        
        return valid_concepts

    # 获取指数因子数据
    def get_indexes_info_and_data(self):
        """
        获取指数因子数据
        
        处理步骤:
        1. 读取指数信息文件
        2. 获取每支股票所属的指数信息
        3. 对数据进行标准清洗流程(极端值处理、标准化、行业市值中性化)
        
        返回:
            indexes_data: 清洗后的指数因子数据字典
        """
        # 初始化结果字典
        indexes_data = {}
        
        # 读取指数信息文件
        indexes_info = pd.read_csv('indexes_info.csv')
        # 将发布日期转换为日期对象
        indexes_info['发布日期'] = pd.to_datetime(indexes_info['发布日期'], format='%Y/%m/%d').dt.date
        
        # 预分配结果矩阵(提高性能)
        index_data = pd.DataFrame(0, columns=self.__stocks_codes, index=self.dates)
        
        # 批量处理每个指数(比单独处理更高效)
        for _, index_info in tqdm(indexes_info.iterrows(), desc='获取指数数据进度'):
            index_code = index_info['指数代码']
            index_start_date = index_info['发布日期']
            
            # 获取该指数有效期内的所有日期
            valid_dates = [date for date in self.dates if date > index_start_date]
            
            if not valid_dates:
                continue
                
            # 对每个有效日期获取指数成分股
            for date in valid_dates:
                try:
                    # 获取指数成分股并找出与目标股票的交集
                    index_stocks_codes = get_index_stocks(index_symbol=index_code, date=date)
                    # 使用集合运算查找交集(比循环更高效)
                    common_stocks = set(index_stocks_codes) & set(self.__stocks_codes)
                    
                    # 向量化操作更新矩阵
                    if common_stocks:
                        index_data.loc[date, list(common_stocks)] += 1
                except Exception as e:
                    print(f"获取指数 {index_code} 在 {date} 的成分股时出错: {str(e)}")
        
        # 应用数据清洗流程(使用向量化操作提高性能)
        # 1. 类型转换
        index_data = index_data.astype(float)
        
        # 2. 极端值处理 - 使用中位数绝对偏差法进行缩尾处理
        index_data = winsorize_med(
            index_data, 
            scale=5,          # 设置为5倍中位数绝对偏差
            inclusive=True,   # 包含边界值
            inf2nan=False,    # 不将无穷值转换为NaN
            axis=1            # 按行处理（横截面）
        )
        
        # 3. 标准化处理 - 使每个横截面的均值为0，标准差为1
        #index_data = standardlize(
        #    index_data, 
        #    inf2nan=False, 
        #    axis=1
        #)
        
        # 4. 中性化处理 - 消除行业和市值对因子的影响
        index_data = neutralize(
            index_data, 
            how=['jq_l2', 'ln_circulating_market_cap'],  # 行业和市值双重中性化
            date=self.__end_date, 
            axis=1, 
            fillna=None,      # 不填充缺失值
            add_constant=True # 添加常数项
        )
        
        # 将处理后的数据添加到结果字典
        indexes_data['included_indexes_number'] = index_data
        
        return indexes_data

    # 获取ST股因子数据
    def get_STs_data(self):
        """
        获取股票是否为ST(特别处理)股的因子数据
        
        处理步骤:
        1. 一次性获取所有股票在所有日期的ST状态
        2. 对数据进行标准清洗流程(极端值处理、标准化、行业市值中性化)
        
        返回:
            STs_data: 清洗后的ST因子数据字典
        """
        # 初始化结果字典
        STs_data = {}
        
        # 一次性获取所有股票在研究期间的ST状态(避免多次API调用)
        ST_data = get_extras(
            'is_st', 
            self.__stocks_codes, 
            start_date=self.__start_date, 
            end_date=self.__end_date, 
            df=True
        )
        
        # 应用数据清洗流程
        # 1. 类型转换
        ST_data = ST_data.astype(float)
        
        # 2. 极端值处理 - 使用中位数绝对偏差法进行缩尾处理
        ST_data = winsorize_med(
            ST_data, 
            scale=5,          # 设置为5倍中位数绝对偏差
            inclusive=True,   # 包含边界值
            inf2nan=False,    # 不将无穷值转换为NaN
            axis=1            # 按行处理（横截面）
        )
        
        # 3. 标准化处理 - 使每个横截面的均值为0，标准差为1
        #ST_data = standardlize(
        #    ST_data, 
        #    inf2nan=False, 
        #    axis=1
        #)
        
        # 4. 中性化处理 - 消除行业和市值对因子的影响
        ST_data = neutralize(
            ST_data, 
            how=['jq_l2', 'ln_circulating_market_cap'],  # 行业和市值双重中性化
            date=self.__end_date, 
            axis=1, 
            fillna=None,      # 不填充缺失值
            add_constant=True # 添加常数项
        )
        
        # 仅保存有效数据(包含非零值)
        if ST_data.any().any():
            STs_data['is_ST'] = ST_data
        
        return STs_data

    # 获取限售解禁因子数据
    def get_locked_shares_data(self):
        """
        获取限售解禁相关因子数据
        
        处理步骤:
        1. 定义需要获取的解禁比例指标
        2. 批量获取数据并进行预处理
        3. 对数据进行标准清洗流程(极端值处理、标准化、行业市值中性化)
        
        返回:
            locked_shares_data: 清洗后的限售解禁因子数据字典
        """
        # 需要获取的限售解禁指标列表
        locked_shares_codes = ['rate1']  # 解禁比例
        
        # 初始化结果字典
        locked_shares_data = {}
        
        # 批量处理每个指标
        for locked_share_code in locked_shares_codes:
            # 存储每个日期的数据
            locked_share_data_list = []
            
            # 对每个交易日获取数据
            for date in tqdm(self.dates, desc=f'获取限售解禁数据: {locked_share_code}'):
                try:
                    # 获取当日限售解禁数据
                    locked_share_data = get_locked_shares(
                        stock_list=self.__stocks_codes, 
                        start_date=date, 
                        end_date=date
                    )
                    
                    # 筛选所需字段
                    if not locked_share_data.empty:
                        locked_share_data = locked_share_data[['code', locked_share_code]]
                        locked_share_data.rename(columns={'code': 'stock_code'}, inplace=True)
                    else:
                        # 创建空数据框
                        locked_share_data = pd.DataFrame({'stock_code': self.__stocks_codes})
                        locked_share_data[locked_share_code] = np.nan
                        
                except Exception as e:
                    print(f"获取{date}限售解禁数据出错: {str(e)}")
                    # 创建空数据框
                    locked_share_data = pd.DataFrame({'stock_code': self.__stocks_codes})
                    locked_share_data[locked_share_code] = np.nan
                
                # 确保数据包含所有目标股票
                stocks_codes_df = pd.DataFrame({'stock_code': self.__stocks_codes})
                locked_share_data = pd.merge(
                    stocks_codes_df, 
                    locked_share_data, 
                    on='stock_code', 
                    how='left'
                )
                
                # 删除股票代码列，只保留因子值
                locked_share_data.drop('stock_code', axis=1, inplace=True)
                
                # 转置数据，使股票代码成为列名
                locked_share_data = locked_share_data.T
                
                # 添加到数据列表
                locked_share_data_list.append(locked_share_data)
            
            # 合并所有日期的数据
            locked_share_data = pd.concat(locked_share_data_list, axis=0)
            
            # 设置日期索引
            locked_share_data.index = self.dates
            locked_share_data.index.name = 'date'
            
            # 确保列名为股票代码
            locked_share_data.columns = self.__stocks_codes
            
            # 应用数据清洗流程
            # 1. 类型转换
            locked_share_data = locked_share_data.astype(float)
            
            # 2. 极端值处理 - 使用中位数绝对偏差法进行缩尾处理
            locked_share_data = winsorize_med(
                locked_share_data, 
                scale=5,          # 设置为5倍中位数绝对偏差
                inclusive=True,   # 包含边界值
                inf2nan=False,    # 不将无穷值转换为NaN
                axis=1            # 按行处理（横截面）
            )
            
            # 3. 标准化处理 - 使每个横截面的均值为0，标准差为1
            #locked_share_data = standardlize(
            #    locked_share_data, 
            #    inf2nan=False, 
            #    axis=1
            #)
            
            # 4. 中性化处理 - 消除行业和市值对因子的影响
            locked_share_data = neutralize(
                locked_share_data, 
                how=['jq_l2', 'ln_circulating_market_cap'],  # 行业和市值双重中性化
                date=self.__end_date, 
                axis=1, 
                fillna=None,      # 不填充缺失值
                add_constant=True # 添加常数项
            )
            
            # 仅保存有效数据(包含非零值)
            if locked_share_data.any().any():
                locked_shares_data[locked_share_code] = locked_share_data
        
        return locked_shares_data
    
    # 获取除权除息因子数据

    def get_XRs_XDs_data(self, tables=[finance.STK_XR_XD]):
        """
        获取股票除权除息数据并转换为因子
        
        参数:
            tables: 除权除息数据表列表，默认为 finance.STK_XR_XD
            
        返回:
            XRs_XDs_data: 清洗处理后的除权除息因子数据字典
        """
        # 初始化结果字典
        XRs_XDs_data = {}
        
        # 定义需要排除的表属性
        attributes_names_to_delete = [
            '__module__', '__dict__', '__weakref__', '__doc__', 
            '_sa_class_manager', '__init__', 'id', 'day'
        ]
        
        # 需要提取的除权除息指标
        target_attributes = [
            'dividend_ratio',     # 股息率
            'transfer_ratio',     # 转增比例
            'at_bonus_ratio_rmb', # 送红股比例
            'dividend_number',    # 每股分红
            'transfer_number',    # 每股转增数
            'bonus_amount_rmb'    # 分红金额
        ]
        
        # 从表中提取所需属性
        for table in tables:
            table_attributes = {
                attr_name: getattr(table, attr_name) 
                for attr_name in dir(table) 
                if attr_name not in attributes_names_to_delete and attr_name in target_attributes
            }
            
            # 使用批处理减少API调用次数
            batch_size = 5  # 每批处理的属性数量
            attribute_batches = [list(table_attributes.items())[i:i+batch_size] 
                                 for i in range(0, len(table_attributes), batch_size)]
            
            for batch in tqdm(attribute_batches, desc='处理除权除息数据批次'):
                # 对每批属性进行处理
                for attr_name, attr_obj in batch:
                    # 预分配结果数组提高效率
                    attr_data = pd.DataFrame(index=self.dates, columns=self.__stocks_codes)
                    
                    # 按日期批量查询数据
                    date_batches = [self.dates[i:i+30] for i in range(0, len(self.dates), 30)]
                    
                    for date_batch in date_batches:
                        try:
                            # 对每个日期批次，执行一次查询获取多个日期的数据
                            # 构建查询条件
                            date_filter = finance.STK_XR_XD.report_date.in_(date_batch)
                            stock_filter = finance.STK_XR_XD.code.in_(self.__stocks_codes)
                            
                            # 执行查询
                            batch_data = finance.run_query(
                                query(
                                    finance.STK_XR_XD.code.label('stock_code'),
                                    finance.STK_XR_XD.report_date.label('date'),
                                    attr_obj
                                ).filter(
                                    stock_filter & date_filter
                                )
                            )
                            
                            # 如果有数据，处理并填充到结果数组
                            if not batch_data.empty:
                                # 设置索引便于快速查找
                                batch_data.set_index(['date', 'stock_code'], inplace=True)
                                
                                # 使用向量化操作填充数据
                                for date in date_batch:
                                    for stock_code in self.__stocks_codes:
                                        try:
                                            attr_data.loc[date, stock_code] = batch_data.loc[(date, stock_code), attr_name]
                                        except KeyError:
                                            # 该日期和股票代码组合不存在，保持为NaN
                                            continue
                                            
                        except Exception as e:
                            print(f"处理{attr_name}数据时出错: {str(e)}")
                    
                    # 数据清洗处理 - 使用向量化操作提高效率
                    if attr_data.isnull().values.all():
                        # 如果全是缺失值，跳过此属性
                        continue
                        
                    # 1. 类型转换
                    attr_data = attr_data.astype(float)
                    
                    # 2. 极端值处理 - 使用中位数绝对偏差法进行缩尾处理
                    attr_data = winsorize_med(
                        attr_data, 
                        scale=5,          # 设置为5倍中位数绝对偏差
                        inclusive=True,   # 包含边界值
                        inf2nan=False,    # 不将无穷值转换为NaN
                        axis=1            # 按行处理（横截面）
                    )
                    
                    # 3. 标准化处理 - 使每个横截面的均值为0，标准差为1
                    #attr_data = standardlize(
                    #    attr_data, 
                    #    inf2nan=False, 
                    #    axis=1
                    #)
                    
                    # 4. 中性化处理 - 消除行业和市值对因子的影响
                    attr_data = neutralize(
                        attr_data, 
                        how=['jq_l2', 'ln_circulating_market_cap'],  # 行业和市值双重中性化
                        date=self.__end_date, 
                        axis=1, 
                        fillna=None,      # 不填充缺失值
                        add_constant=True # 添加常数项
                    )
                    
                    # 仅保存有效数据(包含非NaN值)
                    if attr_data.notnull().any().any():
                        XRs_XDs_data[attr_name] = attr_data
        
        return XRs_XDs_data

    def get_returns_data(self, periods=19):
        """
        获取股票收益率数据
        
        计算指定周期的股票收益率，同时获取市场基准收益率。
        
        参数:
            periods: 收益率计算周期，默认为19（约一个月的交易日）
            
        返回:
            returns_data: 字典，包含个股收益率('returns')和市场收益率('market_returns')
        """
        # 获取所有交易日期
        all_dates = get_all_trade_days()
        
        # 查找结束日期的索引并计算向后延伸periods天的索引
        # 使用列表推导式结合条件查找索引，更高效
        last_date_index = [index + periods for index, date in enumerate(all_dates) if date == self.__end_date][0]
        
        # 获取延伸后的结束日期
        suf_end_date = all_dates[last_date_index]
        
        # 获取计算所需的完整日期范围
        suf_dates = all_dates[(all_dates >= self.__start_date) & (all_dates <= suf_end_date)]
        
        # 初始化结果字典
        returns_data = {}
        
        # 分别计算个股和市场基准的收益率
        for stocks_codes in [self.__stocks_codes, ['000001.XSHG']]:
            # 尝试使用pandas快速计算方法
            try:
                # 临时关闭警告避免干扰
                warnings.filterwarnings('ignore')
                
                # 获取价格数据 - 使用None作为start_date以获取所有历史数据，提高API效率
                prices = get_price(
                    stocks_codes, 
                    start_date=None, 
                    end_date=suf_end_date, 
                    fields=['close']
                )['close']
                
                # 计算收益率 - 使用向量化操作一次性完成所有计算
                returns = prices.shift(periods=-periods) / prices - 1
                
                # 只保留研究期内的日期
                returns = returns[returns.index.isin(self.dates)]
                
                # 根据当前处理的是个股还是市场基准，保存到相应的结果变量
                if stocks_codes == self.__stocks_codes:
                    self.__returns_data = returns_data['returns'] = returns
                elif stocks_codes == ['000001.XSHG']:
                    returns_data['market_returns'] = returns
            
            # 若pandas快速方法失败，使用替代方法
            except:
                # 获取价格数据，明确指定起始日期
                prices = get_price(
                    stocks_codes,  # 使用当前循环的stocks_codes而非硬编码
                    start_date=self.__start_date,
                    end_date=suf_end_date, 
                    fields=['close'], 
                    panel=False
                )
                
                # 转换日期格式为date对象，减少内存占用
                prices['time'] = prices['time'].dt.date
                
                # 创建日期索引的DataFrame
                price_initial = pd.DataFrame(index=suf_dates)
                price_initial.index.name = 'date'
                
                # 重构价格数据为宽格式(每列为一个股票的价格)
                for code, price in prices.groupby('code'):
                    # 预处理数据，提高join效率
                    price = price.drop('code', axis=1)
                    price.set_index('time', inplace=True)
                    price.index.name = 'date'
                    
                    # 重命名close列为股票代码并使用join添加到结果DataFrame
                    # join比merge更适合这种基于索引的操作
                    price = price.rename(columns={'close': code})
                    price_initial = price_initial.join(price)
                
                # 更新prices变量
                prices = price_initial
                
                # 计算收益率 - 向量化操作
                returns = prices.shift(periods=-periods) / prices - 1
                
                # 只保留研究期内的日期
                returns = returns[returns.index.isin(self.dates)]
                
                # 确保列顺序与输入一致
                returns = returns.reindex(columns=stocks_codes)
                
                # 保存结果
                if stocks_codes == self.__stocks_codes:
                    self.__returns_data = returns_data['returns'] = returns
                elif stocks_codes == ['000001.XSHG']:
                    returns_data['market_returns'] = returns
        
        return returns_data
    
    def get_YTMs_data(self):
        """
        获取债券到期收益率(YTM)数据
        
        计算每只股票发行的债券的到期收益率，并进行加权平均，作为债务资本成本的估计。
        处理步骤:
        1. 获取研究期内有效的债券基本信息
        2. 获取债券的票息数据
        3. 计算每只债券的现金流
        4. 使用数值优化方法求解YTM
        5. 对多只债券进行加权平均
        
        返回:
            YTMs_data: 字典，键为'YTM'，值为包含各股票YTM的DataFrame
        """
        # 设置数值计算精度
        getcontext().prec = 28
        
        # 初始化结果存储
        YTMs_data = {}
        
        # 一次性获取所有符合条件的债券基本信息，减少API调用次数
        bonds_info = bond.run_query(
            query(
                bond.BOND_BASIC_INFO.code,
                bond.BOND_BASIC_INFO.company_code,
                bond.BOND_BASIC_INFO.coupon_type_id,
                bond.BOND_BASIC_INFO.coupon_frequency,
                bond.BOND_BASIC_INFO.interest_begin_date,
                bond.BOND_BASIC_INFO.last_cash_date
            ).filter(
                bond.BOND_BASIC_INFO.company_code.in_(self.__stocks_codes),
                # 筛选研究期内有效的债券
                bond.BOND_BASIC_INFO.last_cash_date >= self.__start_date,
                bond.BOND_BASIC_INFO.interest_begin_date <= self.__end_date
            )
        )
        
        # 如果没有符合条件的债券，直接返回空结果
        if bonds_info.empty:
            YTMs_data['YTM'] = pd.DataFrame(index=self.dates, columns=self.__stocks_codes)
            # 使用1个月银行间拆借利率填充空值
            self.get_FCF_discounted_model_params_data()
            YTMs_data['YTM'].fillna(value=self.__interest_rate_1m_data, inplace=True)
            return YTMs_data
        
        # 获取债券代码列表
        bonds_codes = bonds_info['code'].unique().tolist()
        
        # 计算需要获取的票息数据总量
        sum_count_bonds_coupons = bond.run_query(
            query(
                func.count('*')
            ).filter(
                bond.BOND_COUPON.code.in_(bonds_codes),
                bond.BOND_COUPON.pub_date <= self.__end_date
            )
        ).iloc[0, 0]
        
        # 分批获取票息数据，避免单次请求过大
        bonds_coupons_list = []
        batch_size = 5000  # 每批请求的最大数量
        
        for offset in range(0, sum_count_bonds_coupons, batch_size):
            bond_coupon = bond.run_query(
                query(
                    bond.BOND_COUPON.code,
                    bond.BOND_COUPON.pub_date,
                    bond.BOND_COUPON.coupon,
                    bond.BOND_COUPON.coupon_start_date,
                    bond.BOND_COUPON.coupon_end_date
                ).filter(
                    bond.BOND_COUPON.code.in_(bonds_codes),
                    bond.BOND_COUPON.pub_date <= self.__end_date
                ).offset(offset).limit(batch_size)  # 使用limit限制返回记录数
            )
            bonds_coupons_list.append(bond_coupon)
        
        # 合并所有批次的票息数据
        bonds_coupons = pd.concat(bonds_coupons_list, axis=0, ignore_index=True)
        
        # 计算债券每月现金流的函数
        def get_C_month(date_bond_info, F=100):
            """
            计算债券的每月现金流
            
            参数:
                date_bond_info: 包含债券信息的DataFrame
                F: 债券面值，默认为100元
                
            返回:
                C_month: 债券每月现金流列表
            """
            # 转换单行Series为DataFrame
            if isinstance(date_bond_info, pd.Series):
                date_bond_info = pd.DataFrame(date_bond_info).T
                
            date_bond_info = date_bond_info.reset_index(drop=True)
            
            # 获取债券基本信息
            coupon_type = date_bond_info.loc[0, 'coupon_type_id']
            T = date_bond_info.loc[0, 'repayment_period']
            t_step = date_bond_info.loc[0, 'coupon_frequency']
            rs = date_bond_info['coupon']
            rs_start_dates = date_bond_info['coupon_start_date']
            rs_end_dates = date_bond_info['coupon_end_date']
            
            # 验证数据有效性
            if (np.isnan(coupon_type) or np.isnan(T) or np.isnan(t_step) or
                rs.empty or rs_start_dates.empty or rs_end_dates.empty or
                T <= t_step or any(rs <= 0) or any(rs_end_dates <= rs_start_dates)):
                return None
            
            # 转换为整数
            coupon_type = int(coupon_type)
            T = int(T)
            t_step = int(t_step)
            
            # 初始化现金流数组
            C_month = [0] * T
            
            # 根据票息类型计算现金流
            if coupon_type == 701001:  # 利随本清
                r_month = rs.iloc[-1] / 12 / 100  # 月化，百分比化
                C_month[-1] = F * (1 + r_month)
                
            elif coupon_type == 701002:  # 固定利率附息
                r_month = rs.iloc[-1] / 12 / 100
                
                # 使用向量化操作设置固定利率的现金流
                for t_month in range(0, T, t_step):
                    C_month[t_month] = F * r_month
                C_month[-1] += F
                
            else:  # 浮动利率
                start_date = rs_start_dates.min()
                
                # 创建日期索引以加速查找
                date_index = {}
                for idx, (start, end) in enumerate(zip(rs_start_dates, rs_end_dates)):
                    date_index[(start, end)] = idx
                
                # 计算每个付息期的现金流
                for t_month in range(0, T, t_step):
                    current_date = start_date + relativedelta(months=t_month)
                    
                    # 查找适用的利率期间
                    idx = -1
                    for (start, end), index in date_index.items():
                        if start <= current_date < end:
                            idx = index
                            break
                    
                    # 如果找不到匹配的期间，使用最后一个利率
                    if idx == -1:
                        idx = len(rs) - 1
                        
                    r_month = date_bond_info.iloc[idx]['coupon'] / 12 / 100
                    C_month[t_month] = F * r_month
                    
                C_month[-1] += F
                
            return C_month
        
        # 优化的YTM计算函数
        def calc_YTM(C_month, P=100):
            """
            计算债券到期收益率(YTM)
            
            使用数值优化方法求解满足净现值(NPV)等于债券价格的YTM。
            
            参数:
                C_month: 每月现金流列表
                P: 债券当前价格，默认为100元（面值）
                
            返回:
                YTM_month: 月度YTM
            """
            # 定义净现值计算函数
            def calc_NPV(YTM_month_vec, C_month=C_month, P=P):
                YTM_month = YTM_month_vec[0]
                
                # 使用NumPy向量化计算提高性能
                if YTM_month <= -1:  # 避免除以零或负数
                    return [float('inf')]
                    
                # 使用高精度计算避免数值溢出
                t_months = np.arange(len(C_month))
                discount_factors = np.array([float(Decimal(1) / (1 + Decimal(YTM_month)) ** (t + 1)) for t in t_months])
                PV = np.sum(np.array(C_month) * discount_factors)
                
                return [PV - P]
            
            # 计算雅可比矩阵(导数)提高求解精度
            def calc_NPV_jacobian(YTM_month_vec, C_month=C_month):
                YTM_month = YTM_month_vec[0]
                
                if YTM_month <= -1:
                    return [float('inf')]
                    
                # 使用NumPy向量化计算提高性能
                t_months = np.arange(len(C_month))
                J = np.sum(np.array([
                    -(t + 1) * float(Decimal(c) / (1 + Decimal(YTM_month)) ** (t + 2))
                    for t, c in zip(t_months, C_month)
                ]))
                
                return [J]
            
            # 使用二分搜索策略寻找合适的初始值
            # 设置搜索范围和步长
            search_ranges = [
                (0.001, 0.05, 0.005),  # 低收益率范围
                (0.05, 0.15, 0.01),    # 中等收益率范围
                (0.15, 0.35, 0.02)     # 高收益率范围
            ]
            
            for YTM_min, YTM_max, YTM_step in search_ranges:
                YTMs_initial = np.arange(YTM_min, YTM_max, YTM_step)
                
                # 尝试不同的初始值
                for YTM_initial in YTMs_initial:
                    try:
                        # 临时忽略警告
                        warnings.filterwarnings('ignore')
                        YTM_result = fsolve(calc_NPV, [YTM_initial], fprime=calc_NPV_jacobian)[0]
                        warnings.filterwarnings('error')
                        
                        # 验证解的有效性
                        if YTM_result > 0 and abs(calc_NPV([YTM_result])[0]) < 1e-5:
                            return YTM_result
                    except:
                        # 出错时继续尝试下一个初始值
                        warnings.filterwarnings('error')
                        continue
            
            # 如果上述方法都失败，使用牛顿迭代法手动求解
            YTM_guess = 0.05  # 初始猜测值
            max_iter = 50     # 最大迭代次数
            tolerance = 1e-6  # 收敛容差
            
            for _ in range(max_iter):
                npv = calc_NPV([YTM_guess])[0]
                if abs(npv) < tolerance:
                    break
                    
                derivative = calc_NPV_jacobian([YTM_guess])[0]
                if abs(derivative) < 1e-10:
                    break
                    
                # 牛顿迭代公式
                YTM_guess = YTM_guess - npv / derivative
                
                # 防止负值或过大值
                YTM_guess = max(0.001, min(0.5, YTM_guess))
            
            return YTM_guess
        
        # 计算特定日期的现金流现值
        def get_C_month_and_PV(C_month, YTM_month_star, date, date_bond_info):
            """
            计算从指定日期开始的未来现金流和现值
            
            参数:
                C_month: 完整的每月现金流列表
                YTM_month_star: 月度YTM
                date: 计算日期
                date_bond_info: 债券信息
                
            返回:
                tuple: (截断后的现金流列表, 现值)
            """
            rs_start_dates = date_bond_info['coupon_start_date']
            if rs_start_dates.empty:
                return None
                
            # 计算从发行日到计算日的月数
            start_date = rs_start_dates.min()
            interval = relativedelta(date, start_date)
            t_month = interval.years * 12 + interval.months + round(interval.days / 30)
            
            # 截断过去已经支付的现金流
            if t_month <= 0 or t_month > len(C_month):
                return None
                
            C_month_future = C_month[t_month - 1:]
            
            if not C_month_future:
                return None
                
            # 计算现值
            PV = sum([
                c_month / (1 + YTM_month_star) ** (t + 1) 
                for t, c_month in enumerate(C_month_future)
            ])
            
            return C_month_future, PV
        
        # 初始化结果DataFrame
        YTMs_data['YTM'] = pd.DataFrame(index=self.dates, columns=self.__stocks_codes)
        
        # 使用哈希表存储已计算的现金流模式，避免重复计算
        cash_flow_cache = {}
        ytm_cache = {}
        
        # 按日期和股票代码计算YTM
        for date in tqdm(self.dates, desc='计算债券到期收益率(YTM)'):
            date_str = str(date)
            
            for stock_code in self.__stocks_codes:
                # 筛选当前日期和股票的有效债券
                bonds_info_mask = (
                    (bonds_info['company_code'] == stock_code) & 
                    (bonds_info['interest_begin_date'] < date) & 
                    (bonds_info['last_cash_date'] > date)
                )
                
                # 如果没有有效债券，跳过
                if not bonds_info_mask.any():
                    continue
                    
                # 筛选已公布的票息数据
                bonds_coupons_mask = (bonds_coupons['pub_date'] < date)
                date_bonds_info = bonds_info[bonds_info_mask].merge(
                    bonds_coupons[bonds_coupons_mask], 
                    on='code'
                )
                
                if date_bonds_info.empty:
                    continue
                    
                # 计算每只债券的现金流和YTM
                Cs_month = {}
                YTMs_month_star = {}
                PVs = {}
                
                # 处理每只债券
                for code, bond_info in date_bonds_info.groupby('code'):
                    # 计算债券期限(月)
                    interval = relativedelta(
                        bond_info['last_cash_date'].iloc[0], 
                        bond_info['interest_begin_date'].iloc[0]
                    )
                    bond_info = bond_info.copy()
                    bond_info.loc[:, 'repayment_period'] = (
                        interval.years * 12 + interval.months + round(interval.days / 30)
                    )
                    
                    # 生成现金流key用于缓存
                    cache_key = f"{code}_{date_str}"
                    
                    # 检查缓存中是否已有该债券的现金流
                    if cache_key in cash_flow_cache:
                        C_month = cash_flow_cache[cache_key]
                    else:
                        C_month = get_C_month(bond_info)
                        if C_month:
                            cash_flow_cache[cache_key] = C_month
                    
                    if not C_month:
                        continue
                        
                    # 计算YTM并缓存
                    if cache_key in ytm_cache:
                        YTM_month_star = ytm_cache[cache_key]
                    else:
                        YTM_month_star = calc_YTM(C_month)
                        ytm_cache[cache_key] = YTM_month_star
                        
                    YTMs_month_star[code] = YTM_month_star
                    
                    # 计算当前日期的现金流和现值
                    result = get_C_month_and_PV(C_month, YTM_month_star, date, bond_info)
                    if result:
                        Cs_month[code], PVs[code] = result
                
                # 如果没有有效计算结果，跳过
                if not Cs_month or not PVs:
                    continue
                    
                # 等权重平均
                weights = {code: 1.0 / len(Cs_month) for code in Cs_month}
                
                # 计算加权平均现金流和现值
                max_T_month = max(len(C_month) for C_month in Cs_month.values())
                C_month_agg = [0] * max_T_month
                PV_agg = 0
                
                for code, C_month in Cs_month.items():
                    weight = weights[code]
                    PV = PVs[code]
                    
                    # 填充加权现金流
                    for i in range(len(C_month)):
                        C_month_agg[i] += weight * C_month[i]
                        
                    PV_agg += weight * PV
                
                # 计算加权平均YTM
                YTM_month_star = calc_YTM(C_month_agg, PV_agg)
                
                # 处理异常低值
                if YTM_month_star < 0.01:
                    # 使用各债券YTM的平均值作为下限
                    YTM_month_star_average = np.mean(list(YTMs_month_star.values()))
                    YTM_month_star = max(YTM_month_star, YTM_month_star_average)
                    
                # 存储计算结果
                YTMs_data['YTM'].loc[date, stock_code] = YTM_month_star
        
        # 使用1个月银行间拆借利率填充缺失值
        self.get_FCF_discounted_model_params_data()
        YTMs_data['YTM'].fillna(value=self.__interest_rate_1m_data, inplace=True)
        
        return YTMs_data
    
    # 获取FCF_discounted_model所需的参数
    def get_FCF_discounted_model_params_data(self, tables=[
        bank_indicator,
        valuation, 
        balance,
        finance.FINANCE_BALANCE_SHEET,
        income,
        macro.MAC_LEND_RATE
    ]):
        """
        获取构建自由现金流折现模型(FCF Discounted Model)所需的参数
        
        参数:
            tables: 所需数据来源的表列表，默认包含银行指标表、估值表、资产负债表、财务报表、收入表和拆借利率表
            
        返回:
            FCF_discounted_model_params_data: 字典，包含构建模型所需的所有参数数据
        """
        # 定义要排除的属性名称列表
        attributes_names_to_delete = [
            '__module__', '__doc__', '__tablename__', '__dict__', '__weakref__',
            'id', 'code', 'day', 'stat_date', 'stat_month', 'stat_year',
            'pubDate', 'statDate', '__table__', '_sa_class_manager', '__init__', '__mapper__'
        ]
        
        # 定义属性过滤函数 - 决定哪些属性需要被获取
        def get_attributes_filters(table, attribute_name):
            """
            判断某个表的属性是否需要被获取
            
            参数:
                table: 数据表对象
                attribute_name: 属性名称
                
            返回:
                boolean: 是否需要获取该属性
            """
            # 排除基础属性
            original_filter = attribute_name not in attributes_names_to_delete
            
            # 根据不同表设置不同的筛选条件
            if table is bank_indicator:
                # 银行指标表 - 仅获取计息负债及其利率
                extra_filter = attribute_name in [
                    'interest_bearing_liabilities', 
                    'interest_bearing_liabilities_interest_rate'
                ]
            elif table is valuation:
                # 估值表 - 仅获取流通市值
                extra_filter = attribute_name == 'circulating_market_cap'
            elif table is balance:
                # 资产负债表 - 获取现金及短期投资相关项目
                extra_filter = attribute_name in [
                    'cash_equivalents', 'lend_capital', 
                    'trading_assets', 'bought_sellback_assets'
                ]
            elif table is finance.FINANCE_BALANCE_SHEET:
                # 财务报表 - 获取现金相关项目
                extra_filter = attribute_name in [
                    'cash_in_cb', 'deposit_in_ib'
                ]
            elif table is income:
                # 收入表 - 获取税率计算所需项目
                extra_filter = attribute_name in [
                    'income_tax_expense', 'total_profit'
                ]
            elif table is macro.MAC_LEND_RATE:
                # 拆借利率表 - 仅获取利率数据
                extra_filter = attribute_name == 'interest_rate'
            else:
                # 其他表 - 不做额外筛选
                extra_filter = True
                
            # 同时满足原始过滤条件和额外过滤条件
            return original_filter and extra_filter
        
        # 获取季度末日期的辅助函数
        def get_last_day_of_quarter(date):
            """
            获取指定日期所在季度的最后一天
            
            参数:
                date: 指定日期
                
            返回:
                季度末日期
            """
            # 获取当前季度的第一个月
            quarter_start = (date.month - 1) // 3 * 3 + 1
            
            # 计算下一季度的第一个月
            next_quarter_month = quarter_start + 3
            next_quarter_year = date.year
            
            # 如果下一季度是下一年的第一季度
            if next_quarter_month > 12:
                next_quarter_month = 1
                next_quarter_year += 1
                
            # 下一季度第一天的前一天就是当前季度的最后一天
            return datetime.date(next_quarter_year, next_quarter_month, 1) - datetime.timedelta(days=1)
        
        # 获取查询参数的辅助函数
        def get_query_params(table, date, attribute_name):
            """
            根据表类型和日期构建查询参数
            
            参数:
                table: 数据表对象
                date: 查询日期
                attribute_name: 属性名称
                
            返回:
                dict: 查询参数字典
            """
            query_params = {}
            
            # 根据不同表设置不同的查询参数
            if table is bank_indicator:
                # 银行指标表使用年度数据
                stock_code_filter = (table.code.in_(self.__stocks_codes))
                query_params['date'] = None
                query_params['statDate'] = date.year
                query_params['filter'] = stock_code_filter
            elif table is finance.FINANCE_BALANCE_SHEET:
                # 财务报表使用季度报告
                stock_code_filter = (table.code.in_(self.__stocks_codes))
                quarter_end_date = get_last_day_of_quarter(date)
                report_filter = (table.report_date == quarter_end_date) & (table.report_type == 0)
                query_params['filter'] = stock_code_filter & report_filter
            elif table is macro.MAC_LEND_RATE:
                # 拆借利率表需要设置期限和市场
                term_id = '1'  # 1个月期限
                if attribute_name == 'interest_rate_1m':
                    term_id = '1'
                
                # 筛选特定日期、人民币、银行间市场的利率
                date_filter = (table.day == date)
                market_filter = (table.currency_id == '1') & (table.market_id == '5') & (table.term_id == term_id)
                query_params['filter'] = date_filter & market_filter
            else:
                # 其他表使用日期数据
                stock_code_filter = (table.code.in_(self.__stocks_codes))
                query_params['date'] = date
                query_params['statDate'] = None
                query_params['filter'] = stock_code_filter
                
            return query_params
        
        # 创建上市状态掩码，标识股票在每个时间点是否处于上市状态
        def get_mask(stocks_info, dates):
            """
            创建标识股票上市状态的掩码矩阵
            
            参数:
                stocks_info: 股票基本信息DataFrame
                dates: 日期列表
                
            返回:
                DataFrame: 布尔矩阵，True表示股票在该日期处于上市状态
            """
            # 预分配结果矩阵
            mask = pd.DataFrame(False, index=dates, columns=stocks_info.index.tolist())
            
            # 向量化操作，一次性为所有股票设置上市状态
            for stock_code in mask.columns:
                start_date = stocks_info.loc[stock_code, 'start_date']
                end_date = stocks_info.loc[stock_code, 'end_date']
                mask[stock_code] = (mask.index >= start_date) & (mask.index <= end_date)
                
            return mask
        
        # 创建上市状态掩码
        mask = get_mask(self.industry_stocks_info, self.dates)
        
        # 初始化结果字典和表-参数映射字典
        FCF_discounted_model_params_data = {}
        tables_and_params_dict = {}
        
        # 从各表中提取所需属性
        for table in tables:
            # 获取表的所有属性
            table_attributes = dict(vars(table))
            
            # 筛选满足条件的属性
            table_params = {
                attr_name: {'table': table, 'param': attr} 
                for attr_name, attr in table_attributes.items()
                if get_attributes_filters(table, attr_name)
            }
            
            # 合并到总字典中
            tables_and_params_dict.update(table_params)
            
            # 利率数据特殊处理 - 重命名为1个月期利率
            if table is macro.MAC_LEND_RATE and 'interest_rate' in table_params:
                tables_and_params_dict['interest_rate_1m'] = tables_and_params_dict.pop('interest_rate')
        
        # 批量获取参数数据
        for param_name, param_info in tqdm(tables_and_params_dict.items(), desc='获取模型参数'):
            table, param = param_info['table'], param_info['param']
            
            # 存储每个日期的数据
            param_data_list = []
            
            # 获取每个日期的数据
            for date in self.dates:
                # 构建查询参数
                query_params = get_query_params(table, date, param_name)
                
                # 根据表类型执行不同的查询
                if table is finance.FINANCE_BALANCE_SHEET:
                    param_data = finance.run_query(
                        query(
                            table.code.label('stock_code'), 
                            param
                        ).filter(
                            query_params['filter']
                        )
                    )
                elif table is macro.MAC_LEND_RATE:
                    param_data = macro.run_query(
                        query(
                            param
                        ).filter(
                            query_params['filter']
                        )
                    )
                else:
                    param_data = get_fundamentals(
                        query(
                            table.code.label('stock_code'), 
                            param
                        ).filter(
                            query_params['filter']
                        ),
                        date=query_params['date'],
                        statDate=query_params['statDate']
                    )
                    
                # 处理非宏观数据
                if table is not macro.MAC_LEND_RATE:
                    # 创建包含所有股票代码的DataFrame
                    stocks_df = pd.DataFrame({'stock_code': self.__stocks_codes})
                    
                    # 左连接以补全可能缺失的股票
                    param_data = pd.merge(
                        stocks_df, 
                        param_data, 
                        on='stock_code', 
                        how='left'
                    )
                    
                    # 删除股票代码列，只保留参数数据
                    param_data.drop('stock_code', axis=1, inplace=True)
                    
                # 转置数据，使股票代码成为列名
                param_data = param_data.T
                
                # 添加到数据列表
                param_data_list.append(param_data)
                
            # 合并所有日期的数据
            param_data = pd.concat(param_data_list, axis=0)
            
            # 设置日期索引
            param_data.index = self.dates
            param_data.index.name = 'date'
            
            # 设置列名为股票代码（非宏观数据）
            if table is not macro.MAC_LEND_RATE:
                param_data.columns = self.__stocks_codes
                
            # 填充缺失值 - 对非关键参数使用中位数填充
            if param_name not in ['interest_bearing_liabilities', 
                                 'interest_bearing_liabilities_interest_rate', 
                                 'interest_rate_1m']:
                # 使用向量化操作按行填充缺失值
                param_data = param_data.apply(
                    lambda row: row.fillna(row.median()), 
                    axis=1
                )
                
                # 根据上市状态掩码重置未上市股票的数据为NaN
                param_data = param_data.mask(~mask)
                
            # 保存获取的参数数据
            FCF_discounted_model_params_data[param_name] = param_data
        
        # 单位调整
        FCF_discounted_model_params_data['circulating_market_cap'] *= 1e8  # 转换为元
        FCF_discounted_model_params_data['interest_bearing_liabilities_interest_rate'] /= 1.2e3  # 调整利率比例
        FCF_discounted_model_params_data['interest_rate_1m'] /= 1.2e3  # 调整利率比例
        
        # 提取各项数据用于计算组合指标
        # 现金(Cash)：库存现金、存放中央银行款项和存放同业款项
        cash_equivalents = FCF_discounted_model_params_data.pop('cash_equivalents')
        cash_in_cb = FCF_discounted_model_params_data.pop('cash_in_cb')
        deposit_in_ib = FCF_discounted_model_params_data.pop('deposit_in_ib')
        
        # 短期投资：交易性金融资产、拆出资金和买入返售金融资产
        trading_assets = FCF_discounted_model_params_data.pop('trading_assets')
        lend_capital = FCF_discounted_model_params_data.pop('lend_capital')
        bought_sellback_assets = FCF_discounted_model_params_data.pop('bought_sellback_assets')
        
        # 其他关键数据
        interest_bearing_liabilities = FCF_discounted_model_params_data.pop('interest_bearing_liabilities')
        circulating_market_cap = FCF_discounted_model_params_data.pop('circulating_market_cap')
        income_tax_expense = FCF_discounted_model_params_data.pop('income_tax_expense')
        total_profit = FCF_discounted_model_params_data.pop('total_profit')
        interest_bearing_liabilities_interest_rate = FCF_discounted_model_params_data.pop('interest_bearing_liabilities_interest_rate')
        
        # 获取收益率数据
        self.get_returns_data()
        returns = self.__returns_data
        
        # 计算组合指标 - 使用向量化操作提高效率
        # 现金总额
        cash = cash_equivalents + cash_in_cb + deposit_in_ib
        
        # 短期投资总额
        short_investment = trading_assets + lend_capital + bought_sellback_assets
        
        # 净债务 = 计息负债 - (现金 + 短期投资)
        net_debt = interest_bearing_liabilities - (cash + short_investment)
        
        # 企业价值 = 流通市值 + 净债务
        enterprise_value = circulating_market_cap + net_debt
        
        # 计算企业价值权重 - 表示企业在截面中的系统重要性
        FCF_discounted_model_params_data['panal_enterprise_value_weights'] = enterprise_value.apply(
            lambda row: row / row.sum(), 
            axis=1
        )
        
        # 计算债务与企业价值比率
        debt_to_enterprise_value_ratio = net_debt / enterprise_value
        
        # 计算所得税税率
        income_tax_rate = income_tax_expense / total_profit
        
        # 计算加权平均资本成本(WACC)
        # WACC = 权益成本比例 * 权益成本 + 债务成本比例 * 债务成本 * (1-税率)
        FCF_discounted_model_params_data['r_wacc'] = (1 - debt_to_enterprise_value_ratio) * returns \
        + debt_to_enterprise_value_ratio * interest_bearing_liabilities_interest_rate * (1 - income_tax_rate)
        
        # 保存1个月拆借利率数据，用于填充YTM数据缺失值
        self.__interest_rate_1m_data = FCF_discounted_model_params_data['interest_rate_1m']
        
        return FCF_discounted_model_params_data
