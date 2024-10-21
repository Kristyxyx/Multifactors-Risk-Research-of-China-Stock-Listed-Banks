#-*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码

import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from decimal import Decimal, getcontext
from tqdm import tqdm
from jqdata import get_all_trade_days, get_valuation, macro, get_money_flow, get_concepts, get_mtss, finance, bond
from jqfactor import get_all_factors, get_factor_values, standardlize, neutralize, winsorize_med
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
        # 调用交易日取得函数get_all_trade_days传出所有交易日
        all_dates = get_all_trade_days()
        # 利用从开始时期start_date到结束时期end_date的因子数据表的索引.index作为开市时间列表定义时期列表dates
        self.dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]
        
        self.__start_date = self.dates[0]
        self.__end_date = self.dates[-1]
        
        # 传出在日期date时，板块代码为industry_code的板块个股代码列表industry_stocks_codes
        industry_stocks_codes = get_industry_stocks(industry_code=industry_code, date=end_date)
        
        # 板块个股再市时期整理
        # 传出在日期date时，所有个股信息列表all_stocks_info
        all_stocks_info = get_all_securities(types=['stock'], date=end_date)
        # 传出在date时，板块个股代码在板块个股代码列表industry_stocks_codes中的板块个股信息列表industry_stocks_info
        industry_stocks_info = all_stocks_info[all_stocks_info.index.isin(industry_stocks_codes)]
        
        # 板块个股所在行业整理
        # 传出在日期date时，板块个股代码为industry_stocks_codes的板块个股所在行业字典industry_stocks_industries
        industry_stocks_industries = get_industry(security=industry_stocks_codes, date=end_date)
        # 传出板块个股所在行业字典industry_stocks_industries中各代码为stock_code的板块个股所在聚宽一级'jq_l1'和二级行业jq_l2'的代码industry_code和名称industry_name列表jq_l1/2_industry_code/name,过滤industry_stocks_industries[stock_code]字典中不存在'jq_l1/2'索引的情况
        jq_l1_industries_codes = [industry_stocks_industries[stock_code]['jq_l1']['industry_code'] for stock_code in industry_stocks_industries.keys() if industry_stocks_industries[stock_code].get('jq_l1')]
        jq_l1_industries_names = [industry_stocks_industries[stock_code]['jq_l1']['industry_name'] for stock_code in industry_stocks_industries.keys() if industry_stocks_industries[stock_code].get('jq_l1')]
        jq_l2_industries_codes = [industry_stocks_industries[stock_code]['jq_l2']['industry_code'] for stock_code in industry_stocks_industries.keys() if industry_stocks_industries[stock_code].get('jq_l2')]
        jq_l2_industries_names = [industry_stocks_industries[stock_code]['jq_l2']['industry_name'] for stock_code in industry_stocks_industries.keys() if industry_stocks_industries[stock_code].get('jq_l2')]
        # 将板块个股所在行业字典industry_stocks_industries整理为板块个股所在行业表industry_stocks_industries，其列为板块个股所在聚宽一级和二级行业的代码和名称列表jq_l1/2_industry_code/name，索引为原字典的键，即为板块个股代码列表industry_stocks_codes
        industry_stocks_industries = pd.DataFrame(
            {
                'jq_l1_industry_code': jq_l1_industries_codes,
                'jq_l1_industry_name': jq_l1_industries_names,
                'jq_l2_industry_code': jq_l2_industries_codes,
                'jq_l2_industry_name': jq_l2_industries_names
            },
            index=industry_stocks_info.index
        )
        
        # 板块个股所在概念板块整理
        # 传出所有概念板块的信息数据表concepts，列名与信息如下：'index': 概念代码，name: 概念名称，start_date: 开始日期
        concepts = get_concepts()
        # 将聚宽一级行业的名称列表jq_l1_industries_names的取值种类列表set()作为板块个股所在概念板块关键词列表industry_concepts_keywords传出，即将板块名称的设置为概念板块的关键词
        concepts['start_date'] = concepts['start_date'].dt.date
        industry_concepts_keywords = set(jq_l1_industries_names)
        # 对于板块个股所在概念板块关键词列表industry_concepts_keywords中的每个板块个股所在概念板块关键词concepts_keyword
        for industry_concepts_keyword in industry_concepts_keywords:
            # 将所有概念板块的信息数据表concepts中板块个股所在概念板块名称concepts['name']包含板块个股所在概念板块关键词.str.contains(industry_concepts_keyword)，且板块个股所在概念板块创立时间pd.to_datetime(concepts['start_date'])早于日期date的概念板块代码.index和名称.name传出为板块个股所在概念板块代码列表industry_concepts_codes和板块个股所在概念板块名称列表industry_concepts_names
            industry_concepts_codes = concepts[concepts['name'].str.contains(industry_concepts_keyword) & (concepts['start_date'] <= end_date)].index
            industry_concepts_names = concepts[concepts['name'].str.contains(industry_concepts_keyword) & (concepts['start_date'] <= end_date)].name
            # 将板块个股所在概念板块代码industry_concepts_codes和板块个股所在概念板块名称industry_concepts_names转换为相应列表list(industry_concepts_codes), list(industry_concepts_names，并打包dict(zip())为板块个股所在概念板块信息字典industry_concepts_info，其结构为{..., industry_concept_code: industry_concept_name, ....}
            industry_concepts_info = dict(zip(list(industry_concepts_codes), list(industry_concepts_names)))
            
            # 定义板块个股所在概念板块的数据表industry_stocks_concepts，其具有两列，列名为['concept_code', 'concept_name']，索引为板块个股代码列表industry_stocks_codes，所有值均初始化为缺失值np.nan
            industry_stocks_concepts = pd.DataFrame(np.nan, columns=['concept_code', 'concept_name'], index=industry_stocks_codes)
            # 对于板块个股所在概念板块信息字典industry_concepts_info所包含的键值对，即板块个股所在概念板块代码industry_concept_code: 板块个股所在概念板块名称industry_concept_name
            for industry_concept_code, industry_concept_name in industry_concepts_info.items():
                # 对于板块个股所在概念板块代码industry_concept_code在日期date时对应的板块个股所在概念板块的个股代码列表get_concept_stocks()中的板块个股所在概念板块的各个股代码industry_concept_stocks_codes
                for industry_concept_stock_code in get_concept_stocks(industry_concept_code, date=end_date):
                    #  若if板块个股所在概念板块的某个股代码industry_concept_stocks_codes属于in板块个股代码列表industry_stocks_codes
                    if industry_concept_stock_code in industry_stocks_codes:
                        # 板块个股所在概念板块的数据表industry_stocks_concepts中，板块个股所在概念板块某个股代码industry_concept_stock_code作为索引对应的样本从缺失值向量[np.nan, np.nan]更新为[industry_concept_info[0], industry_concept_info[1]]，即此样本的'concept_code'列值和'concept_name'列值分别更新为板块个股所在概念板块代码industry_concept_code和板块个股所在概念板块名称industry_concept_name
                        industry_stocks_concepts.loc[industry_concept_stock_code] = [industry_concept_code, industry_concept_name]
        
        # 板块个股流通市值整理
        # 传出在date时（从date到date时），代码在板块个股代码列表industry_stocks_codes中的板块个股的流通市值industry_stock_circulating_market_cap的表industry_stocks_circulating_market_cap
        #industry_stocks_circulating_market_cap = get_valuation(security=industry_stocks_codes, start_date=date, end_date=date, fields=['circulating_market_cap'], count=None)[['circulating_market_cap']] # 使用[['circulating_market_cap']]索引，生成数据表，有列名；使用['circulating_market_cap']索引，生成序列，无列名
        industry_stocks_circulating_market_cap = get_fundamentals(
            query(
                valuation.code.label('industry_stocks_codes'),
                valuation.circulating_market_cap
                ).filter(
                    income.code.in_(industry_stocks_codes)
                    ), 
            date=end_date
        ).set_index('industry_stocks_codes')
        
        # 将板块个股流通市值表industry_stocks_circulating_market_cap的流通市值'circulating_market_cap'列增加单位亿元'billion_yuan'
        industry_stocks_circulating_market_cap.rename(columns={'circulating_market_cap': 'circulating_market_cap/billion_yuan'}, inplace=True)
        # 将板块个股流通市值表industry_stocks_circulating_market_cap的索引设置为板块个股代码列表industry_stocks_codes
        industry_stocks_circulating_market_cap.index = industry_stocks_info.index
        
        # 按索引，即板块个股代码列表industry_stocks_codes合并上述表格，即板块个股信息列表industry_stocks_info，板块个股所在行业表industry_stocks_industries，板块个股流通市值表industry_stocks_circulating_market_cap，得到新的板块个股信息列表industry_stocks_info
        industry_stocks_info = industry_stocks_info.join([industry_stocks_industries, industry_stocks_concepts, industry_stocks_circulating_market_cap])
        #
        industry_stocks_info.index.name = 'stock_code'
        # 对板块个股信息列表industry_stocks_info按依据'circulating_market_cap/billion_yuan'列的值，在垂直axis=0方向，按照降序ascending=False排序，并替换原表格inplace=True，此列含义缺失值的样本放在最前序号位置na_position='first'
        industry_stocks_info.sort_values(
        by='circulating_market_cap/billion_yuan', 
        axis=0, 
        ascending=False,  
        inplace=True, 
        na_position='first'
        )
        
        # 返回板块个股信息列表industry_stocks_info
        self.industry_stocks_info = industry_stocks_info
        self.__stocks_codes = list(industry_stocks_info.index)

    # 获取因子数据表
        
    # 定义从开始日期start_date到结束日期end_date，个股列表为stocks_codes的因子信息表all_factors_info与数据字典factors_data取得函数get_all_factors_info_and_data
    def get_factors_info_and_data(self):
        # 传出列名为因子代码'factor'，因子名称'factor_intro'，类别'category'的因子信息表格all_factors_info，并将索引设置为因子代码列'factor'
        all_factors_info = get_all_factors().rename(columns={'factor': 'factor_code'}, inplace=False).set_index(keys='factor_code', inplace=False)
        self.all_factors_info = all_factors_info

        # 传出因子信息表格all_factors_info中的索引，即因子代码'factor'列列表
        factors_codes = all_factors_info.index
        # 对于因子代码列表factors_codes中的每个因子代码factors_code，并利用tqdm类进行进度展示
        # 定义因子数据表字典factors_data，其结构为{..., factor_code: factor_data, ...}
        factors_data = {}
        for factor_code in tqdm(factors_codes, desc='calling factors progress'):
            # 传出股票代码为stocks_codes，因子代码为factor_code，从start_date到end_date时的因子数据表factor_data
            factor_data = get_factor_values(securities=self.__stocks_codes, factors=factor_code, start_date=self.__start_date, end_date=self.__end_date, count=None)[factor_code]
            # 设置因子数据表factor_data的索引列名.index.name为'stocks_codes'
            factor_data.index.name = 'date'
            
            # 以每个样本sample为单位进行各种处理，因为，将各样本sample一起处理，相当于将各样本sample横向合并为一列，会使得它们丢失时间索引以及对应的时序信息。也可以对每个股票的因子时间序列factor_code进行处理，但是时序信息比较难预测
            # 缺失值处理
            # 最后进行
            pass
            
            # 极端值处理
            # 本文不采用异常值处理，原因如下：聚宽数据平台的数据质量比较好，因子极端值往往不代表异常值，即测量误差和数据录入错误，更多代表罕见但真实的情况，特别在银行板块中，国有六大行占据优势地位，表现在数据上就是极端值的生成，因此所有指标的数据分布都呈现明显的右偏。即使出现来自国有六大行的极端值，它们也只占数据的一小部分，舍弃它们对模型性能的提升可能是有限的，但可能因此误伤真实的罕见值，导致信息损失，并可能降低模型在这些情况下的性能。本文采取了Fama-French分位数差值处理，以及以决策树模型为基模型
            # 的集成学习模型，这些方法都对极端值稳健和不敏感。因此本文只处理明确的异常值，即百分位数的极值，而不处理极端值。
            # 对经过缺失值处理的因子数据表factor_data进行极端值处理，因子值所在分位数处于[0.01, 0.99]的不变，小于0.05的或大于0.95的替换为0.05分位数或0.95分位数inclusive=True，因子值为无穷的不变，对每个样本进行
            # 处理axis=1
            factor_data = factor_data.astype(float)
            factor_data = winsorize_med(factor_data, scale=5, inclusive=True, inf2nan=False, axis=1)
            
            # 分位数正态化处理
            # 本文不采用正态化处理，因为其会使数据丧失可解释性（是否是原数据的线性映射），改变了数据相对位置（数据比例是否不变），虽然对异常值不敏感但是会因此掩盖异常值
            # 定义分位数正态化转换器quantile_transformer，转换的分布为标准正态分布'normal'，分布转换方向为横向axis=1，即对样本进行正态化，随机种子为random_state
            #quantile_transformer = QuantileTransformer(output_distribution='normal', axis=1, random_state=random_state)
            #factor_data = quantile_transformer.fit(factor_data)
                    
            # 标准化处理
            # 常见的尺度消除方法有标准化，鲁棒缩放（Robust Scaling），最大绝对值缩放（Max-Abs Scaling），归一化和正态化。优缺点：可解释性（是否是原数据的线性映射），是否改变了数据相对位置（数据比例是否不变），异常值鲁棒（掩盖异常值），异常值敏感
            # 对经过异常值处理的因子数据表factor_data进行标准化处理，因子值为无穷的不变，对每个样本进行处理axis=1
            factor_data = standardlize(factor_data, inf2nan=False, axis=1)
            
            # 市值和行业中性化处理
            # 对经过异常值处理的因子数据表factor_data进行市值和行业中性化处理，其中个股市值指标采用流通市值的对数'ln_circulating_market_cap'，行业指标采用聚宽二级行业'jq_l2'（因为聚宽一级行业区分度不够。），指标时间为研究时间research_date，对每个样本进行处理axis=1，不填充缺失值fillna=None，添加常数项add_constant=True（以确保残差的均值为零,这符合中性化处理的
            # 目的，即去除了因子的系统性影响）。
            # 为了消除某些指标对众多因子的冗余信息（多重共线性）和广泛影响，以提取出因子的异质性信息，需要对这些指标进行中性化处理。而如果这些指标能够对众多因子存在广泛影响，那么这些指标一定是足够基本的个股属性，这些基本属性相互耦合嵌套，通过复杂的生成机制产生了众多因子数据。最为基础的两个个股属性为其行业和市值，代表个股的领域及其绩效。
            # 流通市值只考虑在市场上公开交易的股票,排除了限售股、国有股等不可自由交易的股票，在一些银行板块中，国有企业占比较高或者有大量限售股，因此使用流通市值可以更好地反映股票的实际交易状况和市场流动性
            # 在银行板块等其他大多数板块，市值的分布是右偏的，少数大市值公司可能会主导整个市场。取自然对数可以减少这种偏斜,取自然对数可以有效地减少偏斜度,使分布更加对称和正态。这是因为对数变换可以压缩高端值（大市值公
            # 司）之间的差距，同时拉伸低端值（小市值公司）之间的差距。通过取对数，我们可以使市值在市值中性化处理中更符合线性关系和正态分布的假设
            # 对于流通市值因子'circulating_market_cap'，其不应该消除自身，所以只进行行业中性化；其余因子市值和行业中性化。
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
            # 向因子数据表字典factors_data增加键位因子代码为factor_code时对应的值因子数据表factor_data
            factors_data[factor_code] = factor_data
            
        # 返回因子数据表字典factors_data
        return factors_data
    
# 获取其他基本面因子数据

# 定义从开始日期start_date到结束日期end_date，个股列表为stocks_codes，基本面指标所在表对象table的列表为tables的基本面数据字典fundamentals_data取得函数get_all_fundamentals_data
    def get_fundamentals_data(self, tables=[
        valuation, 
        indicator, 
        bank_indicator
    ]):
        
        # 定义根据基本面指标所在表对象table和其属性字典table_attributes_dict的某键，即某属性名称fundamental_name返回总过滤条件字典filters的总过滤条件字典取得函数get_filters
        def get_attributes_filters(table, attribute_name):
            # 将“列表属性字典的某键，即某属性名称attribute_name不属于各表table的无关属性名称列表attributes_name_to_delete”的布尔值传出为原始过滤条件original_filter
            original_filter = attribute_name not in attributes_names_to_delete
            # 当基本面指标所在表对象table是市值表valuation时
            if table is valuation:
                # 将“列表属性字典的某键，即某属性名称attribute_name包含'_ratio'，即包含比率基本面指标属性”的布尔值传出为额外过滤条件extra_filter
                extra_filter = '_ratio' in attribute_name
            # 除此之外
            else:
                # 将真值True传出为额外过滤条件extra_filter
                extra_filter = True
            # 将原始过滤条件original_filter和额外过滤条件extra_filter的析取命传出为总过滤条件字典filters
            attributes_filters = original_filter and extra_filter
            # 返回总过滤条件字典filters
            return attributes_filters
        
        def get_query_params(table, date):
            query_params={}
            
            if table is bank_indicator:
                stock_code_filter = (table.code.in_(self.__stocks_codes))
                columns_filter = True
                query_params['filter'] = stock_code_filter & columns_filter
                query_params['date'] = None
                query_params['statDate'] = date.year
            else:
                stock_code_filter = (table.code.in_(self.__stocks_codes))
                columns_filter = True
                query_params['filter'] = stock_code_filter & columns_filter
                query_params['date'] = date
                query_params['statDate'] = None
        
            return query_params
        
        # 定义各基本面数据表字典fundamentals_data，其元素为各时期dates的fundamentals中的各基本面指标的基本面数据表fundamental_data
        fundamentals_data = {}
        # 定义各基本面指标所在表对象table的基本面指标字典fundamentals_dict，对于其中的键值对，键为基本面指标名称'fundamental'，值为对应的各表table中的基本面指标对象fundamental
        tables_and_fundamentals_dict = {}
        # 定义各表table的无关属性名称列表attributes_name_to_delete
        
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
            'stat_year'
            'pubDate',
            'statDate', 
            '__table__', 
            '_sa_class_manager', 
            '__init__', 
            '__mapper__'
        ]
        
        # 对于基本面指标所在表对象的列表为tables中各基本面指标所在表对象table
        for table in tables:
            # 将基本面指标所在表对象table的属性vars()，转化为字典dict()，并传出为基本面指标所在表对象table的属性字典table_attributes_dict，对于其中的键值对，键为属性名称attribute_name，值为属性对象attribute
            table_attributes_dict = dict(vars(table))
            # 对于基本面指标所在表对象table的属性字典table_attributes_dict其中的键值对，即属性名称attribute_name: 属性对象attribute
            table_and_fundamentals_dict = {attribute_name: {'table': table, 'fundamental': attribute} 
                                for attribute_name, attribute in table_attributes_dict.items()
                                # 筛选其中属性名称attribute_name满足总过滤条件字典filters的属性名称attribute_name: 属性对象attribute，作为宏观名称macro_name: 宏观对象macro字典，返回对应键值对组成的宏观指标所在表对象table的宏观字典table_macros_dict
                                if get_attributes_filters(table, attribute_name)
                                }
            # 将基本面指标所在表对象table的属性字典table_attributes_dict合并入各基本面指标所在表对象table的基本面指标字典fundamentals_dict
            tables_and_fundamentals_dict.update(table_and_fundamentals_dict)
        
        # 对于基本面指标列表fundamentals中的各基本面因子fundamental
        for fundamental_name, fundamental_info in tqdm(tables_and_fundamentals_dict.items(), desc='calling table progress'):
        # 对于宏观指标列表macros中的各宏观因子fundamental
            table, fundamental = fundamental_info['table'], fundamental_info['fundamental']
            # 定义某基本面数据表的列表fundamental_data_list，其元素为各时期date的某基本面指标fundamental的基本面数据表fundamental_data
            fundamental_data_list = []
            # 对于时期列表dates中的各时期date
            for date in self.dates:
                # 定义某时期基本面数据表fundamental_data，请求query数据表包含'stock_code'列，表示个股的代码，fundamental列表示对应个股的基本面指标fundament
                # -al，要求.filter'stock_code'列的个股代码indicator.code属于.in_个股代码列表stocks_codes，查询日期为date
                query_params = get_query_params(table, date)
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
                
                # 定义个股代码数据表stocks_codes_df，其只含有'stock_code'列，取值为个股代码列表stocks_codes
                stocks_codes_df = pd.DataFrame({'stock_code': self.__stocks_codes})
                # 将基本面数据表fundamental_data左连接个股代码数据表stocks_codes_df，以补全某时期date的基本面数据表fundamental_data中由于不在市而未返回的个股代码，并将这些未上市的个股某时期date的基本面数据补全为缺失值
                fundamental_data = pd.merge(stocks_codes_df, fundamental_data, on='stock_code', how='left')
                # 删除某时期date的基本面数据表fundamental_data的'stock_code'列，只剩下fundamental列
                fundamental_data.drop('stock_code', axis=1, inplace=True)
                # 将基本面数据表fundamental_data进行转置，将其仅有的'fundamental'列转置为一个样本，样本每一列为某时期date的每个个股的基本面数据
                fundamental_data = fundamental_data.T
                # 将某时期date的基本面数据表fundamental_data添加到某基本面数据表列表fundamental_data_list中
                fundamental_data_list.append(fundamental_data)
            # 纵向axis=1合并某基本面数据表的列表fundamental_data_list中，dates中各时期的基本面数据表fundamental_data
            fundamental_data = pd.concat(fundamental_data_list, axis=0)
            # 为dates中各时期的基本面数据表fundamental_data添加纵向索引dates代表样本的时期date
            fundamental_data.index = self.dates
            # 将索引重命名为'date'
            fundamental_data.index.name = 'date'
            # 将fundamental_data的列名.columns[i]命名为对应个股名称stocks_codes[i]
            fundamental_data.columns = self.__stocks_codes
            
            # 极端值处理
            fundamental_data = fundamental_data.astype(float)
            fundamental_data = winsorize_med(fundamental_data, scale=5, inclusive=True, inf2nan=False, axis=1)
            # 标准化处理处理
            fundamental_data = standardlize(fundamental_data, inf2nan=False, axis=1)
            # 中性化处理
            fundamental_data = neutralize(
                fundamental_data, 
                how=['jq_l2', 'ln_circulating_market_cap'], 
                date=self.__end_date,
                axis=1, 
                fillna=None, 
                add_constant=True
            )
            
            # 将各时期dates的fundamentals中的某基本面指标fundamental的基本面数据表fundamental_data作为值添加到各基本面数据表字典fundamentals_data，键设定为基本面指标名称fundamental_name
            fundamentals_data[fundamental_name] = fundamental_data
        
        # 删除wacc中的计算参数，防止过拟合
        fundamentals_data.pop('interest_bearing_liabilities')
        fundamentals_data.pop('interest_bearing_liabilities_interest_rate')
        
        # 返回各基本面数据表字典fundamentals_data
        return fundamentals_data

    # 获取宏观因子数据

    # 定义从开始日期start_date到结束日期end_date，个股列表为stocks_codes，宏观指标所在表对象table的列表为tables的宏观数据字典macros_data取得函数get_all_macros_data
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
        
        # 定义根据宏观指标所在表对象table和其属性字典table_attributes_dict的某键，即某属性名称macro_name返回总过滤条件字典filters的总过滤条件字典取得函数get_filters
        def get_attributes_filters(table, attribute_name):
            # 将“列表属性字典的某键，即某属性名称attribute_name不属于各表table的无关属性名称列表attributes_name_to_delete”的布尔值传出为原始过滤条件original_filter
            original_filter = attribute_name not in attributes_names_to_delete
            # 挑选属于银行业的属性，而非金融业甚至社会的属性，防止银行业的信息被稀释
            if table is macro.MAC_RMB_EXCHANGE_RATE:
                # 将“列表属性字典的某键，即某属性名称attribute_name包含'_ratio'，即包含比率宏观指标属性”的布尔值传出为额外过滤条件extra_filter
                extra_filter = attribute_name in ['spot_sell', 'cash_offer_prc', 'safe_prc', 'bank_reduced_prc']
            elif table is macro.MAC_LEND_RATE:
                extra_filter = attribute_name in ['interest_rate']
            elif table is macro.MAC_MONEY_SUPPLY_MONTH:
                extra_filter = attribute_name in ['m0', 'm0_yoy', 'm1', 'm1_yoy', 'm2', 'm2_yoy']
            elif table is macro.MAC_MONEY_SUPPLY_YEAR:
                extra_filter = attribute_name not in ['m0', 'm0_yoy', 'm1', 'm1_yoy', 'm2', 'm2_yoy']
            elif table is macro.MAC_CURRENCY_STATE_YEAR:
                # 中央银行影响（广义）商业银行的动机与效果
                extra_filter = attribute_name in [
                    'foreign_assets',
                    'foreign_exchange',
                    'money_gold',
                    'government_claim',
                    'bank_claim',
                    'other_finance_claim',
                    'non_finance_claim',
                    'reserve_money',
                    'currency_issue',
                    'bank_deposit',
                    'other_finance_deposit',
                    'non_reserve_finance_deposit',
                    'bond_issue',
                    'government_deposit',
                    'owned_capital'
                ]
            elif table is macro.MAC_OTHER_DEPOSIT:
                # 其他存款性公司与商业银行竞争（狭义）的动机与效果
                extra_filter = attribute_name in [
                    'reserve_assets', 
                    'reserve_deposit',
                    'cash_in_vault',
                    'government_claim',
                    'government_claim',
                    'central_bank_claim',
                    'other_claim',
                    'other_finance_claim',
                    'non_finance_claim',
                    'other_resident_claim',
                    'non_finance_liability',
                    'corporate_demand_deposit',
                    'corporate_time_deposit',
                    'personal_deposit',
                    'other_non_finance_liability',
                    'central_bank_liability',
                    'other_deposit_liability',
                    'other_finance_liability',
                    'include_broad_money',
                    'bond_issue', 
                    'paid_in_capital'
                ]
            elif table is macro.MAC_SOCIAL_SCALE_FINANCE:
                extra_filter = attribute_name in [
                    'foreign_loan', 
                    'rmb_loan',
                    'entrust_loan',
                    'trust_loan',
                    'out_fulfilled_scale',
                    'corporate_bond_scale',
                    'non_finance_scale'
                ]
            elif table is macro.MAC_STK_MARKET:
                extra_filter = attribute_name in [
                    'total_trade_amount', 
                    'xshg_avg_pe',
                    'xshe_avg_pe',
                    'xshg_avg_turnover',
                    'xshe_avg_turnover',
                    #'treasury_bond_issue', #以下债券发行数据截止2006年
                    #'company_bond_issue'
                    #'bond_amount', 
                    #'treasury_bond_spot_amount',
                    #'treasury_bond_repurchase_amount'
                ]
            elif table is macro.MAC_GOLD_FOREIGN_RESERVE:
                extra_filter = attribute_name in ['gold', 'foreign']
            # 除此之外
            else:
                # 将真值True传出为额外过滤条件extra_filter
                extra_filter = True
            # 将原始过滤条件original_filter和额外过滤条件extra_filter的析取命传出为总过滤条件字典filters
            attributes_filters = original_filter & extra_filter
            # 返回总过滤条件字典filters
            return attributes_filters
        
        def get_query_params(table, date):
            query_params = {}

            if table is macro.MAC_RMB_EXCHANGE_RATE:
                date_filter = (table.day == date)
                columns_filter = (table.currency_id == 40)
                query_params['filter'] = date_filter & columns_filter
            elif table is macro.MAC_LEND_RATE:
                if macro_name == 'interest_rate_O/N':
                    term_id = '20'
                elif macro_name == 'interest_rate_3m':
                    term_id = '3'
                elif macro_name == 'interest_rate_1y':
                    term_id = '12'
                date_filter = (table.day == date)
                columns_filter = (table.currency_id == '1') & (table.market_id == '5') & (table.term_id == term_id)
                query_params['filter'] = date_filter & columns_filter
            elif table is macro.MAC_MONEY_SUPPLY_MONTH:
                date_filter = (table.stat_month == date.strftime('%Y-%m-%d')[0: 7])
                columns_filter = True
                query_params['filter'] = date_filter & columns_filter
            elif table is macro.MAC_GOLD_FOREIGN_RESERVE:
                date_filter = (table.stat_date == date.strftime('%Y-%m-%d')[0: 7])
                columns_filter = True
                query_params['filter'] = date_filter & columns_filter
            elif table is macro.MAC_MONEY_SUPPLY_YEAR \
            or table is macro.MAC_CURRENCY_STATE_YEAR \
            or table is macro.MAC_OTHER_DEPOSIT \
            or table is macro.MAC_SOCIAL_SCALE_FINANCE \
            or table is macro.MAC_STK_MARKET:
                date_filter = (table.stat_year == date.year)
                columns_filter = True
                query_params['filter'] = date_filter & columns_filter
            else:
                date_filter = (table.day == date)
                columns_filter = True
                query_params['filter'] = date_filter & columns_filter
            return query_params
        
        # 定义各宏观数据表字典macros_data，其元素为各时期dates的macros中的各宏观指标的宏观数据表macro_data
        macros_data = {}
        
        # 定义各宏观指标所在表对象table的宏观指标字典macros_dict，对于其中的键值对，键为宏观指标名称'macro'，值为对应的各表table中的宏观指标对象macro
        tables_and_macros_dict = {}
        
        # 定义各表table的无关属性名称列表attributes_name_to_delete
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
            'stat_year'
            'pubDate',
            'statDate', 
            '__table__', 
            '_sa_class_manager', 
            '__init__', 
            '__mapper__'
        ]
        
        # 对于宏观指标所在表对象的列表为tables中各宏观指标所在表对象table
        for table in tables:
            # 将宏观指标所在表对象table的属性vars()，转化为字典dict()，并传出为宏观指标所在表对象table的属性字典table_attributes_dict，对于其中的键值对，键为属性名称attribute_name，值为属性对象attribute
            table_attributes_dict = dict(vars(table))
            # 对于宏观指标所在表对象table的属性字典table_attributes_dict其中的键值对，即属性名称attribute_name: 属性对象attribute
            table_and_macros_dict = {
                attribute_name: {'table': table, 'macro_sql': attribute} 
                for attribute_name, attribute in table_attributes_dict.items()
                # 筛选其中属性名称attribute_name满足总过滤条件字典filters的属性名称attribute_name: 属性对象attribute，作为宏观名称macro_name: 宏观对象macro字典，返回对应键值对组成的宏观指标所在表对象table的宏观字典table_macros_dict
                if get_attributes_filters(table, attribute_name)
            }
            
            # 将宏观指标所在表对象table的属性字典table_attributes_dict合并入各宏观指标所在表对象table的宏观指标字典macros_dict
            tables_and_macros_dict.update(table_and_macros_dict)
            
        # 将银行间拆借利率interest_rate拆分短期，中期和长期利率
            if table is macro.MAC_LEND_RATE:
                tables_and_macros_dict['interest_rate_1y'] = \
                tables_and_macros_dict['interest_rate_3m'] = \
                tables_and_macros_dict['interest_rate_O/N'] = \
                tables_and_macros_dict.pop('interest_rate')
            
        for macro_name, macro_info in tqdm(tables_and_macros_dict.items(), desc='calling table progress'):
        # 对于宏观指标列表macros中的各宏观因子fundamental
            table, macro_sql = macro_info['table'], macro_info['macro_sql']
            # 定义某宏观数据表的列表macro_data_list，其元素为各时期date的某宏观指标macro的宏观数据表macro_data
            macro_data_list = []
            # 对于时期列表dates中的各时期date
            for date in self.dates:
                # 定义某时期宏观数据表macro_data，请求query数据表包含'stock_code'列，表示个股的代码，macro列表示对应个股的宏观指标fundament
                # -al，要求.filter'stock_code'列的个股代码indicator.code属于.in_个股代码列表stocks_codes，查询日期为date
                macro_data = macro.run_query(
                    query(
                        macro_sql
                    ).filter(
                        get_query_params(table, date)['filter']
                    )
                )
                
                # 将宏观数据表macro_data进行转置，将其仅有的'macro'列转置为一个样本，样本每一列为某时期date的每个个股的宏观数据
                macro_data = macro_data.T
                # 将某时期date的宏观数据表macro_data添加到某宏观数据表列表macro_data_list中
                macro_data_list.append(macro_data)
            # 纵向axis=1合并某宏观数据表的列表macro_data_list中，dates中各时期的宏观数据表macro_data
            macro_data = pd.concat(macro_data_list, axis=0)
            # 为dates中各时期的宏观数据表macro_data添加纵向索引dates代表样本的时期date
            macro_data.index = self.dates
            # 将索引重命名为'date'
            macro_data.index.name = 'date'
            
            if macro_name == 'bond_amount' \
            or macro_name == 'treasury_bond_spot_amount' \
            or macro_name =='treasury_bond_repurchase_amount':
                # 储存原始数据以准备组合成非国债发行额数据non_treasury_bond_amount
                macros_data[macro_name] = macro_data
                if macro_name != 'treasury_bond_repurchase_amount':
                    continue
                else:
                    # 组合非国债发行额数据non_treasury_bond_amount
                    macro_data = macros_data.pop('bond_amount') \
                    - (macros_data.pop('treasury_bond_spot_amount') + macros_data.pop('treasury_bond_repurchase_amount'))
                    macro_name = 'non_treasury_bond_amount'
            
            # 极端值处理
            macro_data = macro_data.astype(float)
            macro_data = winsorize_med(macro_data, scale=5, inclusive=True, inf2nan=False, axis=0)
            # 标准化处理处理
            macro_data = standardlize(macro_data, inf2nan=False, axis=0)
            
            # 将各时期dates的macros中的某宏观指标macro的宏观数据表macro_data作为值添加到各宏观数据表字典macros_data，键设定为宏观指标名称macro_name
            macros_data[macro_name] = macro_data
            
        # 返回各宏观数据表字典macros_data
        return macros_data
    
    # 获取资金流因子

    def get_money_flows_data(self):
    
        money_flows_raw_codes = [
            'change_pct',
            'net_amount_main', 
            'net_pct_main',
            'net_pct_xl',
            'net_pct_l',
            'net_pct_m',
            'net_pct_s'
        ]
        
        money_flows_used_codes = [
            'change_pct', 
            'net_pct_main',
            'net_pct_xl',
            'net_pct_l',
            'net_pct_m',
            'net_pct_s',
            'amount' # 构造的因子放在最后
        ]
        
        money_flows_merged_codes = money_flows_raw_codes + [
            money_flow_code 
            for money_flow_code in money_flows_used_codes 
            if money_flow_code not in money_flows_raw_codes
        ]
        
        money_flows_data = {}
        for money_flow_code in tqdm(money_flows_merged_codes, desc='calling money flows progress'):
            # 对于使用的因子
            if money_flow_code in money_flows_raw_codes:
                money_flow_data_list = []
                for date in self.dates:
                    # 定义某时期基本面数据表fundamental_data，请求query数据表包含'stock_code'列，表示个股的代码，fundamental列表示对应个股的基本面指标fundament
                    # -al，要求.filter'stock_code'列的个股代码indicator.code属于.in_个股代码列表stocks_codes，查询日期为date
                    money_flow_data = get_money_flow(security_list=self.__stocks_codes, start_date=date, end_date=date, fields=['sec_code', money_flow_code], count=None)
                    money_flow_data.rename(columns={'sec_code': 'stock_code'}, inplace=True)
                    # 如果时期参数为date=date时基本面数据表fundamental_datafundamental_data为空，则表明fundamental来自银行业财务指标表bank_indicator，此时将时期参数改为statDate=date.year，即用时期date的年份查询报告日期statDate
                    if money_flow_data.empty:
                        money_flow_data = pd.DataFrame()
                        money_flow_data['stock_code'] = get_money_flow(
                            security_list=self.__stocks_codes, 
                            start_date=date, 
                            end_date=date, 
                            fields=['sec_code'], 
                            count=None
                        )
                        money_flow_data[money_flow_code] = np.nan

                    # 定义个股代码数据表stocks_codes_df，其只含有'stock_code'列，取值为个股代码列表stocks_codes
                    stocks_codes_df = pd.DataFrame({'stock_code': self.__stocks_codes})
                    # 将基本面数据表fundamental_data左连接个股代码数据表stocks_codes_df，以补全某时期date的基本面数据表fundamental_data中由于不在市而未返回的个股代码，并将这些未上市的个股某时期date的基本面数据补全为缺失值
                    money_flow_data = pd.merge(stocks_codes_df, money_flow_data, on='stock_code', how='left')
                    # 删除某时期date的基本面数据表fundamental_data的'stock_code'列，只剩下fundamental列
                    money_flow_data.drop('stock_code', axis=1, inplace=True)
                    # 将基本面数据表fundamental_data进行转置，将其仅有的'fundamental'列转置为一个样本，样本每一列为某时期date的每个个股的基本面数据
                    money_flow_data = money_flow_data.T
                    # 将某时期date的基本面数据表fundamental_data添加到某基本面数据表列表fundamental_data_list中
                    money_flow_data_list.append(money_flow_data)
                # 纵向axis=1合并某基本面数据表的列表fundamental_data_list中，dates中各时期的基本面数据表fundamental_data
                money_flow_data = pd.concat(money_flow_data_list, axis=0)
                # 为dates中各时期的基本面数据表fundamental_data添加纵向索引dates代表样本的时期date
                money_flow_data.index = self.dates
                # 将索引重命名为'date'
                money_flow_data.index.name = 'date'
                # 将fundamental_data的列名.columns[i]命名为对应个股名称stocks_codes[i]
                money_flow_data.columns = self.__stocks_codes
                
                # 为构造因子做准备
                money_flows_data[money_flow_code] = money_flow_data
            
            # 对于构造因子
            if money_flow_code in money_flows_used_codes and money_flow_code == 'amount':
                # 构造成交量因子
                money_flows_data[money_flow_code] = money_flows_data.pop('net_amount_main') / money_flows_data['net_pct_main']
        
            # 极端值处理
            money_flow_data = money_flow_data.astype(float)
            money_flow_data = winsorize_med(money_flow_data, scale=5, inclusive=True, inf2nan=False, axis=1)
            # 标准化处理处理
            money_flow_data = standardlize(money_flow_data, inf2nan=False, axis=1)
            # 中性化处理
            money_flow_data = neutralize(
                money_flow_data, 
                how=['jq_l2', 'ln_circulating_market_cap'], 
                date=self.__end_date, axis=1, 
                fillna=None, 
                add_constant=True
            )
            
            # 将各时期dates的fundamentals中的某基本面指标fundamental的基本面数据表fundamental_data作为值添加到各基本面数据表字典fundamentals_data，键设定为基本面指标名称fundamental_name
            money_flows_data[money_flow_code] = money_flow_data
            
        # 返回各基本面数据表字典fundamentals_data
        return money_flows_data
    
    # 获取股票融资融券因子

    def get_securities_margins_data(self):
        
        securities_margins_data = {}
        
        securities_margins_raw_list = [
            'fin_buy_value',
            'fin_refund_value',
            'sec_sell_value',
            'sec_refund_value'
        ]
        
        securities_margins_list = [
            'fin_sec_value',
            'fin_sec_ratio',
            'fin_refund_ratio',
            'sec_refund_ratio'
        ]
        
        securities_margins_merged_list = securities_margins_raw_list + [
            securities_margin 
            for securities_margin in securities_margins_list
            if securities_margin not in securities_margins_raw_list
        ]
        for securities_margin_name in tqdm(securities_margins_merged_list, desc='calling securities_margins progress'):
            # 对于非构造的因子
            if securities_margin_name in securities_margins_raw_list:
            # 定义某基本面数据表的列表fundamental_data_list，其元素为各时期date的某基本面指标fundamental的基本面数据表fundamental_data
                securities_margin_data_list = []
                # 对于时期列表dates中的各时期date
                for date in self.dates:
                    # 定义某时期基本面数据表fundamental_data，请求query数据表包含'stock_code'列，表示个股的代码，fundamental列表示对应个股的基本面指标fundament
                    # -al，要求.filter'stock_code'列的个股代码indicator.code属于.in_个股代码列表stocks_codes，查询日期为date
                    securities_margin_data = get_mtss(self.__stocks_codes, start_date=date, end_date=date, fields=['sec_code', securities_margin_name])
                    securities_margin_data.rename(columns={'sec_code': 'stock_code'}, inplace=True)

                    # 定义个股代码数据表stocks_codes_df，其只含有'stock_code'列，取值为个股代码列表stocks_codes
                    stocks_codes_df = pd.DataFrame({'stock_code': self.__stocks_codes})
                    # 将基本面数据表fundamental_data左连接个股代码数据表stocks_codes_df，以补全某时期date的基本面数据表fundamental_data中由于不在市而未返回的个股代码，并将这些未上市的个股某时期date的基本面数据补全为缺失值
                    securities_margin_data = pd.merge(stocks_codes_df, securities_margin_data, on='stock_code', how='left')
                    # 删除某时期date的基本面数据表fundamental_data的'stock_code'列，只剩下fundamental列
                    securities_margin_data.drop('stock_code', axis=1, inplace=True)
                    # 将基本面数据表fundamental_data进行转置，将其仅有的'fundamental'列转置为一个样本，样本每一列为某时期date的每个个股的基本面数据
                    securities_margin_data = securities_margin_data.T
                    # 将某时期date的基本面数据表fundamental_data添加到某基本面数据表列表fundamental_data_list中
                    securities_margin_data_list.append(securities_margin_data)
                # 纵向axis=1合并某基本面数据表的列表fundamental_data_list中，dates中各时期的基本面数据表fundamental_data
                securities_margin_data = pd.concat(securities_margin_data_list, axis=0)
                # 为dates中各时期的基本面数据表fundamental_data添加纵向索引dates代表样本的时期date
                securities_margin_data.index = self.dates
                # 将索引重命名为'date'
                securities_margin_data.index.name = 'date'
                # 将fundamental_data的列名.columns[i]命名为对应个股名称stocks_codes[i]
                securities_margin_data.columns = self.__stocks_codes
                
                # 为构造因子做准备
                # 对于raw_list和list的重叠部分
                if securities_margin_name in set(securities_margins_raw_list) & set(securities_margins_list):
                    securities_margin_name += '_original'

                securities_margins_data[securities_margin_name] = securities_margin_data
            
            # 对于构造的因子
            if securities_margin_name in securities_margins_list:
                if securities_margin_name == 'fin_sec_value':
                    pass
                # 构造融资融券比例因子
                elif securities_margin_name == 'fin_sec_ratio':
                    securities_margins_data[securities_margin_name] = securities_margins_data['fin_buy_value'] \
                    / (securities_margins_data['sec_sell_value'] + securities_margins_data['fin_buy_value'])
                # 构造还资还券比例因子
                elif securities_margin_name == 'fin_refund_ratio':
                    securities_margins_data[securities_margin_name] = securities_margins_data.pop('fin_refund_value') \
                    / securities_margins_data.pop('fin_buy_value')
                elif securities_margin_name == 'sec_refund_ratio':
                    securities_margins_data[securities_margin_name] = securities_margins_data.pop('sec_refund_value') \
                    / securities_margins_data.pop('sec_sell_value')
                    
                # 极端值处理
                securities_margin_data = securities_margin_data.astype(float)
                securities_margin_data = winsorize_med(securities_margin_data, scale=5, inclusive=True, inf2nan=False, axis=1)
                # 标准化处理处理
                securities_margin_data = standardlize(securities_margin_data, inf2nan=False, axis=1)
                # 中性化处理
                securities_margin_data = neutralize(
                    securities_margin_data, 
                    how=['jq_l2', 'ln_circulating_market_cap'],
                    date=self.__end_date, 
                    axis=1, 
                    fillna=None, 
                    add_constant=True
                )
                # 将各时期dates的fundamentals中的某基本面指标fundamental的基本面数据表fundamental_data作为值添加到各基本面数据表字典fundamentals_data，键设定为基本面指标名称fundamental_name
                securities_margins_data[securities_margin_name] = securities_margin_data
            
        # 返回各基本面数据表字典fundamentals_data
        return securities_margins_data
    
    # 获取龙虎榜因子数据

    def get_billboards_data(self):
    
        billboards_codes = [
            'amount_ratio',
            'abnormal_code'
        ]
        
        billboards_data = {}
        for billboard_code in billboards_codes:
            if billboard_code == 'amount_ratio' or billboard_code == 'abnormal_code':
                # 对于使用的因子
                billboard_data_list = []
                for date in tqdm(self.dates, desc='calling billboards_data'):
                    # 定义某时期基本面数据表fundamental_data，请求query数据表包含'stock_code'列，表示个股的代码，fundamental列表示对应个股的基本面指标fundament
                    # -al，要求.filter'stock_code'列的个股代码indicator.code属于.in_个股代码列表stocks_codes，查询日期为date
                    billboard_data = get_billboard_list(stock_list=self.__stocks_codes, start_date=date, end_date=date)
                    billboard_data.rename(columns={'code': 'stock_code'}, inplace=True)
                    billboard_data = billboard_data.loc[billboard_data['rank'] == 0, ['stock_code', 'abnormal_code', 'total_value']]
                    # 对相同code列和abnormal_name列的样本的total_value进行求和汇总
                    billboard_data = billboard_data.groupby(['stock_code', 'abnormal_code'], as_index=False).agg({'total_value': 'sum'}).reset_index(drop=True)
                    # 删除code列相同的样本
                    billboard_data = billboard_data.drop_duplicates(subset='stock_code', keep='first')

                    if billboard_code == 'amount_ratio':
                        billboard_data.drop(['abnormal_code'], axis=1, inplace=True)

                    elif billboard_code == 'abnormal_code':
                        billboard_data.drop(['total_value'], axis=1, inplace=True)

                    if billboard_data.empty:
                        billboard_data = pd.DataFrame()
                        billboard_data['stock_code'] = get_billboard_list(stock_list=self.__stocks_codes, start_date=date, end_date=date)['code']
                        billboard_data[billboard_code] = np.nan

                    # 定义个股代码数据表stocks_codes_df，其只含有'stock_code'列，取值为个股代码列表stocks_codes
                    stocks_codes_df = pd.DataFrame({'stock_code': self.__stocks_codes})
                    # 将基本面数据表fundamental_data左连接个股代码数据表stocks_codes_df，以补全某时期date的基本面数据表fundamental_data中由于不在市而未返回的个股代码，并将这些未上市的个股某时期date的基本面数据补全为缺失值
                    billboard_data = pd.merge(stocks_codes_df, billboard_data, on='stock_code', how='left')
                    # 删除某时期date的基本面数据表fundamental_data的'stock_code'列，只剩下fundamental列
                    billboard_data.drop(['stock_code'], axis=1, inplace=True)
                    # 将基本面数据表fundamental_data进行转置，将其仅有的'fundamental'列转置为一个样本，样本每一列为某时期date的每个个股的基本面数据
                    billboard_data = billboard_data.T
                    # 将某时期date的基本面数据表fundamental_data添加到某基本面数据表列表fundamental_data_list中
                    billboard_data_list.append(billboard_data)
                
                # 纵向axis=1合并某基本面数据表的列表fundamental_data_list中，dates中各时期的基本面数据表fundamental_data
                billboard_data = pd.concat(billboard_data_list, axis=0)
            else:
                billboard_data = billboards_data[billboard_code]
            # 为dates中各时期的基本面数据表fundamental_data添加纵向索引dates代表样本的时期date
            billboard_data.index = self.dates
            # 将索引重命名为'date'
            billboard_data.index.name = 'date'
            # 将fundamental_data的列名.columns[i]命名为对应个股名称stocks_codes[i]
            billboard_data.columns = self.__stocks_codes
            
            if billboard_code == 'abnormal_code':
                # 为构造异常虚拟变量做准备
                # 构建abnormal_code列取值集合
                abnormal_codes = set(pd.concat([pd.Series(billboard_data[stock_code].value_counts().index) for stock_code in self.__stocks_codes]))
                # 将取值集合内的元素转换为字符串
                abnormal_codes = [billboard_code + '_' + str(int(abnormal_code)) for abnormal_code in abnormal_codes]
                billboards_codes += abnormal_codes
                
                abnormal_codes_dummies_dict = {}
                for stock_code in self.__stocks_codes:
                    # abnormal_code_dummy列名为abnormal_code列的取值种类列表，每一列代表各样本的abnormal_code列是否取对应abnormal_code值
                    stock_abnormal_codes_dummy = pd.get_dummies(billboard_data[stock_code])
                    # 将每个列名都转换为字符串
                    stock_abnormal_codes_dummy.rename(columns=lambda x: billboard_code + '_' + str(int(x)), inplace=True)
                    if stock_abnormal_codes_dummy.empty:
                        abnormal_codes_dummies_dict[stock_code] = pd.DataFrame(0, index=self.dates, columns=abnormal_codes)
                    else:
                        # 将stock_abnormal_codes_dummy的列名重新索引为包含缺失列名的完整列名列表，并将缺失列值填充为零
                        stock_abnormal_codes_dummy = stock_abnormal_codes_dummy.reindex(columns=abnormal_codes, fill_value=0)
                        abnormal_codes_dummies_dict[stock_code] = stock_abnormal_codes_dummy
                    
                for abnormal_code in abnormal_codes:
                    # abnormal_codes_dummies_dict字典里面有42张旧数据表，每个数据表有n列，将这些旧数据表重构成n个新数据表，即将第i个旧数据表的每列当做每个新数据表中的第i列
                    billboard_data = pd.concat([abnormal_codes_dummies_dict[stock_code][abnormal_code] for stock_code in self.__stocks_codes], axis=1)
                    billboards_data[abnormal_code] = billboard_data

                continue

            # 极端值处理
            billboard_data = billboard_data.astype(float)
            billboard_data = winsorize_med(billboard_data, scale=5, inclusive=True, inf2nan=False, axis=1)
            # 标准化处理处理
            billboard_data = standardlize(billboard_data, inf2nan=False, axis=1)
            # 中性化处理
            billboard_data = neutralize(
                billboard_data, 
                how=['jq_l2', 'ln_circulating_market_cap'], 
                date=self.__end_date, 
                axis=1, 
                fillna=None, 
                add_constant=True
            )
            
            if billboard_data.any().any():
                # 将各时期dates的fundamentals中的某基本面指标fundamental的基本面数据表fundamental_data作为值添加到各基本面数据表字典fundamentals_data，键设定为基本面指标名称fundamental_name
                billboards_data[billboard_code] = billboard_data
            
        # 返回各基本面数据表字典fundamentals_data
        return billboards_data

    # 获取行业因子数据

    # 定义在开始日期start_date和结束日期end_date，代码列表为stocks_codes的个股所在聚宽二级行业的数据表jq_l2_industry_data字典jq_l2_industries_data的取得函数get_concepts_data
    def get_industries_data(self):
        # 传出在日期date时，板块个股代码为stocks_codes的个股所在行业字典industry_stocks_industries
        industry_stocks_industries = get_industry(security=self.__stocks_codes, date=self.__end_date)
        # 传出个股所在行业字典industry_stocks_industries中各代码为stock_code的个股所在聚宽二级行业jq_l2'的代码集合industry_code和名称industry_name集合jq_l2_industry_code/name，过滤掉industry_stocks_industries[stock_code]字典中不存在'jq_l2'索引的
        # 情况
        jq_l2_industries_codes = {industry_stocks_industries[stock_code]['jq_l2']['industry_code'] for stock_code in industry_stocks_industries.keys() if industry_stocks_industries[stock_code].get('jq_l2')}
        jq_l2_industries_names = {industry_stocks_industries[stock_code]['jq_l2']['industry_name'] for stock_code in industry_stocks_industries.keys() if industry_stocks_industries[stock_code].get('jq_l2')}
        
        # 定义个股所在聚宽二级行业的数据表jq_l2_industry_data的字典jq_l2_industries_data，其键为个股所在聚宽二级行业的代码jq_l2_industry_code，各值为个股所在聚宽二级行业的数据表jq_l2_industry_data
        jq_l2_industries_data = {}
        # 对于传出个股所在行业代码集合jq_l2_industries_codes中的个股所在行业的各代码jq_l2_industry_code
        for jq_l2_industry_code in jq_l2_industries_codes:
            # 初始化个股所在行业的数据表jq_l2_industry_data，其具有两列，列名为个股代码列表stocks_codes，索引为开市时期列表dates，的所有哑变量均初始化为0，哑变量表示某代码为stock_code的个股在日期date是否属于代码为q_l2_industry_cod
            # -e的行业
            jq_l2_industry_data = pd.DataFrame(0, columns=self.__stocks_codes, index=self.dates)
            # 将个股所在聚宽二级行业的数据表jq_l2_industry_data的字典jq_l2_industries_data中键为个股所在聚宽二级行业代码jq_l2_industry_code的值传入为个股所在聚宽二级行业的数据表jq_l2_industry_data
            jq_l2_industries_data[jq_l2_industry_code] = jq_l2_industry_data
        
        # 对于开市时期列表dates中的每个日期date
        for date in tqdm(self.dates, desc='industries_data calling progress'):    
            # 对于在日期date时，板块个股代码为stocks_codes的个股所在行业字典get_industry(security=stocks_codes, date=date)的各键值对，某个股代码stock_code: 某个股行业信息字典stock_industries_info
            for stock_code, stock_industries_info in get_industry(security=self.__stocks_codes, date=date).items():
                # 尝试，若无错误，则某个股行业信息字典stock_industries_info不为空且有'jq_l2'索引，代表此个股在时期date时已经上市且其聚宽二级行业代码stock_jq_l2_industry_code不缺失
                try:
                    # 将某个股行业信息字典stock_industries_info的聚宽二级行业信息'jq_l2'的聚宽二级行业代码'industry_code'传入为某个股聚宽二级行业代码stock_jq_l2_industry_code
                    stock_jq_l2_industry_code = stock_industries_info['jq_l2']['industry_code']
                    # 将某个股的旧的聚宽二级行业代码stock_jq_l2_industry_code转换为新的聚宽二级行业代码stock_jq_l2_industry_code
                    if stock_jq_l2_industry_code == 'HY07001' and date < datetime.date(2021, 12, 13) < self.__end_date:
                        stock_jq_l2_industry_code = 'HY07101'
                    elif stock_jq_l2_industry_code == 'HY07002' and date < datetime.date(2021, 12, 13) < self.__end_date:
                        stock_jq_l2_industry_code = 'HY07102'
                    # 将个股所在聚宽二级行业的数据表字典jq_l2_industries_data中某个股所在聚宽二级行业代码jq_l2_industry_code对应的个股所在聚宽二级行业的数据表jq_l2_industry_data的某个股代码stock_code的对应列的索引为日期date哑变量传
                    # 传入为1，代表某代码为stock_code的个股在日期date是否属于代码为q_l2_industry_code的行业
                    jq_l2_industries_data[stock_jq_l2_industry_code] = jq_l2_industry_data
                    jq_l2_industry_data.loc[date, stock_code] = 1

                    # 极端值处理
                    jq_l2_industry_data = jq_l2_industry_data.astype(float)
                    jq_l2_industry_data = winsorize_med(jq_l2_industry_data, scale=5, inclusive=True, inf2nan=False, axis=1)
                    # 标准化处理处理
                    jq_l2_industry_data = standardlize(jq_l2_industry_data, inf2nan=False, axis=1)
                    # 中性化处理
                    jq_l2_industry_data = neutralize(jq_l2_industry_data, how=['jq_l2', 'ln_circulating_market_cap'], date=self.__end_date, axis=1, fillna=None, add_constant=True)

                # 若出现键索引错误KeyError，则某个股行业信息字典stock_industries_info为空或无'jq_l2'索引，代表此个股在时期date时还未上市或其聚宽二级行业代码stock_jq_l2_industry_code缺失
                except KeyError:
                    # 对于个股所在聚宽二级行业的代码集合jq_l2_industries_codes中的某个股所在聚宽二级行业的代码stock_jq_l2_industry_code
                    for stock_jq_l2_industry_code in jq_l2_industries_codes:
                        # 将个股所在聚宽二级行业的数据表字典jq_l2_industries_data中某个股所在聚宽二级行业代码jq_l2_industry_code对应的个股所在聚宽二级行业的数据表jq_l2_industry_data的某个股代码stock_code的对应列的索引为日期date哑变量传
                        # 传入为缺失值np.nan
                        jq_l2_industries_data[stock_jq_l2_industry_code].loc[date, stock_code] = np.nan
        
        # 返回个股所在聚宽二级行业的数据表字典jq_l2_industries_data
        return jq_l2_industries_data

    # 获取概念因子数据

    # 定义在开始日期start_date和结束日期end_date，代码列表为stocks_codes的个股所在概念板块的数据表concept_data的字典concepts_data的取得函数get_concepts_data
    def get_concepts_data(self):
        
        # 传出所有概念板块的信息数据表concepts，列名与信息如下：'index': 概念代码，name: 概念名称，start_date: 开始日期
        concepts = get_concepts()
        
        # 传出在结束日期end_date时，个股代码列表为stocks_codes的个股所在行业字典stocks_industries
        stocks_industries = get_industry(security=self.__stocks_codes, date=self.__end_date)
        # 传出个股所在行业字典stocks_industries中各代码为stock_code的个股所在聚宽一级'jq_l1'的代码code和名称name集合jq_l1_code/name，过滤掉industry_stocks_industries[stock_code]字典中不存在'jq_l2'索引的情况，作为个股所在概念板块关键词集合concepts_keywords传出，即将板块名称的设置为概念板块的关键词
        concepts_keywords = {stocks_industries[stock_code]['jq_l1']['industry_name'] for stock_code in stocks_industries.keys() if stocks_industries[stock_code].get('jq_l2')}
        
        # 定义个股所在概念板块代码列表concepts_code的列表concepts_codes_list
        concepts_codes_list = []
        concepts['start_date'] = concepts['start_date'].dt.date
        # 对于个股所在概念板块关键词列表concepts_keywords中的每个个股所在概念板块关键词concepts_keyword
        for concepts_keyword in concepts_keywords:
            # 将所有概念板块的信息数据表concepts中个股所在概念板块名称concepts['name']包含个股所在概念板块关键词.str.contains(concepts_keyword)，且个股所在概念板块创立时间pd.to_datetime(concepts['start_date'])早于结束时期date的概念板块代码.index传出为个股所在概念板块代码列表concepts_codes，并转化pd.DataFrame()成数据表
            concepts_codes = pd.DataFrame(concepts[concepts['name'].str.contains(concepts_keyword) & (concepts['start_date'] <= self.__end_date)].index)
            # 将个股所在概念板块代码数据表concepts_code传入.append()其列表concepts_codes_list
            concepts_codes_list.append(concepts_codes)
        # 将定义个股所在概念板块代码列表的数据表concepts_codes_list内各个股所在概念板块代码列表concepts_code合并为个股所在概念板块代码列表concepts_code
        concepts_codes = pd.concat(concepts_codes_list, axis=0)
        # 将个股所在概念板块代码数据表concepts_code的唯一一列.loc[:, 0]转换为列表list()
        concepts_codes = list(concepts_codes.loc[:, 0])
        
        # 定义个股所在概念板块的数据表concept_data的字典concepts_data，其键为个股所在概念板块代码concept_code，各值为个股所在概念板块的数据表concept_data
        concepts_data = {}
        # 对于个股所在概念板块代码数据表concepts_code中的各个股所在概念板块代码concept_code
        for concept_code in concepts_codes:
            # 定义个股所在概念板块的数据表concept_data，列名为个股代码列表stocks_codes，索引为开市时期列表dates，所有值均初始化为缺失值0，表示哑变量
            concept_data = pd.DataFrame(0, columns=self.__stocks_codes, index=self.dates)
            # 将个股所在概念板块的数据表concept_data的字典concepts_data中键为个股所在概念板块代码concept_code的值传入为个股所在概念板块的数据表concept_data
            concepts_data[concept_code] = concept_data
            
        # 对于开市时期列表dates中的每个日期date
        for date in tqdm(self.dates, desc='concepts calling progress'):
            # 对于个股所在概念板块代码列表concepts_codes中的各个股所在概念板块代码concept_code
            for concept_code in concepts_codes:
                # 对于个股所在概念板块代码concept_code在日期date时对应的个股所在概念板块的个股代码列表get_concept_stocks()中的个股所在概念板块的各个股代码concept_stock_code
                concept_stocks_codes = get_concept_stocks(concept_code, date=date)
                concepts_data[concept_code].loc[date, set(concept_stocks_codes) & set(self.__stocks_codes)] = 1
                #for concept_stock_code in get_concept_stocks(concept_code, date=date):
                    #  若个股所在概念板块的某个股代码concept_stocks_codes属于个股代码列表stocks_codes
                    #if concept_stock_code in stocks_codes:
                        # 将个股所在概念板块的数据表字典concepts_data中某个股所在概念板块代码concept_code对应的个股所在概念板块的数据表concept_data的某个股代码concept_stock_code的对应列的索引为日期date的数据传入1
                        #concepts_data[concept_code].loc[date, concept_stock_code] = 1
                    # 除此之外，对于个股所在概念板块代码concept_code在日期date时对应的个股所在概念板块的个股代码列表get_concept_stocks()以外的，且在个股列表stocks_codes中的个股，其可能未上市，或者上市但不属于该代码为concept_code的概念板块，或上市但所在概念板块的个股代码列表get_concept_stocks()存在缺失值
                    #else:
                        # 这三种情况交由缺失值处理
                        #pass
                            
        # 为避免直接用concepts_data.items()遍历，从而在循环中改变concepts_data，所以将concepts_data静态复制为concepts_data.copy()后再使用.items()，最后存为concepts_data_items
        concepts_data_items = concepts_data.copy().items()
        # 对于个股所在概念板块的数据表concept_data的字典concepts_data中的键值对各个股所在概念板块代码concept_code: 个股所在概念板块的数据表concept_data
        for concept_code, concept_data in concepts_data_items:
            # 如果个股所在概念板块的数据表concept_data的所有值.any().any()不not存在1值
            if not concept_data.any().any():
                # 删除个股所在概念板块的数据表字典concepts_data中某个股所在概念板块代码concept_code对应的个股所在概念板块的数据表concept_data
                concepts_data.pop(concept_code)
            else:
                # 极端值处理
                concept_data = concept_data.astype(float)
                concept_data = winsorize_med(concept_data, scale=5, inclusive=True, inf2nan=False, axis=1)
                # 标准化处理处理
                concept_data = standardlize(concept_data, inf2nan=False, axis=1)
                # 中性化处理
                concept_data = neutralize(concept_data, how=['jq_l2', 'ln_circulating_market_cap'], date=self.__end_date, axis=1, fillna=None, add_constant=True)
                concepts_data[concept_code] = concept_data
                
        # 返回个股所在概念板块的数据表concept_data的字典concepts_data
        return concepts_data
    
    # 获取指数因子数据，同概念

    def get_indexes_info_and_data(self):
        
        indexes_data = {}
        
        indexes_info = pd.read_csv('indexes_info.csv')

        index_data = pd.DataFrame(0, columns=self.__stocks_codes, index=self.dates)

        for _, index_info in tqdm(indexes_info.iterrows(), desc='index calling progress'):
            index_code = index_info['指数代码']
            index_start_date = index_info['发布日期']
            index_start_date = datetime.datetime.strptime(index_start_date, '%Y/%m/%d').date()
            
            for date in self.dates:
                if date > index_start_date:
                    index_stocks_codes = get_index_stocks(index_symbol=index_code, date=date)
                    index_data.loc[date, set(index_stocks_codes) & set(self.__stocks_codes)] += 1
                    
        # 极端值处理
        index_data = index_data.astype(float)
        index_data = winsorize_med(index_data, scale=5, inclusive=True, inf2nan=False, axis=1) 
        # 标准化处理处理
        index_data = standardlize(index_data, inf2nan=False, axis=1)
        # 中性化处理
        index_data = neutralize(index_data, how=['jq_l2', 'ln_circulating_market_cap'], date=self.__end_date, axis=1, fillna=None, add_constant=True)
                    
        indexes_data['included_indexes_number'] = index_data
        
        return indexes_data

    # 获取股票是否为ST股的因子

    def get_STs_data(self):
        STs_data = {}
        ST_data = get_extras('is_st', self.__stocks_codes, start_date=self.__start_date, end_date=self.__end_date, df=True)
        
        # 极端值处理
        ST_data = ST_data.astype(float)
        ST_data = winsorize_med(ST_data, scale=5, inclusive=True, inf2nan=False, axis=1)
        # 标准化处理处理
        ST_data = standardlize(ST_data, inf2nan=False, axis=1)
        # 中性化处理
        ST_data = neutralize(ST_data, how=['jq_l2', 'ln_circulating_market_cap'], date=self.__end_date, axis=1, fillna=None, add_constant=True)

        if ST_data.any().any():
            STs_data['is_ST'] = ST_data
        
        return STs_data

    # 获取除息除权因子数据

    # 获取限售解禁因子数据

    def get_locked_shares_data(self):
    
        locked_shares_codes = [
            'rate1'
        ]
        
        locked_shares_data = {}
        for locked_share_code in locked_shares_codes:
            # 对于使用的因子
            if locked_share_code in locked_shares_codes:
                locked_share_data_list = []
                for date in tqdm(self.dates, desc='calling locked_shares_data'):
                    # 定义某时期基本面数据表fundamental_data，请求query数据表包含'stock_code'列，表示个股的代码，fundamental列表示对应个股的基本面指标fundament
                    # -al，要求.filter'stock_code'列的个股代码indicator.code属于.in_个股代码列表stocks_codes，查询日期为date
                    locked_share_data = get_locked_shares(stock_list=self.__stocks_codes, start_date=date, end_date=date)[['code', locked_share_code]]
                    locked_share_data.rename(columns={'code': 'stock_code'}, inplace=True)
                    # 如果时期参数为date=date时基本面数据表fundamental_datafundamental_data为空，则表明fundamental来自银行业财务指标表bank_indicator，此时将时期参数改为statDate=date.year，即用时期date的年份查询报告日期statDate
                    if locked_share_data.empty:
                        locked_share_data = pd.DataFrame()
                        locked_share_data['stock_code'] = get_locked_shares(stock_list=self.__stocks_codes, start_date=date, end_date=date)['code']
                        locked_share_data[locked_share_code] = np.nan

                    # 定义个股代码数据表stocks_codes_df，其只含有'stock_code'列，取值为个股代码列表stocks_codes
                    stocks_codes_df = pd.DataFrame({'stock_code': self.__stocks_codes})
                    # 将基本面数据表fundamental_data左连接个股代码数据表stocks_codes_df，以补全某时期date的基本面数据表fundamental_data中由于不在市而未返回的个股代码，并将这些未上市的个股某时期date的基本面数据补全为缺失值
                    locked_share_data = pd.merge(stocks_codes_df, locked_share_data, on='stock_code', how='left')
                    # 删除某时期date的基本面数据表fundamental_data的'stock_code'列，只剩下fundamental列
                    locked_share_data.drop('stock_code', axis=1, inplace=True)
                    # 将基本面数据表fundamental_data进行转置，将其仅有的'fundamental'列转置为一个样本，样本每一列为某时期date的每个个股的基本面数据
                    locked_share_data = locked_share_data.T
                    # 将某时期date的基本面数据表fundamental_data添加到某基本面数据表列表fundamental_data_list中
                    locked_share_data_list.append(locked_share_data)
                # 纵向axis=1合并某基本面数据表的列表fundamental_data_list中，dates中各时期的基本面数据表fundamental_data
                locked_share_data = pd.concat(locked_share_data_list, axis=0)
                # 为dates中各时期的基本面数据表fundamental_data添加纵向索引dates代表样本的时期date
                locked_share_data.index = self.dates
                # 将索引重命名为'date'
                locked_share_data.index.name = 'date'
                # 将fundamental_data的列名.columns[i]命名为对应个股名称stocks_codes[i]
                locked_share_data.columns = self.__stocks_codes
        
            # 极端值处理
            locked_share_data = locked_share_data.astype(float)
            locked_share_data = winsorize_med(locked_share_data, scale=5, inclusive=True, inf2nan=False, axis=1)
            # 标准化处理处理
            locked_share_data = standardlize(locked_share_data, inf2nan=False, axis=1)
            # 中性化处理
            locked_share_data = neutralize(locked_share_data, how=['jq_l2', 'ln_circulating_market_cap'], date=self.__end_date, axis=1, fillna=None, add_constant=True)
            
            if locked_share_data.any().any():
                # 将各时期dates的fundamentals中的某基本面指标fundamental的基本面数据表fundamental_data作为值添加到各基本面数据表字典fundamentals_data，键设定为基本面指标名称fundamental_name
                locked_shares_data[locked_share_code] = locked_share_data
            
        # 返回各基本面数据表字典fundamentals_data
        return locked_shares_data
    
    # 获取除权除息因子数据

    # 定义从开始日期start_date到结束日期end_date，个股列表为stocks_codes，宏观指标所在表对象table的列表为tables的宏观数据字典macros_data取得函数get_all_macros_data
    def get_XRs_XDs_data(self, tables=[
        finance.STK_XR_XD
    ]):
        
        # 定义根据宏观指标所在表对象table和其属性字典table_attributes_dict的某键，即某属性名称macro_name返回总过滤条件字典filters的总过滤条件字典取得函数get_filters
        def get_attributes_filters(table, attribute_name):
            # 将“列表属性字典的某键，即某属性名称attribute_name不属于各表table的无关属性名称列表attributes_name_to_delete”的布尔值传出为原始过滤条件original_filter
            original_filter = attribute_name not in attributes_names_to_delete
            # 挑选属于银行业的属性，而非金融业甚至社会的属性，防止银行业的信息被稀释
            if table is finance.STK_XR_XD:
                # 将“列表属性字典的某键，即某属性名称attribute_name包含'_ratio'，即包含比率宏观指标属性”的布尔值传出为额外过滤条件extra_filter
                extra_filter = attribute_name in [
                    'dividend_ratio', 
                    'transfer_ratio', 
                    'at_bonus_ratio_rmb', 
                    'dividend_number', 
                    'transfer_number', 
                    'bonus_amount_rmb'
                ]
                # 将原始过滤条件original_filter和额外过滤条件extra_filter的析取命传出为总过滤条件字典filters
                attributes_filters = original_filter & extra_filter
            # 除此之外
            else:
                # 将真值True传出为额外过滤条件extra_filter
                extra_filter = True
                # 将原始过滤条件original_filter和额外过滤条件extra_filter的析取命传出为总过滤条件字典filters
                attributes_filters = original_filter & extra_filter
            # 返回总过滤条件字典filters
            return attributes_filters
        
        def get_query_params(table, date):
            query_params = {}
            
            stock_code_filter = (table.code.in_(self.__stocks_codes))
            date_filter = (table.report_date == date)
            columns_filter = True
            query_params['filter'] = stock_code_filter & date_filter & columns_filter
            
            return query_params
        
        # 定义各宏观数据表字典macros_data，其元素为各时期dates的macros中的各宏观指标的宏观数据表macro_data
        XRs_XDs_data = {}
        # 定义各宏观指标所在表对象table的宏观指标字典macros_dict，对于其中的键值对，键为宏观指标名称'macro'，值为对应的各表table中的宏观指标对象macro
        tables_and_XRs_XDs_dict = {}
        # 定义各表table的无关属性名称列表attributes_name_to_delete
        attributes_names_to_delete = ['__module__', '__dict__', '__weakref__', '__doc__', '_sa_class_manager', '__init__', 'id', 'day']
        
        # 对于宏观指标所在表对象的列表为tables中各宏观指标所在表对象table
        for table in tables:
            # 将宏观指标所在表对象table的属性vars()，转化为字典dict()，并传出为宏观指标所在表对象table的属性字典table_attributes_dict，对于其中的键值对，键为属性名称attribute_name，值为属性对象attribute
            table_attributes_dict = dict(vars(table))
            # 对于宏观指标所在表对象table的属性字典table_attributes_dict其中的键值对，即属性名称attribute_name: 属性对象attribute
            table_and_XRs_XDs_dict = {attribute_name: {'table': table, 'XR_XD': attribute} 
                                    for attribute_name, attribute in table_attributes_dict.items()
                                    # 筛选其中属性名称attribute_name满足总过滤条件字典filters的属性名称attribute_name: 属性对象attribute，作为宏观名称macro_name: 宏观对象macro字典，返回对应键值对组成的宏观指标所在表对象table的宏观字典table_macros_dict
                                    if get_attributes_filters(table, attribute_name)
                                    }
            
            # 将宏观指标所在表对象table的属性字典table_attributes_dict合并入各宏观指标所在表对象table的宏观指标字典macros_dict
            tables_and_XRs_XDs_dict.update(table_and_XRs_XDs_dict)
            
        for XR_XD_name, XR_XD_info in tqdm(tables_and_XRs_XDs_dict.items(), desc='calling table progress:'):
        # 对于宏观指标列表macros中的各宏观因子fundamental
            table, XR_XD = XR_XD_info['table'], XR_XD_info['XR_XD']
            # 定义某宏观数据表的列表macro_data_list，其元素为各时期date的某宏观指标macro的宏观数据表macro_data
            XR_XD_data_list = []
            # 对于时期列表dates中的各时期date
            for date in self.dates:
                # 定义某时期宏观数据表macro_data，请求query数据表包含'stock_code'列，表示个股的代码，macro列表示对应个股的宏观指标fundament
                # -al，要求.filter'stock_code'列的个股代码indicator.code属于.in_个股代码列表stocks_codes，查询日期为date
                XR_XD_data = finance.run_query(
                    query(
                        table.code.label('stock_code'),
                        XR_XD
                    ).filter(
                        get_query_params(table, date)['filter']
                    )
                )
                
                # 定义个股代码数据表stocks_codes_df，其只含有'stock_code'列，取值为个股代码列表stocks_codes
                stocks_codes_df = pd.DataFrame({'stock_code': self.__stocks_codes})
                # 将基本面数据表fundamental_data左连接个股代码数据表stocks_codes_df，以补全某时期date的基本面数据表fundamental_data中由于不在市而未返回的个股代码，并将这些未上市的个股某时期date的基本面数据补全为缺失值
                XR_XD_data = pd.merge(stocks_codes_df, XR_XD_data, on='stock_code', how='left')
                # 删除某时期date的基本面数据表fundamental_data的'stock_code'列，只剩下fundamental列
                XR_XD_data.drop('stock_code', axis=1, inplace=True)
                # 将宏观数据表macro_data进行转置，将其仅有的'macro'列转置为一个样本，样本每一列为某时期date的每个个股的宏观数据
                XR_XD_data = XR_XD_data.T
                # 将某时期date的宏观数据表macro_data添加到某宏观数据表列表macro_data_list中
                XR_XD_data_list.append(XR_XD_data)
            # 纵向axis=1合并某宏观数据表的列表macro_data_list中，dates中各时期的宏观数据表macro_data
            XR_XD_data = pd.concat(XR_XD_data_list, axis=0)
            # 为dates中各时期的宏观数据表macro_data添加纵向索引dates代表样本的时期date
            XR_XD_data.index = self.dates
            # 将索引重命名为'date'
            XR_XD_data.index.name = 'date'
            # 将列名重命名为个股代码
            XR_XD_data.columns = self.__stocks_codes
            
            # 极端值处理
            XR_XD_data = XR_XD_data.astype(float)
            XR_XD_data = winsorize_med(XR_XD_data, scale=5, inclusive=True, inf2nan=False, axis=1)
            # 标准化处理处理
            XR_XD_data = standardlize(XR_XD_data, inf2nan=False, axis=1)
            # 中性化处理
            XR_XD_data = neutralize(XR_XD_data, how=['jq_l2', 'ln_circulating_market_cap'], date=self.__end_date, axis=1, fillna=None, add_constant=True)
            

            # 将各时期dates的macros中的某宏观指标macro的宏观数据表macro_data作为值添加到各宏观数据表字典macros_data，键设定为宏观指标名称macro_name
            if XR_XD_data.any().any():
                XRs_XDs_data[XR_XD_name] = XR_XD_data
            
        # 返回各宏观数据表字典macros_data
        return XRs_XDs_data

    # 获取股价收益率数据
    def get_returns_data(self, periods=19):
        all_dates = get_all_trade_days()
        last_date_index = [index + periods for index, date in enumerate(all_dates) if date == self.__end_date][0]
        suf_end_date = all_dates[last_date_index]
        suf_dates = all_dates[(all_dates >= self.__start_date) & (all_dates <= suf_end_date)]
        
        returns_data = {}
        for stocks_codes in [self.__stocks_codes, ['000001.XSHG']]:
            # 尝试利用pandas.panel对象的快速计算方法
            try:
                warnings.filterwarnings('ignore')

                # 我们使用收盘价计算每只股票在每个时期的收益率。这种做法与多因子模型的目的和设计相一致,因为:
                # 收盘价提供了一个标准化的、可比较的股价基准,使得不同股票和不同时期的收益率计算具有一致性。
                # 多因子模型关注的是股票收益率的截面差异,而不是日内的价格波动。使用收盘价可以过滤掉日内噪音,更好地捕捉因子对收益率的影响。
                # 大多数因子数据(如财务比率、分析师预测等)都是以日度或月度频率更新的,使用收盘价计算的收益率可以与这些因子数据的频率相匹配。
                # 使用收盘价计算收益率是学术研究和实务中的常见做法,便于与其他研究和基准进行比较和评估。
                prices = get_price(stocks_codes, start_date=None, end_date=suf_end_date, fields=('close'))['close']
                # 在学术研究和量化投资实践中,月度数据（大约20个交易日）是最常用的选择，月度数据有以下优点:
                # 与大多数公司的财务报告频率(季度或年度)相匹配,便于利用财务数据构建因子。
                # 与许多宏观经济数据的发布频率一致,便于研究宏观因素对股票收益的影响。
                # 相比日度或周度数据,月度数据包含较少的短期噪音和异常值,更能捕捉因子的中长期效应。
                # 月度调仓频率与许多机构投资者的实际做法相一致,便于模型结果的实践应用。

                returns = prices.shift(periods=-periods) / prices - 1

                returns = returns[returns.index.isin(self.dates)]

                if stocks_codes == self.__stocks_codes:
                    self.__returns_data = returns_data['returns'] = returns
                elif stocks_codes == ['000001.XSHG']:
                    returns_data['market_returns'] = returns

            # 若pandas.penal对象已过期
            except:
                prices = get_price(self.__stocks_codes, start_date=self.__start_date, end_date=suf_end_date, fields=('close'), panel=False)
                prices['time'] = prices['time'].dt.date

                price_initial = pd.DataFrame(index=suf_dates)
                price_initial.index.name = 'date'
                for code, price in prices.groupby('code'):
                    price = price.drop('code', axis=1)

                    price.set_index('time', inplace=True)
                    price.index.name = 'date'

                    price = price.rename(columns={'close': code})
                    price_initial = price_initial.join(price)

                prices = price_initial

                returns = prices.shift(periods=-periods) / prices - 1

                # 筛选开始日期到结束日期内的成交日，排除suf_end_date
                returns = returns[returns.index.isin(self.dates)]

                returns = returns.reindex(columns=self.__stocks_codes)

                if stocks_codes == self.__stocks_codes:
                    self.__returns_data = returns_data['returns'] = returns
                elif stocks_codes == ['000001.XSHG']:
                    returns_data['market_returns'] = returns

        return returns_data
    
    def get_YTMs_data(self):
        
        # 设置 decimal 的精度
        getcontext().prec = 28

        YTMs_data = {}
        
        bonds_info = bond.run_query(
            query(
                bond.BOND_BASIC_INFO.code,
                bond.BOND_BASIC_INFO.company_code,
                bond.BOND_BASIC_INFO.coupon_type_id,
                bond.BOND_BASIC_INFO.coupon_frequency,
                bond.BOND_BASIC_INFO.interest_begin_date,
                bond.BOND_BASIC_INFO.last_cash_date # 现金流折现模型中的实际到期日
            ).filter(
                bond.BOND_BASIC_INFO.company_code.in_(self.__stocks_codes),
                # 与研究期无重合的债券有效期不构成研究期的债务，也不产生债务资本成本率，下同
                bond.BOND_BASIC_INFO.last_cash_date >= self.__start_date,
                # 债券没有不确定性，发行后预期到期收益率就已定，在研究期末前就可知
                bond.BOND_BASIC_INFO.interest_begin_date <= self.__end_date,
            )
        )
        bonds_codes = bonds_info['code']
        
        sum_count_bonds_coupons = bond.run_query(
            query(
                func.count('*')
            ).filter(
                bond.BOND_COUPON.code.in_(bonds_codes),
                # 债券没有不确定性，报告后预期到期收益率就可知，在研究期末前就已知
                bond.BOND_COUPON.pub_date <= self.__end_date
                )
        ).iloc[0,0]
        
        bonds_coupons_list = []
        for count_bonds_coupons in range(0, sum_count_bonds_coupons, 5000):
            bond_coupon = bond.run_query(
                query(
                    bond.BOND_COUPON.code,
                    bond.BOND_COUPON.pub_date,
                    bond.BOND_COUPON.coupon,
                    bond.BOND_COUPON.coupon_start_date,
                    bond.BOND_COUPON.coupon_end_date
                ).filter(
                    bond.BOND_COUPON.code.in_(bonds_codes),
                    # 债券没有不确定性，报告后预期到期收益率就可知，在研究期末前就已知
                    bond.BOND_COUPON.pub_date <= self.__end_date
                ).offset(count_bonds_coupons)
            )
            bonds_coupons_list.append(bond_coupon)
        bonds_coupons = pd.concat(bonds_coupons_list, axis=0)
        
        def get_C_month(date_bond_info, F=100):
            if isinstance(date_bond_info, pd.Series):
                date_bond_info = pd.DataFrame(date_bond_info.T)
                
            date_bond_info.reset_index(inplace=True)
            
            coupon_type = date_bond_info.loc[0, 'coupon_type_id']
            T = date_bond_info.loc[0, 'repayment_period']
            t_step = date_bond_info.loc[0, 'coupon_frequency']
            rs = date_bond_info['coupon']
            rs_start_dates = date_bond_info['coupon_start_date']
            rs_end_dates = date_bond_info['coupon_end_date']
            if np.isnan(coupon_type) \
            or np.isnan(T) \
            or np.isnan(t_step) \
            or rs.empty \
            or rs_start_dates.empty \
            or rs_end_dates.empty \
            or T <= t_step \
            or any(rs <= 0) \
            or any(rs_end_dates <= rs_start_dates):
                # 终止函数运行
                return
            else:
                coupon_type = int(coupon_type)
                T = int(T)
                t_step = int(t_step)
                
                C_month = [0] * T
                
                if coupon_type == 701001:  # 利随本清
                    r_month = rs.iloc[-1] / 12 / 100 # 月化，百分比化
                    
                    C_month[-1] = F * (1 + r_month)
                elif coupon_type == 701002:  # 固定利率附息
                    r_month = rs.iloc[-1] / 12 / 100
                    
                    for t_month in range(0, T, t_step):
                        C_month[t_month] = F * r_month
                    C_month[-1] += F
                else:  # 浮动利率
                    start_date = rs_start_dates.min()
                    # 债券到期收益率(Yield to Maturity,YTM)是在债券存续期间固定不变的,无论债券发行时的票面利率如何,也不受实际利息支付次数的影响
                    # 所以直接考虑债券发行时的未来现金流，计算债券发行是的YTM即可
                    for t_month in range(0, T, t_step):
                        current_date = start_date + relativedelta(months=t_month)
                        mask = (rs_start_dates <= current_date) & \
                            (rs_end_dates > current_date) 
                        if not mask.any():
                            mask[-1:] = True
                        
                        r_month = date_bond_info.loc[mask, 'coupon'].iloc[0] / 12 / 100
                        C_month[t_month] = F * r_month
                    C_month[-1] += F
                
            return C_month

        def calc_YTM(C_month, P=100):
            def calc_NPV(YTM_month_vec, C_month=C_month, P=P):
                # 使用Decimal类型可以提供高精度的数值计算，可以避免浮点数溢出问题
                # YTM_month自动被fsolve向量化为np.array
                YTM_month = YTM_month_vec[0]
                PV = sum([Decimal(c_month) / (1 + Decimal(YTM_month)) ** (t_month + 1) for t_month, c_month in enumerate(C_month)]) # t_month从0开始，需要+1
                # 将PV的类型由Decimal转换回float
                PV = float(PV)
                return [PV - P]
            
            def calc_NPV_jacobian(YTM_month_vec, C_month=C_month):
                YTM_month = YTM_month_vec[0]
                # 计算关于YTM_month的jacobian矩阵
                J = sum([- (t_month + 1) * Decimal(c_month) / (1 + Decimal(YTM_month)) ** (t_month + 2) for t_month, c_month in enumerate(C_month)])
                J = float(J)
                return [J]
            
            YTM_month_pre = 0.04
            YTM_month_suf = 0.2
            YTM_month_step = 0.05
            for iteration in range(10):
                YTMs_month_initial = np.arange(YTM_month_pre, YTM_month_suf, YTM_month_step)
                YTMs_month = []

                for YTM_month_initial in YTMs_month_initial:
                    try:
                        # 忽视warnings防止YTM_month的计算被中断
                        YTM_month_star = fsolve(calc_NPV, YTM_month_initial, fprime=calc_NPV_jacobian)[0]
                        # 如果不报错，直接返回YTMs_month_star跳出while循环
                        return YTM_month_star
                    except RuntimeWarning:
                        # 恢复对于warnings作为error的过滤
                        warnings.filterwarnings('ignore')
                        YTM_month = fsolve(calc_NPV, YTM_month_initial, fprime=calc_NPV_jacobian)[0]
                        YTMs_month.append(YTM_month)
                        warnings.filterwarnings('error')
                        continue
                
                # 找到在失败值中最优的YTMs_month_star
                YTM_month_star = min(YTMs_month, key=lambda YTM_month: abs(calc_NPV([YTM_month])[0])) # 先将输入YTM_month向量化为YTM_month_vec == [YTM_month]，再将输出[YTM_month_star]标量化为YTM_month_star == [YTM_month_star][0]
                
                YTMs_month.sort()
                ord_star = YTMs_month.index(YTM_month_star)
                ord_interval = 1
                YTM_month_pre = YTMs_month[ord_star - 1]
                YTM_month_suf = YTMs_month[ord_star + 1]
                # 在初猜值区间进行最优解二分查找，且防止YTM_month_suf与YTM_month_pre相等
                while YTM_month_pre == YTM_month_suf:
                    if ord_star - ord_interval >= 0 and ord_star + ord_interval <= len(YTMs_month):
                        ord_interval += 1
                        YTM_month_pre = YTMs_month[ord_star - ord_interval]
                        YTM_month_suf = YTMs_month[ord_star + ord_interval]
                    # YTMs_month中元素均相等，强行
                    else:
                        YTM_month_pre *= 0.8
                        YTM_month_suf /= 0.7
                        
                YTM_month_step = (YTM_month_suf - YTM_month_pre) / 5
            
            # 到达最大递归次数，递归求解仍然失败，则直接return最后的解
            return YTM_month_star
        
        def get_C_month_and_PV(C_month, YTM_month_star, date, date_bond_info):
            rs_start_dates = date_bond_info['coupon_start_date']
            if rs_start_dates.empty:
                return
            else:
                start_date = rs_start_dates.min()
                interval = relativedelta(date, start_date)
                t_month = interval.years * 12 + interval.months + round(interval.days / 30)
                # 截断过去已经支付的现金流
                C_month = C_month[t_month - 1:]
                
                PV = sum([c_month / (1 + YTM_month_star) ** (t_month + 1) for t_month, c_month in enumerate(C_month)])
                return C_month, PV

        YTMs_data['YTM'] = pd.DataFrame(index=self.dates, columns=self.__stocks_codes)
        
        # 分别初始化防止连续赋值时产生链接
        YTMs = YTMs_data['YTM'].to_dict('dict')
        Cs_month_agg = YTMs_data['YTM'].to_dict('dict')
        for date in tqdm(self.dates, desc='YTMs calling progress'):
            for stock_code in self.__stocks_codes:
                bonds_info_mask = \
                (bonds_info['company_code'] == stock_code) & \
                (bonds_info['interest_begin_date'] < date) & \
                (bonds_info['last_cash_date'] > date)
                bonds_coupons_mask = (bonds_coupons['pub_date'] < date)
                date_bonds_info = bonds_info[bonds_info_mask].merge(bonds_coupons[bonds_coupons_mask], on='code')
                #with pd.option_context('expand_frame_repr', False, 'display.max_rows', None):  

                if not date_bonds_info.empty:
                    Cs_month = {}
                    YTMs_month_star = {}
                    PVs = {}
                    weights = {}
                    for code, date_bond_info in date_bonds_info.groupby('code'):
                        interval = relativedelta(date_bond_info['last_cash_date'].iloc[0], date_bond_info['interest_begin_date'].iloc[0])
                        date_bond_info = date_bond_info.copy()
                        date_bond_info.loc[:, 'repayment_period'] = interval.years * 12 + interval.months + round(interval.days / 30)

                        C_month = get_C_month(date_bond_info)
                        if not C_month:
                            continue
                        YTMs_month_star[code] = YTM_month_star = calc_YTM(C_month)
                        Cs_month[code], PVs[code] = get_C_month_and_PV(C_month, YTM_month_star, date, date_bond_info)
                    
                    # 没有债券实际成交量数据，所以将现金流按照等权（将所有债权的发行量视为相等）相加而非按照发行量在总发行量的权重进行相加
                    weights_keys = Cs_month.keys()
                    weights_values = [1 / len(weights_keys)] * len(weights_keys)
                    weights = dict(zip(weights_keys, weights_values))
                        
                    if Cs_month and PVs:
                        max_T_month = max([len(C_month) for code, C_month in Cs_month.items()])
                        C_month_agg = [0] * max_T_month
                        PV_agg = 0
                        for code, C_month in Cs_month.items():
                            PV = PVs[code]
                            weight = weights[code]
                            
                            C_month_agg[:len(C_month)] = [c_agg + weight * c_delta for c_agg, c_delta in zip(C_month_agg, C_month)]
                            PV_agg += weight * PV
                            
                        date_str = str(date)
                        Cs_month_agg[stock_code][date_str] = C_month_agg

                        if date > self.__start_date:
                            past_dates_str = [
                                past_date_str 
                                for past_date_str, past_C_month_agg in Cs_month_agg[stock_code].items() 
                                if isinstance(past_C_month_agg, list)
                                and past_date_str != date_str
                                and past_C_month_agg == C_month_agg
                            ]
                            if past_dates_str:
                                last_date_str = past_dates_str[-1]
                                # date_bonds_info与YTM_month_star一一对应，构建双射，则
                                # 如果date_bonds_info与上一交易日的date_bonds_info，即last_date_bonds_info相等，则YTM_month_star与上一交易日的YTM_month_star，即last_YTM_month_star必相等，不必浪费时间计算
                                YTMs_data['YTM'].loc[date, stock_code] = YTMs[stock_code][date_str] = YTMs[stock_code][last_date_str]
                                continue
                            
                        YTM_month_star = calc_YTM(C_month_agg, PV_agg)
                        if YTM_month_star < 0.01:
                            YTM_month_star_average = np.mean(list(YTMs_month_star.values()))
                            YTMs_data['YTM'].loc[date, stock_code] = YTMs[stock_code][date_str] = max(YTM_month_star, YTM_month_star_average)
                        else:
                            YTMs_data['YTM'].loc[date, stock_code] = YTMs[stock_code][date_str] = YTM_month_star
        
        # 为了定义self.__interest_rate_1m_data而调用其定义方法
        self.get_FCF_discounted_model_params_data()
        # 用1个月银行间拆借利率填充空值，因为拆借利率的口径是1个月
        YTMs_data['YTM'].fillna(value=self.__interest_rate_1m_data, inplace=True)
        return YTMs_data
    
    # 获取FCF_discounted_mode所需的参数
    def get_FCF_discounted_model_params_data(self, tables=[
        bank_indicator,
        valuation, 
        balance,
        finance.FINANCE_BALANCE_SHEET,
        income,
        macro.MAC_LEND_RATE
    ]):

        # 定义根据基本面指标所在表对象table和其属性字典table_attributes_dict的某键，即某属性名称FCF_discounted_model_param_name返回总过滤条件字典filters的总过滤条件字典取得函数get_filters
        def get_attributes_filters(table, attribute_name):
            # 将“列表属性字典的某键，即某属性名称attribute_name不属于各表table的无关属性名称列表attributes_name_to_delete”的布尔值传出为原始过滤条件filter
            original_filter = attribute_name not in attributes_names_to_delete
            if table is bank_indicator:
                extra_filter = \
                attribute_name == 'interest_bearing_liabilities' \
                or attribute_name == 'interest_bearing_liabilities_interest_rate'
                attributes_filters = original_filter and extra_filter
            # 当基本面指标所在表对象table是市值表valuation时
            elif table is valuation:
                # 将“列表属性字典的某键，即某属性名称attribute_name包含'_ratio'，即包含比率基本面指标属性”的布尔值传出为额外过滤条件extra_filter
                extra_filter = attribute_name == 'circulating_market_cap'
                attributes_filters = original_filter and extra_filter
            elif table is balance:
                extra_filter = \
                attribute_name == 'cash_equivalents' \
                or attribute_name == 'lend_capital' \
                or attribute_name == 'trading_assets' \
                or attribute_name == 'bought_sellback_assets'
                # 将原始过滤条件filter和额外过滤条件extra_filter的析取命传出为总过滤条件字典filters
                attributes_filters = original_filter and extra_filter
            elif table is finance.FINANCE_BALANCE_SHEET:
                extra_filter = \
                attribute_name == 'cash_in_cb' \
                or attribute_name == 'deposit_in_ib'
                #or attribute_name == 'cash_equivalents'\
                #or attribute_name == 'deposit_client' # 以上两个字段缺失值太多
                # 将原始过滤条件filter和额外过滤条件extra_filter的析取命传出为总过滤条件字典filters
                attributes_filters = original_filter and extra_filter
            elif table is income:
                extra_filter = \
                attribute_name == 'income_tax_expense' \
                or attribute_name == 'total_profit'
                # 将原始过滤条件filter和额外过滤条件extra_filter的析取命传出为总过滤条件字典filters
                attributes_filters = original_filter and extra_filter
            elif table is macro.MAC_LEND_RATE:
                extra_filter = attribute_name == 'interest_rate'
                attributes_filters = original_filter and extra_filter
            # 除此之外
            else:
                # 将真值True传出为额外过滤条件extra_filter
                extra_filter = True
                # 将原始过滤条件filter和额外过滤条件extra_filter的析取命传出为总过滤条件字典filters
                attributes_filters = original_filter and extra_filter
            # 返回总过滤条件字典filters
            return attributes_filters

        # 根据给定日期返回所在季度的最后一天
        def get_last_day_of_quarter(date):
            # 获取季度的第一个月
            quarter_start = (date.month - 1) // 3 * 3 + 1
            # 计算下一季度的第一个月
            next_quarter_start = quarter_start + 3 if quarter_start < 10 else 1
            # 计算下一季度第一天的前一天，即当前季度的最后一天
            last_day_of_quarter_of_date = datetime.date(date.year, next_quarter_start, 1) - datetime.timedelta(days=1)

            return last_day_of_quarter_of_date

        def get_query_params(table, date, attribute_name, get_last_day_of_quarter=get_last_day_of_quarter):
            query_params={}

            if table is bank_indicator:
                stock_code_filter = (table.code.in_(self.__stocks_codes))
                columns_filter = True
                query_params['date'] = None
                query_params['statDate'] = date.year
                query_params['filter'] = stock_code_filter & columns_filter
            elif table is finance.FINANCE_BALANCE_SHEET:
                stock_code_filter = (table.code.in_(self.__stocks_codes))
                columns_filter = \
                (table.report_date == get_last_day_of_quarter(date)) & \
                (table.report_type == 0)
                query_params['filter'] = stock_code_filter & columns_filter
            elif table is macro.MAC_LEND_RATE:
                if attribute_name == 'interest_rate_1m':
                    term_id = '1'
                date_filter = (table.day == date)
                columns_filter = (table.currency_id == '1') & (table.market_id == '5') & (table.term_id == term_id)
                query_params['filter'] = date_filter & columns_filter
            else:
                stock_code_filter = (table.code.in_(self.__stocks_codes))
                columns_filter = True
                query_params['date'] = date
                query_params['statDate'] = None
                query_params['filter'] = stock_code_filter & columns_filter

            return query_params
        
        # 定义不在市的股票掩码取得函数
        def get_mask(industry_stocks_info, dates):
            # 定义一个掩码mask,标识每个个股在每个时间点上是否处于上市状态
            mask = pd.DataFrame(index=dates, columns=industry_stocks_info.index.tolist())
            # 对于每个代码为stock的个股
            for stock_code in mask.columns:
                # 对于代码为stock_code的个股，查找板块个股代码列表industry_stocks_info对应的个股代码，传出其上市日期start_date和退市日日期end_date
                start_date = industry_stocks_info.loc[stock_code, 'start_date']
                end_date = industry_stocks_info.loc[stock_code, 'end_date']
                # 标识代码为stock的个股在每个时间点上是否处于上市状态
                mask[stock_code] = (mask.index >= start_date) & (mask.index <= end_date)
            return mask
        
        mask = get_mask(self.industry_stocks_info, self.dates)

        # 定义各基本面数据表字典FCF_discounted_model_params_data，其元素为各时期dates的fundamentals中的各基本面指标的基本面数据表FCF_discounted_model_param_data
        FCF_discounted_model_params_data = {}
        # 定义各基本面指标所在表对象table的基本面指标字典fundamentals_dict，对于其中的键值对，键为基本面指标名称'fundamental'，值为对应的各表table中的基本面指标对象fundamental
        tables_and_FCF_discounted_model_params_dict = {}
        
        # 定义各表table的无关属性名称列表attributes_name_to_delete
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
            'stat_year'
            'pubDate',
            'statDate', 
            '__table__', 
            '_sa_class_manager', 
            '__init__', 
            '__mapper__'
        ]
        
        # 对于基本面指标所在表对象的列表为tables中各基本面指标所在表对象table
        for table in tables:
            # 将基本面指标所在表对象table的属性vars()，转化为字典dict()，并传出为基本面指标所在表对象table的属性字典table_attributes_dict，对于其中的键值对，键为属性名称attribute_name，值为属性对象attribute
            table_attributes_dict = dict(vars(table))
            # 对于基本面指标所在表对象table的属性字典table_attributes_dict其中的键值对，即属性名称attribute_name: 属性对象attribute
            table_and_FCF_discounted_model_params_dict = {attribute_name: {'table': table, 'FCF_discounted_model_param': attribute} 
                                for attribute_name, attribute in table_attributes_dict.items()
                                # 筛选其中属性名称attribute_name满足总过滤条件字典filters的属性名称attribute_name: 属性对象attribute，作为宏观名称macro_name: 宏观对象macro字典，返回对应键值对组成的宏观指标所在表对象table的宏观字典table_macros_dict
                                if get_attributes_filters(table, attribute_name)
                                }

            # 将基本面指标所在表对象table的属性字典table_attributes_dict合并入各基本面指标所在表对象table的基本面指标字典fundamentals_dict
            tables_and_FCF_discounted_model_params_dict.update(table_and_FCF_discounted_model_params_dict)
            
            # 将银行间拆借利率interest_rate拆分中短期利率
            if table is macro.MAC_LEND_RATE:
                tables_and_FCF_discounted_model_params_dict['interest_rate_1m'] = tables_and_FCF_discounted_model_params_dict.pop('interest_rate')

        # 对于基本面指标列表fundamentals中的各基本面因子fundamental
        for FCF_discounted_model_param_name, FCF_discounted_model_param_info in tqdm(tables_and_FCF_discounted_model_params_dict.items(), desc='calling table progress'):
            # 对于宏观指标列表macros中的各宏观因子fundamental
            table, FCF_discounted_model_param = FCF_discounted_model_param_info['table'], FCF_discounted_model_param_info['FCF_discounted_model_param']
            # 定义某基本面数据表的列表FCF_discounted_model_param_data_list，其元素为各时期date的某基本面指标fundamental的基本面数据表FCF_discounted_model_param_data
            FCF_discounted_model_param_data_list = []
            # 对于时期列表dates中的各时期date
            for date in self.dates:
                # 定义某时期基本面数据表FCF_discounted_model_param_data，请求query数据表包含'stock_code'列，表示个股的代码，fundamental列表示对应个股的基本面指标fundament
                # -al，要求.filter'stock_code'列的个股代码indicator.code属于.in_个股代码列表stocks_codes，查询日期为date
                query_params = get_query_params(table, date, FCF_discounted_model_param_name)
                if table is finance.FINANCE_BALANCE_SHEET:
                    FCF_discounted_model_param_data = finance.run_query(
                        query(
                            table.code.label('stock_code'), 
                            FCF_discounted_model_param
                        ).filter(
                            query_params['filter']
                        )
                    )
                elif table is macro.MAC_LEND_RATE:
                    FCF_discounted_model_param_data = macro.run_query(
                        query(
                            FCF_discounted_model_param
                        ).filter(
                            query_params['filter']
                        )
                    )
                else:
                    FCF_discounted_model_param_data = get_fundamentals(
                        query(
                            table.code.label('stock_code'), 
                            FCF_discounted_model_param
                        ).filter(
                            query_params['filter']
                        ),
                        date=query_params['date'],
                        statDate=query_params['statDate']
                    )
                    
                if table is not macro.MAC_LEND_RATE:
                    # 定义个股代码数据表stocks_codes_df，其只含有'stock_code'列，取值为个股代码列表stocks_codes
                    stocks_codes_df = pd.DataFrame({'stock_code': self.__stocks_codes})
                    # 将基本面数据表FCF_discounted_model_param_data左连接个股代码数据表stocks_codes_df，以补全某时期date的基本面数据表FCF_discounted_model_param_data中由于不在市而未返回的个股代码，并将这些未上市的个股某时期date的基本面数据补全为缺失值
                    FCF_discounted_model_param_data = pd.merge(stocks_codes_df, FCF_discounted_model_param_data, on='stock_code', how='left')
                    # 删除某时期date的基本面数据表FCF_discounted_model_param_data的'stock_code'列，只剩下fundamental列
                    FCF_discounted_model_param_data.drop('stock_code', axis=1, inplace=True)
                    # 将基本面数据表FCF_discounted_model_param_data进行转置，将其仅有的'fundamental'列转置为一个样本，样本每一列为某时期date的每个个股的基本面数据
                FCF_discounted_model_param_data = FCF_discounted_model_param_data.T
                # 将某时期date的基本面数据表FCF_discounted_model_param_data添加到某基本面数据表列表FCF_discounted_model_param_data_list中
                FCF_discounted_model_param_data_list.append(FCF_discounted_model_param_data)

            # 纵向axis=1合并某基本面数据表的列表FCF_discounted_model_param_data_list中，dates中各时期的基本面数据表FCF_discounted_model_param_data
            FCF_discounted_model_param_data = pd.concat(FCF_discounted_model_param_data_list, axis=0)
            # 为dates中各时期的基本面数据表FCF_discounted_model_param_data添加纵向索引dates代表样本的时期date
            FCF_discounted_model_param_data.index = self.dates
            # 将索引重命名为'date'
            FCF_discounted_model_param_data.index.name = 'date'
            if table is not macro.MAC_LEND_RATE:
            # 将FCF_discounted_model_param_data的列名.columns[i]命名为对应个股名称stocks_codes[i]
                FCF_discounted_model_param_data.columns = self.__stocks_codes
                
            # 将不重要的FCF_discounted_model_param_data中的缺失值用行中位数进行填充，防止在后续因为个别不重要参数为缺失值，导致其组合的指标也为缺失值
            if FCF_discounted_model_param_name != 'interest_bearing_liabilities' \
            or FCF_discounted_model_param_name != 'interest_bearing_liabilities_interest_rate' \
            or FCF_discounted_model_param_name != 'interest_rate_1m':
                FCF_discounted_model_param_data = FCF_discounted_model_param_data.apply(
                    lambda FCF_discounted_model_param_data_row: FCF_discounted_model_param_data_row.fillna(
                        FCF_discounted_model_param_data_row.median()
                    ), 
                    axis=1
                )
                # 将填充后FCF_discounted_model_param_data中的不在市股票的数据重新标记为np.nan
                FCF_discounted_model_param_data[~mask] = np.nan
            
            # 将各时期dates的fundamentals中的某基本面指标fundamental的基本面数据表FCF_discounted_model_param_data作为值添加到各基本面数据表字典FCF_discounted_model_params_data，键设定为基本面指标名称FCF_discounted_model_param_name
            FCF_discounted_model_params_data[FCF_discounted_model_param_name] = FCF_discounted_model_param_data

        # 单位调整组合
        FCF_discounted_model_params_data['circulating_market_cap'] *= 1e8
        FCF_discounted_model_params_data['interest_bearing_liabilities_interest_rate'] /= 1.2e3
        FCF_discounted_model_params_data['interest_rate_1m'] /= 1.2e3

        # 数据导出与删除
        # 现金(Cash):库存现金:银行保险库内持有的各种面值的人民币及外币现金。存放中央银行款项:银行存放于中央银行的款项,包括法定准备金和超额存款准备金。存放同业款项:银行存放于其他银行的款项,通常用于资金清算、头寸调拨等。
        # 传出并删除原材料数据       
        cash_equivalents = FCF_discounted_model_params_data.pop('cash_equivalents')
        cash_in_cb = FCF_discounted_model_params_data.pop('cash_in_cb')
        deposit_in_ib = FCF_discounted_model_params_data.pop('deposit_in_ib')
        trading_assets = FCF_discounted_model_params_data.pop('trading_assets')
        lend_capital = FCF_discounted_model_params_data.pop('lend_capital')
        bought_sellback_assets = FCF_discounted_model_params_data.pop('bought_sellback_assets')
        interest_bearing_liabilities = FCF_discounted_model_params_data.pop('interest_bearing_liabilities')
        circulating_market_cap = FCF_discounted_model_params_data.pop('circulating_market_cap')
        income_tax_expense = FCF_discounted_model_params_data.pop('income_tax_expense')
        total_profit = FCF_discounted_model_params_data.pop('total_profit')
        interest_bearing_liabilities_interest_rate = FCF_discounted_model_params_data.pop('interest_bearing_liabilities_interest_rate')
        
                
        # 为了定义self.__returns_data而调用其定义方法
        self.get_returns_data()
        returns = self.__returns_data
        
        # 数据组合
        cash = cash_equivalents + cash_in_cb + deposit_in_ib
        # 短期投资(Short-term Investment):拆出资金:银行短期拆借给其他金融机构的资金,期限通常在一年以内。交易性金融资产:银行为交易目的而持有的金融资产,如债券、基金、股票等,预期在短期内出售以获利。买入返售金融资产:银行以证券或票据作为抵押,短期买入并约定未来返售的金融资产。
        short_investment = trading_assets + lend_capital + bought_sellback_assets
        net_debt = interest_bearing_liabilities - (cash + short_investment)
        enterprise_value = circulating_market_cap + net_debt
        # 计算每个公司的enterprise_value相对于其所在截面的enterprise_value总和的比例，以企业的企业价值在截面中的比重代表其系统重要性
        FCF_discounted_model_params_data['panal_enterprise_value_weights'] = enterprise_value.apply(
            lambda panal_enterprise_value_row: panal_enterprise_value_row / panal_enterprise_value_row.sum(), 
            axis=1
        )
        debt_to_enterprise_value_ratio = net_debt / enterprise_value

        # 银行的 D/E 比率通常较高,这是由银行的业务模式和资本结构所决定的。对大多数银行来说,D/E 比率大于 90% 是正常的,这与非金融企业有很大不同。
        # 以下是一些原因:
        # 银行的主要业务是借入资金(如吸收存款)并借出资金(如发放贷款)。存款在会计上被记为负债,但实际上更类似于银行的"原材料"而非融资来源。这导致银行的负债规模远大于其权益规模。
        # 银行的资本监管要求相对较低。根据巴塞尔协议,银行的核心一级资本充足率(核心一级资本/风险加权资产)要求为 4.5%,总资本充足率要求为 8%。这意味着银行可以用较少的资本支持较大的资产规模。
        # 银行的资产通常具有较低的风险权重。根据监管要求,银行持有的政府债券、优质企业贷款等资产的风险权重较低,这使得银行可以用较少的资本支持较大的资产规模。
        # 银行的杠杆倍数较高。银行的资本与资产的比例通常在 5-10% 之间,这意味着银行的资产规模可以达到其资本的 10-20 倍。这种高杠杆经营是银行业务模式的特点。
        # 银行的收入主要来自利差(借出资金的利率与借入资金的利率之差)。高杠杆可以放大这种利差收入,提高银行的权益回报率(ROE)。
        income_tax_rate = income_tax_expense / total_profit
        FCF_discounted_model_params_data['r_wacc'] = (1 - debt_to_enterprise_value_ratio) * returns \
        + debt_to_enterprise_value_ratio * interest_bearing_liabilities_interest_rate * (1 - income_tax_rate)
        
        # 用于填充YTM数据缺失值
        self.__interest_rate_1m_data = FCF_discounted_model_params_data['interest_rate_1m']

        # 返回各基本面数据表字典FCF_discounted_model_params_data
        return FCF_discounted_model_params_data
