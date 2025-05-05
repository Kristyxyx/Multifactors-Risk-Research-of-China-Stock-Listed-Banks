# %%
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from sklearn.feature_selection import RFECV

class CustomRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        random_state,
        format_input=True,
        format_output=True,
        dt_params_dist_1={
            'criterion': ['friedman_mse', 'absolute_error'],
            'max_depth': Integer(2, 10),
            'min_samples_split': Integer(2, 20),
            'max_features': Categorical(['sqrt', 'log2', None]),
        },
        ada_params_dist={
            'loss': ['linear', 'square'],
            'n_estimators': Integer(10, 100),
            'learning_rate': Real(1e-2, 1e0, prior='log-uniform'),
        },
        dt_params_dist_2={
            'criterion': ['friedman_mse', 'absolute_error'],
            'max_depth': Integer(2, 5), # 用于蒸馏ada2，因此应该浅一些
            'min_samples_split': Integer(2, 20),
            'max_features': Categorical(['sqrt', 'log2', None]), #
        },
        bayes_search_param={
            'n_iter': 20,
            'n_points': 5, # 使用比较大的n_initial_points参数可以显著减少"The objective has been evaluated at this point before"的警告
            'cv': 5,
            'scoring': 'neg_mean_squared_error',
            'n_jobs': -1,
            'verbose': 1
        },
        RFECV_param={
        'step': 0.05,
        'cv': 5,
        'scoring': 'neg_mean_squared_error',
        'min_features_to_select': 10
        }
    ):
        self.random_state = random_state
        self.format_input = format_input
        self.format_output = format_output

        # trian模式需要训练，从而需要定义用于训练的模型
        # 定义各模型的超参数区间及搜索模型
        self.dt_params_dist_1 = dt_params_dist_1
        self.ada_params_dist=ada_params_dist
        self.dt_params_dist_2 = dt_params_dist_2
        self.bayes_search_param = bayes_search_param
        self.RFECV_param = RFECV_param

        # 定义模型
        self.dt = DecisionTreeRegressor(random_state=self.random_state)
        self.ada = AdaBoostRegressor(estimator=None, random_state=self.random_state)

        self.bayes_search = BayesSearchCV(
            estimator=self.dt, # 形式上的参数，fit()中会更改，相当于None
            search_spaces=self.dt_params_dist_1, # 形式上的参数，fit()中会更改，相当于None
            random_state=self.random_state
        ).set_params(**self.bayes_search_param)

        self.rfecv = RFECV(estimator=self.dt).set_params(**self.RFECV_param)  # 形式上的参数`estimator`，fit()中会更改，相当于None

    def _format(self, X, mode):
        # Convert to numpy array if DataFrame or Series
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()

        # For 2D arrays with shape (n_samples, 1), which are likely target variables (y)
        # Convert to 1D array with shape (n_samples,)
        if len(X.shape) == 2 and X.shape[1] == 1 and mode == 'y':
            X = X.ravel()

        return X

    def fit(self, X, y):
        if self.format_input:
            X = self._format(X, mode='X')
            y = self._format(y, mode='y')

        # 传递模型的pipeline
        # self.model在fit前都需要转变为clone(self.model)防止__init__中的self.model被链接并拟合
        # 寻找最优决策树
        dt_1_bayes_search = clone(self.bayes_search).set_params(estimator=clone(self.dt), search_spaces=self.dt_params_dist_1)
        # 将一个估计器（如决策树dt）传递给BayesSearchCV的estimator参数时，在BayesSearchCV拟合后，这个估计器也会被拟合，并被设置为性能最好的超参数组合下拟合得到的决策树
        # 所以也需要对dt进行clone，防止__init__中的self.dt被链接并拟合
        dt_1_bayes_search.fit(X, y)
        dt_1_best = dt_1_bayes_search.best_estimator_

        # 将最优决策树输入adaboost，寻找最优adaboost
        ada_1 = clone(self.ada).set_params(estimator=dt_1_best)
        ada_1_bayes_search = clone(self.bayes_search).set_params(estimator=ada_1, search_spaces=self.ada_params_dist)
        ada_1_bayes_search.fit(X, y)
        self.ada_best_ = ada_1_bayes_search.best_estimator_

        if X.shape[1] >= 2:
            rfecv = clone(self.rfecv).set_params(estimator=self.ada_best_)
            rfecv.fit(X, y)
            # Store the RFECV support mask
            self.rfecv_support_ = rfecv.support_
            X_selected = rfecv.transform(X)

            # 将最优特征子集输入决策树，寻找最优决策树
            dt_2_bayes_search = clone(self.bayes_search).set_params(estimator=clone(self.dt), search_spaces=self.dt_params_dist_2)
            dt_2_bayes_search.fit(X_selected, y)
            dt_2_best = dt_2_bayes_search.best_estimator_

            # 将最优决策树输入adaboost，寻找最优adaboost
            ada_2 = clone(self.ada).set_params(estimator=dt_2_best)
            ada_2_bayes_search = clone(self.bayes_search).set_params(estimator=ada_2, search_spaces=self.ada_params_dist)
            ada_2_bayes_search.fit(X_selected, y)
            self.ada_best_ = ada_2_bayes_search.best_estimator_

        return self # 为了链式调用，即CustomRegressor.fit(X, y).xxx == self.xxx

    def predict(self, X, ada_best=None, rfecv_support_=None):
        if self.format_input and isinstance(X, (pd.DataFrame, pd.Series)):
            self.index = X.index
            X = self._format(X, mode='X')

        if ada_best is not None:
            self.ada_best_ = ada_best

        if rfecv_support_ is not None:
            self.rfecv_support_ = rfecv_support_

        # Apply feature selection if it was used during training
        if hasattr(self, 'rfecv_support_') and self.rfecv_support_ is not None:
            # If X is a numpy array
            X = X[:, self.rfecv_support_]

        if self.format_output and hasattr(self, 'index'):
            # Convert the predictions to a DataFrame with the original index
            return pd.DataFrame(self.ada_best_.predict(X), index=self.index)
        else:
            # If the input is not a DataFrame, return the predictions as a numpy array
            return self.ada_best_.predict(X)

    # analog fit_transform from transformer
    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)

class ComplementRegressor(BaseEstimator, RegressorMixin):
    """A regressor that returns 1 minus the predictions of a base regressor.

    Parameters
    ----------
    base_regressor : estimator object
        The base regressor whose predictions will be complemented.
    """

    def __init__(self, base_regressor=None):
        self.base_regressor = base_regressor

        # Clone the base_regressor to avoid modifying the original
        self.base_regressor_ = clone(self.base_regressor)

    def fit(self, X, y):
        """Fit the base regressor to the training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        self.base_regressor_.fit(X, y)
        return self

    def predict(self, X, y):
        """Return 1 minus the predictions of the base regressor.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Test data.
        y : array-like, shape (n_samples,)

        Returns
        -------
        y - self.base_regressor_.predict(X) : array-like, shape (n_samples,)
            The complement of the predictions of the base regressor.
        """
        # Return the complement
        return y - self.base_regressor_.predict(X)


# %%
import random
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import xgboost as xgb
from tslearn.clustering import KShape
from sklearn.preprocessing import StandardScaler
from econml.dml import CausalForestDML
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.tree import export_graphviz
import graphviz
import io
from PIL import Image
from factors_risks_dicts_generator import generate_factors_risks_dicts

class Modeler():

    def __init__(self, mode='train', factors_datas_names=[
        'factors_data',
        'fundamentals_data',
        'macros_data',
        'money_flows_data',
        'securities_margins_data',
        'industries_data',
        'indexes_data'
    ],
    other_datas_names=[
        'bank_stocks_info',
        'returns_data',
        'FCF_discounted_model_params_data'
    ],
    random_state=20250301
    ):
        self.mode = mode
        self.random_state = random_state
        # 将所有factors_datas以外的数据都定义为类属性
        for other_data_name in other_datas_names:
            other_data = pickle.load(open(f'./data/exported_data/{other_data_name}_{mode}.pkl', mode='rb+'))
            if other_data_name == 'bank_stocks_info':
                self.industry_stocks_info = other_data
            elif other_data_name == 'returns_data':
                other_data['market_returns'].rename(columns={'000001.XSHG': 0}, inplace=True)
                self.returns_data = other_data
            elif other_data_name == 'FCF_discounted_model_params_data':
                self.FCF_discounted_model_params_data = other_data

        self.index = self.FCF_discounted_model_params_data['r_wacc'].index
        self.columns = self.FCF_discounted_model_params_data['r_wacc'].columns

        # 合并多个因子数据表
        factors_datas = {}
        # 导入数据
        for factors_data_name in factors_datas_names:
            factors_data = pickle.load(open(f'./data/exported_data/{factors_data_name}_{mode}.pkl', mode='rb+'))
            factors_datas[factors_data_name] = factors_data

        self.factors_datas = factors_datas
        pickle.dump(obj=factors_datas, file=open(file=f'./data/exported_data/factors_datas_{self.mode}.pkl', mode='wb+'), protocol=4)

    # 检查各因子是否为空表
    def check_factor_data_nan(self):
        return {
            factors_data_name: [
                factor_data_name
                for factor_data_name, factor_data in factors_data.items()
                if not factor_data.any().any()
            ]
            for factors_data_name, factors_data in tqdm(
                self.factors_datas.items(),
                desc='handling factors datas missing values progress'
            )
        }

    def _fill_nan_col(self, factor_data_without_type1_missing, factor_data):
        #print('factor_data_without_type1_missing', factor_data_without_type1_missing)
        # 找到factor_data_without_type1_missing中全为缺失值的列名
        missing_stocks_codes = factor_data_without_type1_missing.columns[factor_data_without_type1_missing.isnull().all()]
        print('missing_stocks_codes', missing_stocks_codes)
        # 导出对应列在factor_data各行的分位数数据表，索引为缺失值列名
        missing_stocks_codes_quantiles = factor_data.rank(pct=True, axis=1)[missing_stocks_codes]
        # 计算对应列在factor_data各行的分位数据表的平均值series，索引为缺失值列名
        missing_stocks_codes_quantiles_mean = missing_stocks_codes_quantiles.mean()
        #print('missing_stocks_codes_quantiles_mean', missing_stocks_codes_quantiles_mean)
        # fill_values是factor_data_without_type1_missing中全为缺失值的各列在对应分位数平均值处的值series，索引为列名
        fill_values = factor_data_without_type1_missing.quantile(missing_stocks_codes_quantiles_mean, axis=1)

        # 兼容fill_values与factor_data_without_type1_missing
        fill_values = fill_values.T
        fill_values.columns = missing_stocks_codes

        # 用以上series填充对应缺失值列的缺失值
        factor_data_without_type1_missing.update(fill_values) # 不可将df赋给df，前者这样应该取values，变为np.array，但是还是会warning，所以使用update
        #print('fill_values', fill_values)
        #print('factor_data_without_type1_missing_filled', factor_data_without_type1_missing)

        return factor_data_without_type1_missing

    # 缺失值处理
    def _handle_missing_values(self, factor_data):
        '''factor_data中有些因子值缺失，而这样缺失值要么是由于个股在上市前或者退市后，因子值不存在；要么是因为个股在市期间，其因子值没有被披露或被统计。在进行缺失值
        处理时，忽略前一种缺失值，而填充后一种缺失值。
            填充缺失值一般有三种方法，即SimpleImputation，KNNImputation和IterativeImputation。对于本面板数据，SimpleImputation（如均值填充或中位数填充）可能不
        适合，因为它没有考虑时间序列的特性和个股之间的相关性。简单地用一个常数填充缺失值可能会引入偏差，尤其是当缺失值的比例较高时。而KNNImputation可以考虑个股之间的
        相关性，但它也没有考虑时间序列的特性。此外，KNNImputation在处理大规模面板数据时可能会比较慢，因为它需要计算所有个股之间的距离矩阵。所以应该选择IterativeImpu
        -tion。优缺点：考虑时间序列的特性，个股之间的相关性。
            然而，在使用IterativeImputer填充所有缺失值后再删除第一类缺失值可能不是最佳方案。这是因为IterativeImputer在估计缺失值时会考虑所有的特征，包括那些本不应
        该存在和被填充的第一类缺失值。这可能会影响估计的质量。
            所以，以下方案是更好的选择。首先，识别出那些不包含第一类缺失值的样本日期，并仅使用这些样本来训练IterativeImputer。然后，使用训练后的IterativeImputer来
        估计所有样本中的第二类缺失值。'''
        # 定义一个掩码mask,标识每个个股在每个时间点上是否处于上市状态
        mask = pd.DataFrame(index=self.index, columns=self.columns)
        # 传出因子数据表每一行axis=1的中位数，组成各截面的中位数向量medians
        medians = factor_data.median(axis=1)
        # 对于每个代码为stock_code的个股
        for stock_code in mask.columns:
            # 对于代码为stock_code的个股，查找板块个股代码列表industry_stocks_info对应的个股代码，传出其上市日期start_date和退市日日期end_date
            start_date = self.industry_stocks_info.loc[stock_code, 'start_date']
            end_date = self.industry_stocks_info.loc[stock_code, 'end_date']
            # 标识代码为stock的个股在每个时间点上是否处于上市状态
            mask[stock_code] = (mask.index >= start_date) & (mask.index <= end_date)

            # 对于代码为stock_code的个股，如果其在各时间上的因子值factor_data[stock_code]均为缺失值np.nan
            if factor_data[stock_code].isnull().all():
                #print('factor_data', factor_data, 'stock_code', stock_code)
                # 将此代码为stock_code的个股其在各时间上的因子值factor_data[stock_code]传为各截面的中位数向量medians，防止被Imputer忽略
                medians = pd.DataFrame(medians)
                medians.columns = [stock_code]
                factor_data.update(medians)
                #print('factor_data', factor_data)

        # 根据掩码mask，传出不包含第一类缺失值(不在市)的日期索引indexes_without_type1_missing
        indexes_without_type1_missing = mask.all(axis=1)
        # 定义不包含第一类缺失值的日期索引indexes_without_type1_missing对应的因子数据样本factor_data_without_type1_missing
        factor_data_without_type1_missing = factor_data.loc[indexes_without_type1_missing]

        # 对于factor_data_without_type1_missing中全为缺失值的列，得到其在整个数据factor_data中的分位数平均值，取此分位数在factor_data_without_type1_missing中各行对应的值填充缺失值列
        factor_data_without_type1_missing = self._fill_nan_col(factor_data_without_type1_missing, factor_data)

        # 定义IterativeImputer，所有缺失值被填充后需要再次加入训练，再次填充原有缺失值，直至缺失值收敛，这样的递归次数max_iter为50，随机种子random_state为self.random_state
        imputer = IterativeImputer(
            random_state=self.random_state,
            # 使用XGBoost填充缺失值
            estimator=xgb.XGBRegressor(),
            max_iter=50,
            tol=1e-3
            )

        # 使用factor_data_without_type1_missing来训练IterativeImputer
        imputer.fit(factor_data_without_type1_missing)
        # 利用imputer填充factor_data的全部缺失值，传出为填充后因子数据表factor_data_imputed
        factor_data_imputed = imputer.transform(factor_data)
        factor_data_imputed = pd.DataFrame(factor_data_imputed)
        factor_data_imputed.index = self.index
        factor_data_imputed.columns = self.columns

        # 将填充后因子数据表factor_data_imputed中的第一类缺失值重新标记为np.nan，即使用训练后的imputer来估计所有因子数据表中的第二类缺失值
        factor_data_imputed[~mask] = np.nan

        return factor_data_imputed

    # Fama-French-3分位数差值处理
    def _process_ff3_quantile_difference(self, factor_data):
        '''FF3处理形成截面股价收益率:
        合理性:如果您的研究目的是探究因子对股价收益率的截面预测能力,并且假设股价收益率的截面分布与因子的截面分布相关,那么按照FF3处理形成截面股价收益率是合适的。
        优点:这种方法能够消除股价收益率的极值影响,使得截面股价收益率的分布更加稳定,便于研究因子的预测能力。
        缺点:这种方法忽略了个股市值的影响,可能无法反映市场整体的收益率变化。'''
        panal_factor_data_quantiles = factor_data.quantile([0.3, 0.7], axis=1)

        return panal_factor_data_quantiles.loc[0.7] - panal_factor_data_quantiles.loc[0.3]

    # 企业价值加权处理
    def _average_by_enterprise_value(self, factor_data):
        '''按个股市值加权形成截面股价收益率:
        合理性:如果您的研究目的是探究因子对市场整体收益率的预测能力,并且假设个股的市值反映了其在市场中的重要性,那么按个股市值加权形成截面股价收益率是合适的。
        优点:这种方法考虑了个股市值的影响,能够反映市场整体的收益率变化,更接近实际的投资组合收益。
        缺点:这种方法可能受到大市值股票的主导,小市值股票的影响可能被掩盖。
        就本文的银行板块研究目的来说，选择按个股市值加权形成截面股价收益率。'''
        weighted_factor_data = self.enterprise_value_weights * factor_data
        panal_factor_data = weighted_factor_data.sum(axis=1)

        panal_factor_data = pd.DataFrame(panal_factor_data)
        panal_factor_data.index = self.index
        #panal_factor_data.columns = [factor_data_name] 不重置列名，防止与其他panal_factor_data运算时因为列名不一致而出现两行缺失值

        return panal_factor_data

    # 数据清理，即缺失值处理和Fama-French-3分位数差值处理
    def clean_and_average_factors_datas(self):
        self.enterprise_value_weights = self.FCF_discounted_model_params_data['panal_enterprise_value_weights']
        industry_factors_datas = self.factors_datas.copy()

        for factors_data_name, factors_data in self.factors_datas.items():
            for factor_data_name, factor_data in tqdm(factors_data.items(), desc='handling factors data progress'):
                if factors_data_name != 'macros_data': #factor_data_name == 'PEG':
                    # 缺失值处理
                    try:
                        factor_data_imputed = self._handle_missing_values(factor_data)
                    except:
                        print(factor_data_name, factors_data_name)
                        raise

                    # 企业价值加权处理
                    industry_factor_data = self._average_by_enterprise_value(factor_data_imputed)
                    # 对r_wacc以enterprise_value在截面的权重加权求和求出enterprise_value加权r_wacc，
                    #分别求出circulating_value加权r_E和Debts加权r_D，再以截面总circulating_value和总Debts在截面的总enterprise_value的权重加权得到enterprise_value加权r_wacc
                    # 二者过程等价
                    industry_factors_datas[factors_data_name][factor_data_name] = industry_factor_data

        self.industry_factors_datas = industry_factors_datas

        return industry_factors_datas

    def clean_and_average_r_wacc_data(self):
        self.enterprise_value_weights = self.FCF_discounted_model_params_data['panal_enterprise_value_weights']
        r_wacc = self.FCF_discounted_model_params_data['r_wacc']

        r_wacc_imputed = self._handle_missing_values(r_wacc)

        industry_r_wacc = self._average_by_enterprise_value(r_wacc_imputed)

        industry_r_wacc_data = {'r_wacc': industry_r_wacc}

        self.industry_r_wacc_data = industry_r_wacc_data

        return industry_r_wacc_data

    def transform_factors_datas_from_dict_to_df(self):
        """Transform dictionary of factors data into a single DataFrame more efficiently using concat."""
        # Create a list to hold all individual factor DataFrames
        factor_dfs = []

        # Iterate through all industry_factors_data dictionaries
        for _, factors_data in self.industry_factors_datas.items():
            # For each factor in the industry_factors_data dictionary
            for factor_data_name, factor_data in factors_data.items():
                # Create a copy of the DataFrame with the factor name as column name
                factor_df = factor_data.copy()
                # To prevent column name conflicts when factors have the same name across different data types,
                # use a unique identifier by combining factor name with data source
                factor_df.columns = [factor_data_name]
                factor_dfs.append(factor_df)

        # Concatenate all factor DataFrames at once (horizontally)
        factors_df = pd.concat(factor_dfs, axis=1)
        
        # Create an IterativeImputer to fill any remaining missing values
        imputer = IterativeImputer(
            random_state=self.random_state,
            # Use XGBoost for imputation
            estimator=xgb.XGBRegressor(),
            max_iter=50,
            tol=1e-3
        )
    
        # Apply the IterativeImputer to fill missing values
        industry_factors_df = imputer.fit_transform(factors_df)
        # Convert back to DataFrame with original index and column names
        industry_factors_df = pd.DataFrame(industry_factors_df, index=self.index, columns=factors_df.columns)
    
        # Store the result as an instance variable
        self.industry_factors_df = industry_factors_df
        return industry_factors_df

    # 定义标准化因子风险归类数据表
    def standardize_factors_risks_data(self):
        # 导入手动划分的因子风险归类数据表factors_risks_data，其有三列，分别是因子中文名称、因子代码和因子所属风险
        try:
            factors_risks_data = pd.read_csv('./data/dict/factors_risks_dicts.csv')
        except FileNotFoundError:
            generate_factors_risks_dicts()

        # 定义因子代码列表
        factors_codes = self.industry_factors_df.columns.tolist()

        # 只保留最后一个重复值
        factors_risks_data.drop_duplicates(subset=['factor_code'], keep='last', inplace=True)
        factors_risks_data = factors_risks_data.set_index('factor_code')
        # 将factors_risks_data中的risk列标准化为0-1变量
        factors_risks_data_standardized = pd.DataFrame(0, columns=['default_risk', 'liquidity_risk', 'market_risk'], index=factors_risks_data.index)

        # 使用向量化操作直接设置对应的风险类型为1
        risk_series = factors_risks_data['risk']
        factors_risks_data_standardized['default_risk'] = risk_series.str.contains('Default Risk', regex=False).astype(int)
        factors_risks_data_standardized['liquidity_risk'] = risk_series.str.contains('Liquidity Risk', regex=False).astype(int)
        factors_risks_data_standardized['market_risk'] = risk_series.str.contains('Market Risk', regex=False).astype(int)

        # 检查是否有缺失的因子代码
        missing_factors = [code for code in factors_codes if code not in factors_risks_data_standardized.index]
        
        # 为缺失的因子代码添加新行，用随机的0或1填充风险列
        if missing_factors:
            # 设置随机种子以保持可重复性
            random.seed(self.random_state)
            
            # 为每个缺失的因子创建新的行
            missing_data = pd.DataFrame(
            index=missing_factors,
            columns=['default_risk', 'liquidity_risk', 'market_risk'],
            data=[[random.randint(0, 1) for _ in range(3)] for _ in range(len(missing_factors))]
            )
            
            # 添加到标准化的数据框中
            factors_risks_data_standardized = pd.concat([factors_risks_data_standardized, missing_data])
        
        # 删除factors_risks_data_standardized中不在factors_codes中的行
        factors_risks_data_standardized = factors_risks_data_standardized.loc[factors_risks_data_standardized.index.isin(factors_codes)]

        # 使用KShape对factor时间序列进行聚类
        # 提取factor时间序列
        factors_ts = self.industry_factors_df.copy()

        # 标准化时间序列数据用于聚类
        scaler = StandardScaler()
        factors_ts_scaled = pd.DataFrame(
            scaler.fit_transform(factors_ts),
            index=factors_ts.index,
            columns=factors_ts.columns
        )

        # 定义风险类型
        init_centroids = []
        risk_types = ['default_risk', 'liquidity_risk', 'market_risk']

        for risk_type in risk_types:
            # 获取属于该风险类型的因子名称列表
            risk_factors_codes = factors_risks_data_standardized.loc[
                factors_risks_data_standardized[risk_type] == 1
            ].index.tolist()

            # 从标准化后的时间序列数据中提取这些因子的数据并计算平均值
            risk_centroid = factors_ts_scaled[risk_factors_codes].mean(axis=1).values
            init_centroids.append(risk_centroid)

        # 转换为正确的形状以适配KShape要求
        init_centroids = np.array(init_centroids)
        init_centroids = init_centroids.reshape(len(risk_types), len(factors_ts_scaled), 1)
        # 这里的`init_centroids`是一个3D数组，表示3个初始质心，每个质心对应一个聚类中心

        # 执行KShape聚类
        n_clusters = 3  # 假设我们希望聚类为3种风险类型
        kshape = KShape(n_clusters=n_clusters, random_state=self.random_state, init=init_centroids)

        # 创建适合KShape的输入数据格式
        factors_ts_tensor = np.zeros((len(factors_ts_scaled.columns), len(factors_ts_scaled), 1))
        factors_ts_tensor = factors_ts_scaled.values.T.reshape(len(factors_ts_scaled.columns), len(factors_ts_scaled), 1)

        kshape.fit(factors_ts_tensor)

        # 获取聚类标签
        cluster_labels = kshape.labels_

        # 将聚类标签重构为新的factors_risks_data_standardized
        # Map cluster IDs to column indices
        for i, risk_type in enumerate(risk_types):
            # For each risk type, set 1 for factors that belong to that cluster
            factors_risks_data_standardized[risk_type] = (cluster_labels == i).astype(int)

        self.industry_factors_df = factors_ts_scaled
        self.factors_risks_data_standardized = factors_risks_data_standardized

        return factors_risks_data_standardized

    def visualize_risk_factors_clusters(
        self,
        windows=(1, 30, 120),
        risk_types=None,
        num_factors=3,
        colors=None,
        linestyles=None,
        figsize=None,
        random_seed=None
    ):
        """
        Visualize moving averages of risk factors grouped by risk type.
        
        Parameters:
        -----------
        windows : tuple
            Window sizes for moving averages
        risk_types : list, optional
            List of risk types to include. If None, uses ['default_risk', 'liquidity_risk', 'market_risk']
        num_factors : int
            Number of factors to randomly sample for each risk type
        colors : list, optional
            List of colors for plotting different factors. If None, uses ['blue', 'green', 'red']
        linestyles : tuple, optional
            Tuple of linestyles for different window sizes. If None, uses (':', '--', '-')
        figsize : tuple, optional
            Figure size (width, height) in inches. If None, calculates based on number of rows and columns
        random_seed : int, optional
            Random seed for reproducibility when sampling factors. If None, uses self.random_state
        
        Returns:
        --------
        matplotlib.figure.Figure
            The figure containing the visualized moving averages
        """
        # Set default values
        if risk_types is None:
            risk_types = ['default_risk', 'liquidity_risk', 'market_risk']
        if colors is None:
            colors = ['blue', 'green', 'red']
        if linestyles is None:
            linestyles = (':', '--', '-')
        if random_seed is None:
            random_seed = self.random_state
            
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Group factors by risk type
        sampled_risks_factors_codes = {}
        
        # For each risk type, randomly select factors
        for risk_type in risk_types:
            # Get factors that belong to this risk type (where value is 1)
            risk_factors_codes = self.factors_risks_data_standardized[
                self.factors_risks_data_standardized[risk_type] == 1
            ].index.tolist()
            
            # Randomly sample factors (or fewer if not enough)
            sampled_risks_factors_codes[risk_type] = random.sample(
                risk_factors_codes,
                k=min(num_factors, len(risk_factors_codes))
            )
        
        # Flatten the dictionary to get all sampled factors
        sampled_factors_codes = [
            sampled_risk_factor_code
            for sampled_risk_factors_codes in sampled_risks_factors_codes.values()
            for sampled_risk_factor_code in sampled_risk_factors_codes
        ]
        
        # Create displayed dataframe with these factors
        sampled_factors_ts = self.industry_factors_df[sampled_factors_codes]
        
        # Setup subplot grid dimensions
        num_row = len(windows)
        num_cols = len(risk_types)
        
        # Calculate figure size if not provided
        if figsize is None:
            figsize = (5 * num_cols, 5 * num_row)
        
        # Create figure and axes
        fig, axes = plt.subplots(
            nrows=num_row,
            ncols=num_cols,
            figsize=figsize,
            squeeze=False
        )
        
        # First loop through windows (rows)
        for i, window in enumerate(windows):
            linestyle = linestyles[i] if i < len(linestyles) else '-'
            # Then loop through risk types (columns)
            for j, risk_type in enumerate(risk_types):
                # Get the factors for this risk type
                sampled_risk_factors_codes = sampled_risks_factors_codes[risk_type]

                # Plot each factor in this risk type
                for k, sampled_risk_factor_code in enumerate(sampled_risk_factors_codes):
                    # Calculate moving average
                    sampled_risk_factors_ts = sampled_factors_ts[sampled_risk_factor_code].rolling(window=window).mean()
                    # Plot with different colors for different factors
                    axes[i, j].plot(
                        self.index,
                        sampled_risk_factors_ts,
                        label=sampled_risk_factor_code,
                        color=colors[k % len(colors)],
                        linestyle=linestyle
                    )

                # Set labels and title
                axes[i, j].set_title(f'{window}-Day MA for {risk_type}')
                axes[i, j].set_xlabel('Date')
                axes[i, j].set_ylabel('Value')
                axes[i, j].grid(True)
                axes[i, j].legend()

        plt.tight_layout()

        return fig

    # 取得系统性风险和非系统性风险风险溢价
    def get_industry_risks_premiums_totals(self, risks_premiums_totals_curs=None):
        self.industry_risks_premiums_totals = {}

        if self.mode == 'train':
            assert risks_premiums_totals_curs is None, 'train模式下，风险溢价模型的初始值为None'
            self.risks_premiums_totals_curs = {}
        elif self.mode == 'test':
            assert risks_premiums_totals_curs is not None, 'test模式下，风险溢价模型的初始值不能为None'
            self.risks_premiums_totals_curs = risks_premiums_totals_curs

        # 载入数据
        r_wacc = self.industry_r_wacc_data['r_wacc']
        interest_rate_1m = self.FCF_discounted_model_params_data['interest_rate_1m']
        market_returns = self.returns_data['market_returns']

        # 定义r_wacc风险溢价
        self.industry_risk_premium = r_wacc - interest_rate_1m
        market_risk_premium_common = market_returns - interest_rate_1m

        cur = CustomRegressor(random_state=self.random_state)
        if self.mode == 'train':
            industry_risk_premium_common = cur.fit_predict(market_risk_premium_common, self.industry_risk_premium)
            self.risks_premiums_totals_curs['common_risk'] = cur

            # 创建一个新的CustomRegressor对象用于idiosyncratic_risk
            idio_cur = CustomRegressor(random_state=self.random_state)
            idio_cur.ada_best_ = ComplementRegressor(
            base_regressor=cur.ada_best_
            )
            self.risks_premiums_totals_curs['idiosyncratic_risk'] = idio_cur

        elif self.mode == 'test':
            industry_risk_premium_common = cur.predict(
                market_risk_premium_common,
                self.risks_premiums_totals_curs['common_risk'].ada_best_,
                self.risks_premiums_totals_curs['common_risk'].rfecv_support_
            )

        self.industry_risks_premiums_totals['common_risk'] = industry_risk_premium_common
        industry_risk_premium_idiosyncratic = self.industry_risk_premium - industry_risk_premium_common
        self.industry_risks_premiums_totals['idiosyncratic_risk'] = industry_risk_premium_idiosyncratic

        return self.industry_risks_premiums_totals, self.risks_premiums_totals_curs

    def get_industry_risks_premiums_components(self, risks_premiums_components_curs=None):
        self.industry_risks_premiums_components = {}

        if self.mode == 'train':
            assert risks_premiums_components_curs is None, 'train模式下，风险溢价模型的初始值为None'
            self.risks_premiums_components_curs = {}
        elif self.mode == 'test':
            assert risks_premiums_components_curs is not None, 'test模式下，风险溢价模型的初始值不能为None'
            self.risks_premiums_components_curs = risks_premiums_components_curs

        for risk_premium_component_name in {'default_risk', 'liquidity_risk','market_risk'}:
            risk_premium_component_factors_codes = self.factors_risks_data_standardized.loc[self.factors_risks_data_standardized[risk_premium_component_name] == 1].index.tolist()
            risk_premium_component_factors_df = self.industry_factors_df[risk_premium_component_factors_codes]

            cur = CustomRegressor(random_state=self.random_state)
            if self.mode == 'train':
                risk_premium_component = cur.fit_predict(risk_premium_component_factors_df, self.industry_risk_premium)
                self.risks_premiums_components_curs[risk_premium_component_name] = cur

            elif self.mode == 'test':
                risk_premium_component = cur.predict(
                    risk_premium_component_factors_df,
                    self.risks_premiums_components_curs[risk_premium_component_name]
                    .ada_best_,
                    self.risks_premiums_components_curs[risk_premium_component_name].rfecv_support_
                )

            self.industry_risks_premiums_components[risk_premium_component_name] = risk_premium_component

        return self.industry_risks_premiums_components, self.risks_premiums_components_curs

    def estimate_causal_effects_with_dml(self):
        """
        使用DynamicDML估计三种风险溢价与总风险溢价之间的因果关系

        针对每个风险溢价组件(default_risk, liquidity_risk, market_risk)作为处理变量，
        其他两个风险溢价组件作为混淆变量，估计其对系统性和非系统性风险溢价的因果效应

        Returns:
            dict: 嵌套字典，格式为{处理变量: {结果变量: DynamicDML模型}}
        """

        # 初始化结果字典
        causal_forest_forests = {}

        # 风险溢价组件
        risk_components = ['default_risk', 'liquidity_risk', 'market_risk']
        # 总风险溢价
        risk_totals = ['common_risk', 'idiosyncratic_risk']

        # 创建控制变量
        index = pd.to_datetime(self.index)

        # 第1个控制变量就是从1到len(self.index)的整数序列，表示时间序列的趋势性
        X_1 = pd.Series(range(1, len(index) + 1), index=index, name='control_variable')

        # 第2个到第12个控制变量表示self.index是否在某月份内，表示时间序列的季节性
        X_2_to_12 = pd.get_dummies(index.month, prefix='month', drop_first=True)
        X_2_to_12.index = index

        # 将第1个控制变量与第2个到第12个控制变量合并为一个DataFrame
        X = pd.concat([X_1, X_2_to_12], axis=1)
        X.index = self.index

        # 遍历每一个风险溢价作为处理变量
        for treatment in risk_components:
            causal_forest_forests[treatment] = {}

            # 选择其他两个风险溢价作为混淆变量
            confounders = [
                confounder for confounder in risk_components
                if confounder != treatment
            ]

            # 准备处理变量数据
            T = self.industry_risks_premiums_components[treatment].to_numpy().ravel()

            # 准备混淆变量数据 - 合并为一个DataFrame
            W = pd.concat(
                [
                    self.industry_risks_premiums_components[confounder] for confounder in confounders
                ],
                axis=1
            )
            W.columns = confounders

            # 遍历每一个总风险溢价作为结果变量
            for outcome in risk_totals:
                # 准备结果变量数据
                Y = self.industry_risks_premiums_totals[outcome].to_numpy().ravel()

                # 创建DynamicDML模型
                cfdml = CausalForestDML(
                    model_y=CustomRegressor(random_state=self.random_state, format_input=False),
                    model_t=CustomRegressor(random_state=self.random_state, format_input=False),
                    cv=5,  # 交叉验证折数
                    n_estimators=200,  # 树的数量
                    max_depth=6,  # 最大树深度
                    min_samples_leaf=10,  # 叶节点最小样本数
                    max_features='sqrt',  # 特征抽样策略
                    honest=True,  # 使用诚实树提高推断可靠性
                    random_state=self.random_state
                )

                # 拟合模型
                cfdml.fit(Y, T, X=X, W=W)

                # 存储模型
                causal_forest_forests[treatment][outcome] = cfdml

        # 将结果存储为类属性以便后续分析
        self.causal_forest_forests = causal_forest_forests

        return causal_forest_forests

    def _construct_causal_data(self, treatment, outcome):
        """Generate enhanced visualizations for the specified causal model."""
        # Get the model
        model = self.causal_forest_forests[treatment][outcome]
        
        # Prepare the data
        index = pd.to_datetime(self.index)
        X_1 = pd.Series(range(1, len(index) + 1), index=index, name='days_sequence')
        X_2_to_12 = pd.get_dummies(index.month, prefix='month', drop_first=True)
        X_2_to_12.index = index
        X = pd.concat([X_1, X_2_to_12], axis=1)
        X.index = self.index
        
        # Choose other risk components as confounders
        risk_components = ['default_risk', 'liquidity_risk', 'market_risk']
        confounders = [c for c in risk_components if c != treatment]
        W = pd.concat([self.industry_risks_premiums_components[c] for c in confounders], axis=1)
        W.columns = confounders

        # Treatment and outcome variables
        T = self.industry_risks_premiums_components[treatment].to_numpy().ravel()
        Y = self.industry_risks_premiums_totals[outcome].to_numpy().ravel()

        # Generate enhanced visualizations
        return model, X, W, T, Y

    def visualize_causal_forest_trees(self):
        """
        Visualize the last decision tree for each combination of risk component and risk outcome.

        Returns:
            matplotlib.figure.Figure: The figure containing the visualized trees
        """
        # Define risk types and outcomes to visualize
        risk_components = ['default_risk', 'liquidity_risk', 'market_risk']
        risk_totals = ['common_risk', 'idiosyncratic_risk']

        # Create a figure with more vertical space for titles
        fig, axes = plt.subplots(
            len(risk_totals),
            len(risk_components),
            figsize=(30, 23),
            dpi=300
        )

        # Then proceed with the individual tree visualizations
        for row_idx, outcome in enumerate(risk_totals):
            for col_idx, treatment in enumerate(risk_components):
                # Get the current axis
                ax = axes[row_idx, col_idx]

                # Get the CausalForestDML model
                cfdml_model = self.causal_forest_forests[treatment][outcome]

                # Access the forest estimator from the CATE model
                # Note: Accessing internal attributes like model_cate and estimators_ can be fragile
                forest = cfdml_model.model_cate.estimators_[0]

                # Get all trees in the forest
                estimators = forest.estimators_
                n_estimators = len(estimators)

                # Get only the last tree
                tree = estimators[-1]

                # Create temp file for DOT data
                dot_data = io.StringIO()

                # Define feature names based on the structure used in _construct_causal_data
                feature_names = ['days_sequence'] + [f'month_{m}' for m in range(2, 13)]

                # Export tree to DOT format
                export_graphviz(
                    tree,
                    out_file=dot_data,
                    feature_names=feature_names, # Use defined feature names
                    filled=True,
                    rounded=True,
                    special_characters=True,
                    proportion=True # Show proportions instead of counts
                )

                # Create graphviz object
                graph = graphviz.Source(dot_data.getvalue())

                # Render to PNG
                png_data = graph.pipe(format='png')

                # Convert to image
                img = Image.open(io.BytesIO(png_data))

                # Display image
                ax.imshow(np.array(img))

                # Set subplot titles with smaller fontsize and better positioning
                title_text = f"{treatment} to {outcome}\nTree #{n_estimators}/{n_estimators}"
                ax.set_title(title_text, fontsize=10, pad=2) # Adjusted fontsize and padding

                # Remove ticks and labels but keep the frame (border) visible
                ax.set_xticks([])
                ax.set_yticks([])
                # Ensure the frame/border is visible
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)

        # Add overall title with more space above the plot
        plt.suptitle(
            "$\mathsf{CausalForest}$'s Last $\mathsf{DecisionTree}$ for Effect $\operatorname{CATE}(z)$ of Each Risk Component on Each Risk Outcome",
            fontsize=20
        )

        # Adjust layout to make room for titles
        plt.tight_layout()
        #plt.subplots_adjust(top=1)  # Adjust top margin to make space for suptitle

        # Return the figure without showing it
        return fig

    def visualize_causal_effect(self, treatment_name, outcome_name):
        """
        Generate comprehensive visualizations for causal effects analysis.

        Parameters:
            treatment_name (str): Name of the treatment variable
            outcome_name (str): Name of the outcome variable

        Returns:
            list: A list of matplotlib figure objects for the generated visualizations
        """
        # Get effects and their confidence intervals
        model, X, _, T, Y = self._construct_causal_data(treatment_name, outcome_name)

        # Initialize list to store figures
        figures = []

        effects = model.effect(X)
        effect_inference = model.effect_inference(X)
        effects_intervals = effect_inference.conf_int()
        # Get ATE as a scalar value
        ate = model.ate(X)

        # Get p-value for ATE - properly handle whether it's a method or attribute
        ate_inference = model.ate_inference(X)
        p_value = ate_inference.pvalue()

        # ---------- 1. CATE Over Time ----------
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        # Sort by time
        time_values = X['days_sequence'].values
        sort_idx = np.argsort(time_values)
        sorted_time = time_values[sort_idx]
        sorted_effects = effects[sort_idx]
        # Plot CATE line and individual points (outliers/scatter)
        ax1.plot(
            sorted_time,
            sorted_effects,
            linewidth=1.5,
            label='$\operatorname{CATE}$'
        )
        ax1.scatter(
            sorted_time,
            sorted_effects,
            alpha=0.3,
            s=10
        ) # Removed label for points

        # Ensure the intervals are 1D arrays and sort them
        lower_ci = np.ravel(effects_intervals[0])[sort_idx]
        upper_ci = np.ravel(effects_intervals[1])[sort_idx]
        # Add label for Confidence Interval
        ax1.fill_between(
            sorted_time,
            lower_ci,
            upper_ci,
            alpha=0.2,
            color='blue',
            label='95% CI'
        )

        # Add trend line
        # Ensure sorted_effects is 1D
        sorted_effects_1d = np.ravel(sorted_effects)
        z = np.polyfit(sorted_time, sorted_effects_1d, 3)  # Cubic fit
        p = np.poly1d(z)
        x_range = np.linspace(min(sorted_time), max(sorted_time), 100)
        ax1.plot(x_range, p(x_range), "r--", label="Trend (Polyfit)", linewidth=2)
        ax1.axhline(
            y=0,
            color='gray',
            linestyle='--',
            alpha=0.7,
            label='Zero Effect' # Added label
        )
        ax1.axhline(
            y=ate,
            color='red',
            linestyle='-',
            alpha=0.7,
            label=f'ATE: {ate:.4f} (p={p_value:.4f})'
        )

        ax1.set_title(
            f"Temporal Effect $\operatorname{{CATE}}$ on {outcome_name} by {treatment_name}",
            fontsize=14
        )
        ax1.set_xlabel("$\operatorname{Day}$", fontsize=12)
        ax1.set_ylabel("$\operatorname{CATE}$", fontsize=12)

        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10) # Unified legend call
        plt.tight_layout()
        figures.append(fig1)

        # ---------- 2. CATE by Month ----------
        fig2, ax2 = plt.subplots(figsize=(12, 6))

        # Calculate effects by month
        month_effects = []
        month_errors = []
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        valid_months = []

        # First handle January (which is typically the reference month in dummy encoding)
        is_jan = np.ones(len(X), dtype=bool)
        for col in X.columns:
            if col.startswith('month_'):
                is_jan = is_jan & (X[col] == 0)

        if np.any(is_jan):
            month_effects.append(np.mean(effects[is_jan]))
            month_errors.append(np.std(effects[is_jan]) / np.sqrt(np.sum(is_jan)))
            valid_months.append('Jan')

        # Handle other months
        for i, month in enumerate(month_names[1:], 2):  # Start from February (2)
            col = f'month_{i}'
            if col in X.columns:
                is_month = X[col] == 1
            if np.any(is_month):
                month_effects.append(np.mean(effects[is_month]))
                month_errors.append(
                np.std(effects[is_month]) / np.sqrt(np.sum(is_month))
                )
                valid_months.append(month)

        # Plot as bar chart with error bars
        x_pos = np.arange(len(valid_months))
        # Added label for bars (representing mean effect per month)
        ax2.bar(
            x_pos,
            month_effects,
            yerr=month_errors,
            align='center',
            alpha=0.7,
            color='skyblue',
            ecolor='black',
            capsize=10,
            label='Monthly Mean CATE (Error Bars: SE)' # Label for bars and error bars
        )
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(valid_months)

        ax2.set_title(
            f"Seasonal Effects $\operatorname{{CATE}}$ on {outcome_name} by {treatment_name}",
            fontsize=14
        )
        ax2.set_ylabel('$\operatorname{CATE}$', fontsize=12)
        ax2.set_xlabel('$\operatorname{isMonth}$', fontsize=12)
        ax2.axhline(
            y=0,
            color='gray',
            linestyle='--',
            alpha=0.7,
            label='Zero Effect' # Added label
        )
        ax2.axhline(
            y=ate,
            color='red',
            linestyle='-',
            alpha=0.7,
            label=f'ATE: {ate:.4f} (p={p_value:.4f})'
        )
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=10) # Unified legend call
        plt.tight_layout()
        figures.append(fig2)

        # ---------- 3. CATE by Treatment ----------
        fig3, ax3 = plt.subplots(figsize=(12, 6))

        # Clean T by replacing infinite values with NaN
        T_array = np.array(T)

        # Sort T_filtered and effects_filtered based on T_filtered values
        sort_idx_t = np.argsort(T_array)
        sorted_T = T_array[sort_idx_t]
        sorted_effects_t = effects[sort_idx_t]

        # Plot CATE vs Treatment line and points
        ax3.plot(
            sorted_T,
            sorted_effects_t,
            linewidth=1.5,
            alpha=0.8,
            color='darkgreen',
            label='$\operatorname{CATE}$' # Updated label
        )
        ax3.scatter(
            sorted_T,
            sorted_effects_t,
            alpha=0.4,
            s=10,
            color='green'
        ) # Removed label for points

        # Get and plot confidence intervals for CATE (using original effect inference)
        lower_ci_t = np.ravel(effects_intervals[0])[sort_idx_t]
        upper_ci_t = np.ravel(effects_intervals[1])[sort_idx_t]

        # Plot confidence intervals with label
        ax3.fill_between(
            sorted_T,
            lower_ci_t,
            upper_ci_t,
            alpha=0.2,
            color='mediumseagreen',
            label='95% CI' # Updated label
        )

        # Add a trend line
        # Using polynomial fit on sorted data
        sorted_effects_t_1d = np.ravel(sorted_effects_t)
        # Calculate polynomial coefficients for cubic fit
        z_t = np.polyfit(sorted_T, sorted_effects_t_1d, 3)
        # Create polynomial function
        p_t = np.poly1d(z_t)
        # Generate x-values for the trend line
        t_range = np.linspace(min(sorted_T), max(sorted_T), 100)
        # Plot the trend line
        ax3.plot(t_range, p_t(t_range), "r--", label="Trend (Polyfit)", linewidth=2)

        # Add ATE and zero lines
        ax3.axhline(
            y=0,
            color='gray',
            linestyle='--',
            alpha=0.7,
            label='Zero Effect' # Added label
        )
        ax3.axhline(
            y=ate,
            color='red',
            linestyle='-',
            alpha=0.7,
            label=f'ATE: {ate:.4f} (p={p_value:.4f})'
        )

        # Set title and labels
        ax3.set_title(
            f"Effect $\operatorname{{CATE}}$ on {outcome_name} by {treatment_name}", # Updated title
            fontsize=14
        )
        ax3.set_xlabel(f'$T$: {treatment_name}', fontsize=12)
        ax3.set_ylabel('$\operatorname{CATE}$', fontsize=12) # Updated Y label

        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10) # Unified legend call
        plt.tight_layout()
        figures.append(fig3)

        # ---------- 4. CATE Distribution ----------
        fig4, ax4 = plt.subplots(figsize=(12, 6))

        # Ensure effects is 1D for histogram
        effects_1d = np.ravel(effects)

        # Plot histogram with kernel density estimate
        sns.histplot(
            effects_1d,
            kde=True,
            ax=ax4,
            bins=30,
            color='purple',
            alpha=0.6
        )

        # Add a rug plot to show individual effects
        sns.rugplot(
            effects_1d,
            ax=ax4,
            color='black',
            alpha=0.3
        )

        ax4.set_title(
            f"Distribution of Effects $\operatorname{{CATE}}(Z)$ of {treatment_name} on {outcome_name}",
            fontsize=14
        )
        ax4.set_xlabel("$\operatorname{CATE}(Z)$", fontsize=12)
        ax4.set_ylabel("$\hat{p}_n \left( \operatorname{CATE}(Z) \\right)$", fontsize=12)

        ax4.axvline(
            x=ate,
            color='red',
            linestyle='-',
            linewidth=2,
            label=f'ATE: {ate:.4f} (p={p_value:.4f})'
        )
        ax4.axvline(
            x=0,
            color='gray',
            linestyle='--',
            alpha=0.7,
            label='Zero Effect' # Added label for the zero line
        )

        ax4.grid(alpha=0.3)
        ax4.legend() # Display the legend with labels for ATE and Zero Effect
        plt.tight_layout()
        figures.append(fig4)

        # ---------- 5. Treatment vs. Outcome by Season ----------
        fig5, ax5 = plt.subplots(figsize=(12, 6))

        # Explicitly replace infinity values with NaN
        Y_array = np.array(Y)

        # Divide time into quarters
        q_labels = ["Q1", "Q2", "Q3", "Q4"]
        time_quartiles = pd.qcut(X['days_sequence'], 4, labels=q_labels)

        # Create a colormap
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, 4))

        # Plot each quartile with different colors
        for i, (label, color) in enumerate(zip(q_labels, colors)):
            mask = time_quartiles == label
            ax5.scatter(
            T_array[mask],
            Y_array[mask],
            alpha=0.6,
            color=color,
            label=label # Label for scatter points (used in legend)
            )

        # Add regression lines for each quartile
        for i, (label, color) in enumerate(zip(q_labels, colors)):
            mask = time_quartiles == label
            # Filter out potential NaNs or Infs before polyfit
            z = np.polyfit(T_array[mask], Y_array[mask], 1)
            p = np.poly1d(z)
            # Define x_range based on valid data for the quartile
            t_min, t_max = np.min(T_array[mask]), np.max(T_array[mask])
            x_range = np.linspace(t_min, t_max, 20)
            # Plot regression line - no separate label, associated by color
            ax5.plot(x_range, p(x_range), color=color, linestyle='--')

        ax5.set_title(
            f"{treatment_name}-{outcome_name} Relationship: $Y = \\theta(Z) T + g(X) + \epsilon_Y$ by Season",
            fontsize=14
        )

        ax5.set_xlabel(f"${{T}}$: {treatment_name}", fontsize=12)
        ax5.set_ylabel(f"${{Y}}$: {outcome_name}", fontsize=12)

        ax5.grid(alpha=0.3)
        # Legend will show labels from scatter plot ('Q1', 'Q2', 'Q3', 'Q4')
        ax5.legend(title="Time Period")
        plt.tight_layout()
        figures.append(fig5)

        # ---------- 6. Feature Importance for CATE ----------
        #try:
        feature_importances = model.feature_importances()

            # Ensure feature_importances is 1D
            #if len(feature_importances.shape) > 1:
                #feature_importances = feature_importances.flatten()

            # Only plot if we have feature importances
            #if len(feature_importances) > 0:
        fig6, ax6 = plt.subplots(figsize=(10, 6))

        # Get feature names
        feature_names = X.columns.tolist()

        ## Ensure lengths match
        #min_length = min(len(feature_importances), len(feature_names))
        #valid_features = feature_importances[:min_length]
        #valid_names = feature_names[:min_length]

        # Sort feature importances (ascending)
        sort_idx = np.argsort(feature_importances)

        # Take at most 10 most important features
        if len(sort_idx) > 10:
            sort_idx = sort_idx[-10:]

        # Get sorted values
        sorted_importances = feature_importances[sort_idx]
        sorted_feature_names = [feature_names[i] for i in sort_idx]

        # Plot horizontal bar chart
        ax6.barh(
            range(len(sorted_importances)),
            sorted_importances, align='center',
            color='teal'
        )
        ax6.set_yticks(range(len(sorted_importances)))
        ax6.set_yticklabels(sorted_feature_names)

        ax6.set_title("Feature Importance for $\mathsf{CauslForest}$", fontsize=14)
        ax6.set_xlabel("Importance", fontsize=12)
        ax6.grid(alpha=0.3)
        plt.tight_layout()
        figures.append(fig6)
        #except Exception as e:
            #print(f"Could not plot feature importance: {e}")

        return figures


# %%
if __name__ == '__main__':
    #import pickle
    #from Modeler import Modeler

    mode = input('Dataset is for train or test?')
    mol = Modeler(mode)

    # %%
    mol.check_factor_data_nan()

    # %%
    try:
        mol.industry_factors_datas = pickle.load(open('./data/modeled_data/bank_factors_datas_train.pkl', mode='rb+'))
    except:
        mol.industry_factors_datas = mol.clean_and_average_factors_datas()
        pickle.dump(obj=mol.industry_factors_datas, file=open(file='./data/modeled_data/bank_factors_datas_train.pkl', mode='wb+'), protocol=4)

    # %%
    try:
        mol.industry_r_wacc_data = pickle.load(open('./data/modeled_data/bank_r_wacc_data_train.pkl', mode='rb+'))
    except:
        mol.industry_r_wacc_data = mol.clean_and_average_r_wacc_data()
        pickle.dump(obj=mol.industry_r_wacc_data, file=open(file='./data/modeled_data/bank_r_wacc_data_train.pkl', mode='wb+'), protocol=4)

    # %%
    try:
        mol.industry_factors_df = pd.read_csv('./data/modeled_data/bank_factors_df_train.csv', index_col='date', parse_dates=['date'])
        mol.factors_risks_data_standardized = pd.read_csv('./data/dict/factors_risks_dicts_standardized.csv', index_col='factor_code')
    except:
        mol.industry_factors_df = mol.transform_factors_datas_from_dict_to_df()
        mol.factors_risks_data_standardized = mol.standardize_factors_risks_data()
        mol.industry_factors_df.to_csv('./data/modeled_data/bank_factors_df_train.csv', encoding='utf-8')
        mol.factors_risks_data_standardized.to_csv('./data/dict/factors_risks_dicts_standardized.csv', encoding='utf-8', index=True, index_label='factor_code')

    # %%
    fig_type = 'risk_factors_clusters'
    filename = rf"./image/{fig_type}.png"

    try:
        # 尝试从文件加载图像
        fig = mpimg.imread(filename)
        print(f"Loaded figure {fig_type} from {filename}")

        # 显示加载的图像
        plt.figure(figsize=(20, 15), dpi=300)
        plt.imshow(fig)
    except FileNotFoundError:
        # 创建新图形并保存
        fig = mol.visualize_risk_factors_clusters(random_seed=52)
        # 确保保存格式和文件扩展名匹配
        fig.savefig(filename, dpi=300)
        print(f"Saved figure {fig_type} to {filename}")

        # 直接显示创建的图形
        plt.figure(fig.number)  # 激活这个图形窗口
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()  # 确保图形显示

    # %%
    try:
        mol.industry_risks_premiums_totals = pickle.load(open('./data/modeled_data/bank_risks_premiums_totals.pkl', mode='rb+'))
        mol.risks_premiums_totals_curs = pickle.load(open('./model/risks_premiums_totals_curs.pkl', mode='rb+'))
    except:
        mol.industry_risks_premiums_totals, mol.risks_premiums_totals_curs = mol.get_industry_risks_premiums_totals()
        pickle.dump(obj=mol.industry_risks_premiums_totals, file=open(file='./data/modeled_data/bank_risks_premiums_totals.pkl', mode='wb+'), protocol=4)
        pickle.dump(obj=mol.risks_premiums_totals_curs, file=open(file='./model/risks_premiums_totals_curs.pkl', mode='wb+'), protocol=4)

    # %%
    try:
        mol.industry_risks_premiums_components = pickle.load(open('./data/modeled_data/bank_risks_premiums_components.pkl', mode='rb+'))
        mol.risks_premiums_components_curs = pickle.load(open('./model/risks_premiums_components_curs.pkl', mode='rb+'))
    except:
        mol.industry_risks_premiums_components, mol.risks_premiums_components_curs = mol.get_industry_risks_premiums_components()
        pickle.dump(obj=mol.industry_risks_premiums_components, file=open(file='./data/modeled_data/bank_risks_premiums_components.pkl', mode='wb+'), protocol=4)
        pickle.dump(obj=mol.risks_premiums_components_curs, file=open(file='./model/risks_premiums_components_curs.pkl', mode='wb+'), protocol=4)

    # %%
    try:
        mol.causal_forest_forests = pickle.load(open('./model/bank_causal_forest_forests.pkl', mode='rb+'))
    except:
        mol.causal_forest_forests = mol.estimate_causal_effects_with_dml()
        pickle.dump(obj=mol.causal_forest_forests, file=open(file='./model/bank_causal_forest_forests.pkl', mode='wb+'), protocol=4)

    # %%
    fig_type = 'causal_forest_trees'
    filename = rf"./image/causal_effect/{fig_type}.png"

    try:
        # 尝试从文件加载图像
        fig = mpimg.imread(filename)

        # 显示加载的图像
        plt.figure(figsize=(20, 15), dpi=300)
        plt.imshow(fig)
        print(f"Loaded figure {fig_type} from {filename}")

    except FileNotFoundError:
        # 创建新图形并保存
        fig = mol.visualize_causal_forest_trees()
        # 确保保存格式和文件扩展名匹配
        fig.savefig(filename, dpi=300)

        # 直接显示创建的图形
        plt.figure(fig.number)  # 激活这个图形窗口
        print(f"Saved figure {fig_type} to {filename}")

    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()  # 确保图形显示

    # %%
    # Save all figures to files with descriptive names
    fig_types = [
        'cate_over_time',
        'cate_by_month',
        'ate_by_treatment',
        'ate_distribution',
        'treatment_vs_outcome',
        'causal_forest_feature_importance'
    ]
    n_types = len(fig_types)

    for treatment in ['default_risk', 'liquidity_risk', 'market_risk']:
        for outcome in ['common_risk', 'idiosyncratic_risk']:
            try:
                for fig_type in fig_types:
                    filename = rf"./image/causal_effect/{fig_type}_{treatment}_vs_{outcome}.png"
                    # 尝试从文件加载图像
                    fig = mpimg.imread(filename)
                    print(f"Loaded figure {fig_type} from {filename}")

                    # 显示加载的图像
                    plt.figure(figsize=(20, 15), dpi=300)
                    plt.imshow(fig)

                    plt.axis('off')
                    plt.tight_layout(pad=0)
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    plt.show()  # 确保图形显示
            except:
                figs = mol.visualize_causal_effect(treatment, outcome)
                for fig, fig_type in zip(figs, fig_types):
                    filename = rf"./image/causal_effect/{fig_type}_{treatment}_vs_{outcome}.png"

                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"Saved figure {fig} to {filename}")

                # 直接显示创建的图形
                plt.figure(fig.number)  # 激活这个图形窗口

                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.show()  # 确保图形显示




