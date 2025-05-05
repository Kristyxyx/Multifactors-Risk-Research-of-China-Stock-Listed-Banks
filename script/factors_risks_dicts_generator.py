# %%
import pandas as pd

def generate_factors_risks_dicts(
    factors_datas_dicts_path: str = r'../data/dict/factors_datas_dicts.csv',
    factors_risks_dicts_path: str = r'../data/dict/factors_risks_dicts.csv'
):
    """
    生成因子风险分类数据
    :param factors_datas_dicts_path: 因子数据字典路径
    :param factors_risks_dicts_path: 因子风险分类数据路径
    :return: None
    """
    factors_datas_dicts = pd.read_csv(factors_datas_dicts_path)

    # --- 关键词定义 ---
    # 定义与各类风险相关的关键词
    keywords = {
        'Market Risk': ['市场', '市值', '价格', '波动', '利率', '汇率', '收益率', '价值变动', '市盈率', '市净率', '市销率', '市现率', '指数', '振幅', 'beta', '贝塔', '股价'],
        'Liquidity Risk': ['流动', '现金', '周转', '支付', '变现', '偿付', '资金', '存贷', '拆借', '货币', '速动', '现钞', '存款', '换手率', '交易量', '成交量', '成交额', '流量', '融资', '融券'],
        'Default Risk': ['违约', '信用', '坏账', '不良', '杠杆', '负债', '偿债', '风险', '拨备', '资本充足', '贷款', '债务', '产权比率', '损失', '呆账', '借款', '金融负债']
    }

    # --- 风险分类处理 ---
    factors_risks_list = [] # 用于存储结果的列表

    # 遍历DataFrame的每一行
    for index, row in factors_datas_dicts.iterrows():
        factor_code = row['因子代码']
        factor_name = row['因子名称']
        # 获取'description'和'supplementary_description'列的内容，处理NaN值为空字符串
        details = str(row['细节']) if pd.notna(row['细节']) else ''
        supp_details = str(row['补充性细节']) if pd.notna(row['补充性细节']) else ''

        # 合并用于搜索的文本，包含因子名称、细节和补充细节
        search_text = f"{factor_name} {details} {supp_details}".lower() # 转换为小写以便不区分大小写匹配

        identified_risks = [] # 存储当前因子识别出的风险类型

        # 检查每种风险类型的关键词是否存在于搜索文本中
        for risk_type, risk_keywords in keywords.items():
            # 使用any()来检查是否有任何一个关键词存在于文本中
            if any(keyword.lower() in search_text for keyword in risk_keywords):
                identified_risks.append(risk_type)

        # 如果识别到了风险类型
        if identified_risks:
            # 对风险类型进行排序，确保输出顺序一致性
            risk_string = ', '.join(sorted(identified_risks))

            # 将结果添加到列表中
            factors_risks_list.append(
                {
                    'factor_name': factor_name,
                    'factor_code': factor_code,
                    'risk': risk_string
                }
            )
        # else:
            # 如果没有匹配到任何关键词，可以选择跳过该因子，或者标记为 'Unclassified'
            # 这里我们选择跳过，只输出有明确分类的因子
            # pass

    # --- 输出结果 ---
    # 将结果列表转换为DataFrame
    factors_risks_dicts = pd.DataFrame(factors_risks_list)

    # 将结果DataFrame保存为CSV文件
    # 使用 'utf-8-sig' 编码确保在Excel中正确显示中文
    factors_risks_dicts.to_csv(factors_risks_dicts_path, index=False, encoding='utf-8-sig')

# %%
if __name__ == '__main__':
    factors_risks_data_generator()


