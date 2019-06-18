import pandas as pd
import os

relationships = ["没有关系", "一对多", "多对一", "多对多"]


def get_all_path(data_path):
    '''
    获得目录下所有文件路径
    :param data_path:
    :return:
    '''
    datas = os.listdir(data_path)
    for i in range(len(datas)):
        datas[i] = os.path.join(data_path, datas[i])
    return datas


def get_all_df(paths):
    '''
    根据路径读取所有DataFrame
    :param paths:
    :return:
    '''
    df_arr = []
    for data in paths:
        df = pd.read_excel(data)
        print(df.columns)
        df_arr.append(df)
    return df_arr


def get_same_col(df1, df2):
    '''
    获得两个数据集的相同列名
    :param df1:
    :param df2:
    :return:
    '''
    same_cols = []
    col1 = df1.columns
    col2 = df2.columns
    for col in col1:
        if col in col2:
            same_cols.append(col)
    return same_cols


def get_relationship(df1, df2, same_cols):
    '''
    判断两个表之间的关系
    :param df1:
    :param df2:
    :param same_cols:
    :return:
    '''
    if len(same_cols) < 1:
        return "没有关系"
    left = "一"
    right = "一"
    for col in same_cols:
        df = df1[col]
        for i in df:
            # print(i)
            temp = df2[df2[col] == i]
            if len(temp) > 1:
                right = "多"
                break
        df = df2[col]
        for i in df:
            temp = df1[df1[col] == i]
            if len(temp) > 1:
                left = "多"
                break
    return left + "对" + right


def build_levels(df_arr, current, last, df_levels, level):
    '''
    给数据表分层
    :param df_arr:
    :param main:
    :return:
    '''
    if len(df_arr) <= current or check_df_in_dict(df_levels, current) or not len(
            get_same_col(df_arr[current], df_arr[last])):
        return
    if not level in df_levels.keys():
        df_levels[str(level)] = []
    df_levels[str(level)].append((current, df_arr[current]))
    for i in range(0, len(df_arr)):
        cols = get_same_col(df_arr[current], df_arr[i])
        if len(cols):
            build_levels(df_arr, i, current, df_levels, level + 1)


def build_levels_pro(df_arr, df_levels, main):
    '''
    给数据集分层改进版
    :param df_arr:所有表格
    :param df_levels:层次字典
    :param main:主表的索引
    :return:
    '''
    stack = []
    df_levels[str(main)] = []
    df_levels[str(main)].append((main, df_arr[main]))
    level = 1
    current = 1
    next_num = 0
    stack.append((main, df_arr[main]))
    while len(stack):
        temp = stack.pop()
        current -= 1
        for i in range(len(df_arr)):
            if i == temp[0] or check_df_in_dict(df_levels, i):
                continue
            col = get_same_col(temp[1], df_arr[i])
            if len(col):
                if not level in df_levels.keys():
                    df_levels[str(level)] = []
                df_levels[str(level)].append((i, df_arr[i]))
                stack.append((i, df_arr[i]))
                next_num += 1
        if current <= 0:
            current = next_num
            next_num = 0
            level += 1


def check_df_in_dict(df_levels, df_num):
    '''
    判断层次表中存在某个表
    :param df_levels:
    :param df_num:
    :return:
    '''
    values = df_levels.values()
    for value in values:
        for i in value:
            res = i[0] == df_num
            if res:
                return True
    return False


def merge_all_df(df_levels):
    '''
    按层次拼接各个表
    :param df_levels:
    :return:
    '''
    levels = len(df_levels)  # 层数
    for level in range(levels - 1):  # 从最后一层开始遍历
        dfs_high = df_levels[str(levels - 2 - level)]
        dfs_low = df_levels[str(levels - 1 - level)]
        for df_high in dfs_high:
            for df_low in dfs_low:
                col = get_same_col(df_high[1], df_low[1])
                if len(col):
                    print(col)


def add_table_by_cols(df1, df2, cols):
    '''
    把df2中非col列根据col值添加到相应的df1中
    :param df1:
    :param df2:
    :param cols: 相同的列
    :return:
    '''
    if not len(cols):
        return
    col = cols[0]
    same_col_value = df1[col]  # df1中col列的数值

    df2_colmuns = list(df2.columns)
    for i in cols:
        df2_colmuns.remove(i)  # 获得df2中不同于df1的列
    for i in df2_colmuns:  # 在df1中添加df2的列并赋值0
        df1[i] = 0
    for j in same_col_value:  # 遍历df1中col的值
        df_merged = df2[df2[col] == j]  # 找到df2中col值为j的行
        for i in df2_colmuns:  # 给df1中相应位置赋值
            df1.loc[df1[col] == j, i] = df_merged[i].sum()
    print(df1)


data_path = "dataset"
if __name__ == '__main__':
    datas = get_all_path(data_path)
    df_arr = get_all_df(datas)
    # df_levels = {}
    # build_levels_pro(df_arr, df_levels, 0)
    # merge_all_df(df_levels)

    # print(df_levels)
    # same_cols = get_same_col(df_arr[1], df_arr[3])
    # print(same_cols)
    # relationship = get_relationship(df_arr[1], df_arr[3], same_cols)
    # print(relationship)

    add_table_by_cols(df_arr[3], df_arr[2], ["ProductID"])
