# Global Weekly PSI planning and simulation

# written by Yasushi Ohsugi with chatGPT
# as of 2042/05/02

# license : MIT license

# start of code


import pandas as pd
import csv

import math
import numpy as np

import datetime
import calendar


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly.offline as offline
import plotly.io as pio

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.express as px


from copy import deepcopy

import itertools

import re

import copy


# for images directory
import os

# if not os.path.exists("temp_images"):
#    os.mkdir("temp_images")


# 幅優先探索 (Breadth-First Search)
from collections import deque


# 可視化トライアル
# node dictの在庫Iを可視化
def show_node_I4bullwhip_color(node_I4bullwhip):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # x, y, z軸のデータを作成
    x = np.arange(len(node_I4bullwhip["HAM_N"]))

    n = len(node_I4bullwhip.keys())
    y = np.arange(n)

    X, Y = np.meshgrid(x, y)

    z = list(node_I4bullwhip.keys())

    Z = np.zeros((n, len(x)))

    # node_I4bullwhipのデータをZに格納
    for i, node_name in enumerate(z):
        Z[i, :] = node_I4bullwhip[node_name]

    # 3次元の棒グラフを描画
    dx = dy = 1.2  # 0.8
    dz = Z
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    for i in range(n):
        ax.bar3d(
            X[i],
            Y[i],
            np.zeros_like(dz[i]),
            dx,
            dy,
            dz[i],
            color=colors[i % len(colors)],
            alpha=0.8,
        )

    # 軸ラベルを設定
    ax.set_xlabel("Week")
    ax.set_ylabel("Node")
    ax.set_zlabel("Inventory")

    # y軸の目盛りをnode名に設定
    ax.set_yticks(y)
    ax.set_yticklabels(z)

    plt.show()


def show_psi_3D_graph_node(node):

    node_name = node.name

    # node_name = psi_list[0][0][0][:-7]
    # node_name = psiS2P[0][0][0][:-7]

    psi_list = node.psi4demand

    # 二次元マトリクスのサイズを定義する
    x_size = len(psi_list)
    y_size = len(psi_list[0])

    # x_size = len(psiS2P)
    # y_size = len(psiS2P[0])

    # x軸とy軸のグリッドを生成する
    x, y = np.meshgrid(range(x_size), range(y_size))

    # y軸の値に応じたカラーマップを作成
    color_map = plt.cm.get_cmap("cool")

    # z軸の値をリストから取得する
    z = []

    for i in range(x_size):
        row = []
        for j in range(y_size):

            row.append(len(psi_list[i][j]))
            # row.append(len(psiS2P[i][j]))

        z.append(row)

    ravel_z = np.ravel(z)

    norm = plt.Normalize(0, 3)
    # norm = plt.Normalize(0,dz.max())

    # 3Dグラフを作成する
    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")

    z_like = np.zeros_like(z)

    # ********************
    # x/yの逆転
    # ********************
    original_matrix = z

    inverted_matrix = []

    for i in range(len(original_matrix[0])):
        inverted_row = []
        for row in original_matrix:
            inverted_row.append(row[i])
        inverted_matrix.append(inverted_row)

    z_inv = inverted_matrix

    # colors = plt.cm.terrain_r(norm(z_inv))
    # colors = plt.cm.terrain_r(norm(dz))

    # ********************
    # 4色での色分け
    # ********************

    # 色分け用のデータ
    color_data = [1, 2, 3, 4]

    # 色は固定
    # colorsのリストは、S/CO/I/Pに対応する
    # colors = ['cyan', 'blue', 'red', 'gold']
    # colors = ['cyan', 'blue', 'maroon', 'gold']
    colors = ["cyan", "blue", "brown", "gold"]

    y_list = np.ravel(y)

    c_map = []

    for index in y_list:

        c_map.append(colors[index])

    # ********************
    # bar3D
    # ********************

    ax.bar3d(
        np.ravel(x),
        np.ravel(y),
        np.ravel(np.zeros_like(z)),
        0.05,
        0.05,
        np.ravel(z_inv),
        color=c_map,
    )

    ax.set_title(node_name, fontsize="16")  # タイトル

    plt.show()


def visualise_psi_label(node_I_psi, node_name):

    # データの定義
    x, y, z = [], [], []

    for i in range(len(node_I_psi)):

        for j in range(len(node_I_psi[i])):

            # node_idx = node_name.index('JPN')

            node_label = node_name[i]  # 修正

            for k in range(len(node_I_psi[i][j])):
                x.append(j)
                y.append(node_label)
                z.append(k)

    text = []

    for i in range(len(node_I_psi)):

        for j in range(len(node_I_psi[i])):

            for k in range(len(node_I_psi[i][j])):

                text.append(node_I_psi[i][j][k])

    # y軸のラベルを設定
    y_axis = dict(tickvals=node_name, ticktext=node_name)

    # 3D散布図の作成
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                text=text,
                marker=dict(size=5, color=z, colorscale="Viridis", opacity=0.8),
            )
        ]
    )

    # レイアウトの設定
    fig.update_layout(
        title="Node Connections",
        scene=dict(xaxis_title="Week", yaxis_title="Location", zaxis_title="Lot ID"),
        width=800,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90),
    )

    # グラフの表示
    # fig.show()
    return fig


# visualise I 3d bar
def visualise_inventory4demand_3d_bar(root_node, out_filename):

    nodes_list = []
    node_psI_list = []

    nodes_list, node_psI_list = extract_nodes_psI4demand(root_node)

    # visualise with 3D bar graph
    fig = visualise_psi_label(node_psI_list, nodes_list)

    offline.plot(fig, filename=out_filename)
    # offline.plot(fig, filename = out_filename)


def visualise_inventory4supply_3d_bar(root_node, out_filename):
    nodes_list = []
    node_psI_list = []
    plan_range = root_node.plan_range

    nodes_list, node_psI_list = extract_nodes_psI4supply(root_node, plan_range)

    # visualise with 3D bar graph
    fig = visualise_psi_label(node_psI_list, nodes_list)

    offline.plot(fig, filename=out_filename)


def visualise_I_bullwhip4supply(root_node, out_filename):

    plan_range = root_node.plan_range

    # node_all_psiからIを抽出してnode_psI_list生成してvisualise
    node_all_psi = {}

    node_all_psi = get_all_psi4supply(root_node, node_all_psi)

    # X
    week_len = 53 * plan_range + 1
    # week_len = len(node_yyyyww_lotid[0]) # node数が入る所を[0]で数える・・・

    # Y
    nodes_list = list(node_all_psi.keys())
    node_len = len(nodes_list)

    node_psI_list = [[] * i for i in range(node_len)]

    # make bullwhip data    I_lot_step_week_node

    # lot_stepの「値=長さ」の入れ物 x軸=week y軸=node
    #
    # week_len = len(node_yyyyww_lotid)ではなく 53 * plan_range でmaxに広げておく
    I_lot_step_week_node = [[None] * week_len for _ in range(node_len)]

    for node_name, psi_list in node_all_psi.items():

        node_index = nodes_list.index(node_name)

        supply_inventory_list = [[] * i for i in range(53 * plan_range)]
        # supply_inventory_list = [[]*i for i in range(len(psi_list))]

        for week in range(53 * plan_range):
            # for week in range(len(psi_list)):

            step_lots = psi_list[week][2]

            week_pos = week
            node_pos = nodes_list.index(node_name)

            I_lot_step_week_node[node_pos][week_pos] = len(step_lots)

            supply_inventory_list[week] = step_lots

        node_psI_list[node_index] = supply_inventory_list

    # bullwhip visualise
    I_visual_df = pd.DataFrame(I_lot_step_week_node, index=nodes_list)

    data = [go.Bar(x=I_visual_df.index, y=I_visual_df[0])]

    layout = go.Layout(
        title="Inventory Bullwip animation Global Supply Chain",
        xaxis={"title": "Location node"},
        yaxis={"title": "Lot-ID count", "showgrid": False},
        font={"size": 10},
        width=800,
        height=400,
        showlegend=False,
    )

    frames = []
    for week in I_visual_df.columns:
        frame_data = [go.Bar(x=I_visual_df.index, y=I_visual_df[week])]
        frame_layout = go.Layout(
            annotations=[
                go.layout.Annotation(
                    x=0.95,
                    y=1,
                    xref="paper",
                    yref="paper",
                    text=f"Week number: {week}",
                    showarrow=False,
                    font={"size": 14},
                )
            ]
        )
        frame = go.Frame(data=frame_data, layout=frame_layout)
        frames.append(frame)

    fig = go.Figure(data=data, layout=layout, frames=frames)

    offline.plot(fig, filename=out_filename)


def visualise_I_bullwhip4demand(root_node, out_filename):

    plan_range = root_node.plan_range

    # **********************************
    # make bullwhip data    I_lot_step_week_node
    # **********************************

    # *********************************
    # node_all_psiからIを抽出してnode_psI_list生成してvisualise
    # *********************************
    node_all_psi = {}

    node_all_psi = get_all_psi4demand(root_node, node_all_psi)

    # X
    week_len = 53 * plan_range + 1
    # week_len = len(node_yyyyww_lotid[0]) # node数が入る所を[0]で数える・・・

    # Y
    nodes_list = list(node_all_psi.keys())
    node_len = len(nodes_list)

    # make bullwhip data    I_lot_step_week_node
    # lot_stepの「値=長さ」の入れ物 x軸=week y軸=node

    I_lot_step_week_node = [[None] * week_len for _ in range(node_len)]

    #@240526
    node_psI_list = [[] * i for i in range(node_len)]


    for node_name, psi_list in node_all_psi.items():
        node_index = nodes_list.index(node_name)

        supply_inventory_list = [[] * i for i in range(53 * plan_range)]
        # supply_inventory_list = [[]*i for i in range(len(psi_list))]

        for week in range(53 * plan_range):

            step_lots = psi_list[week][2]

            week_pos = week
            node_pos = nodes_list.index(node_name)

            I_lot_step_week_node[node_pos][week_pos] = len(step_lots)

            supply_inventory_list[week] = step_lots

        node_psI_list[node_index] = supply_inventory_list

    # ********************************
    # bullwhip visualise
    # ********************************
    I_visual_df = pd.DataFrame(I_lot_step_week_node, index=nodes_list)

    data = [go.Bar(x=I_visual_df.index, y=I_visual_df[0])]

    layout = go.Layout(
        title="Inventory Bullwip animation Global Supply Chain",
        xaxis={"title": "Location node"},
        yaxis={"title": "Lot-ID count", "showgrid": False},
        font={"size": 10},
        width=800,
        height=400,
        showlegend=False,
    )

    frames = []
    for week in I_visual_df.columns:
        frame_data = [go.Bar(x=I_visual_df.index, y=I_visual_df[week])]
        frame_layout = go.Layout(
            annotations=[
                go.layout.Annotation(
                    x=0.95,
                    y=1,
                    xref="paper",
                    yref="paper",
                    text=f"Week number: {week}",
                    showarrow=False,
                    font={"size": 14},
                )
            ]
        )
        frame = go.Frame(data=frame_data, layout=frame_layout)
        frames.append(frame)

    fig = go.Figure(data=data, layout=layout, frames=frames)

    offline.plot(fig, filename=out_filename)


# sub modules definition
def extract_nodes_psI4demand(root_node):

    plan_range = root_node.plan_range

    # *********************************
    # node_all_psiからIを抽出してnode_psI_list生成してvisualise
    # *********************************
    node_all_psi = {}  # node:psi辞書に抽出

    node_all_psi = get_all_psi4demand(root_node, node_all_psi)
    # node_all_psi = get_all_psi4demand(root_node_outbound, node_all_psi)

    # get_all_psi4supply(root_node_outbound)

    # X
    week_len = 53 * plan_range + 1

    # Y
    nodes_list = list(node_all_psi.keys())

    node_len = len(nodes_list)

    node_psI_list = [[] * i for i in range(node_len)]

    for node_name, psi_list in node_all_psi.items():

        node_index = nodes_list.index(node_name)

        # supply_inventory_list = [[]*i for i in range( 53 * plan_range )]
        supply_inventory_list = [[] * i for i in range(len(psi_list))]

        for week in range(len(psi_list)):

            step_lots = psi_list[week][2]

            supply_inventory_list[week] = step_lots

        node_psI_list[node_index] = supply_inventory_list

    return nodes_list, node_psI_list


def extract_nodes_psI4demand_postorder(root_node):

    plan_range = root_node.plan_range

    # *********************************
    # node_all_psiからIを抽出してnode_psI_list生成してvisualise
    # *********************************
    node_all_psi = {}  # node:psi辞書に抽出

    node_all_psi = get_all_psi4demand_postorder(root_node, node_all_psi)
    # node_all_psi = get_all_psi4demand(root_node_outbound, node_all_psi)

    # get_all_psi4supply(root_node_outbound)

    week_len = 53 * plan_range + 1

    # Y
    nodes_list = list(node_all_psi.keys())

    node_len = len(nodes_list)

    node_psI_list = [[] * i for i in range(node_len)]

    for node_name, psi_list in node_all_psi.items():

        node_index = nodes_list.index(node_name)

        # supply_inventory_list = [[]*i for i in range( 53 * plan_range )]
        supply_inventory_list = [[] * i for i in range(len(psi_list))]

        for week in range(len(psi_list)):

            step_lots = psi_list[week][2]

            supply_inventory_list[week] = step_lots

        node_psI_list[node_index] = supply_inventory_list

    return nodes_list, node_psI_list


def extract_nodes_psI4supply(root_node, plan_range):
    # *********************************
    # node_all_psiからIを抽出してnode_psI_list生成してvisualise
    # *********************************
    node_all_psi = {}  # node:psi辞書に抽出

    node_all_psi = get_all_psi4supply(root_node, node_all_psi)

    # X
    week_len = 53 * plan_range + 1

    # Y
    nodes_list = list(node_all_psi.keys())

    node_len = len(nodes_list)

    node_psI_list = [[] * i for i in range(node_len)]

    for node_name, psi_list in node_all_psi.items():

        node_index = nodes_list.index(node_name)

        supply_inventory_list = [[] * i for i in range(len(psi_list))]

        for week in range(len(psi_list)):

            step_lots = psi_list[week][2]

            supply_inventory_list[week] = step_lots

        node_psI_list[node_index] = supply_inventory_list

    return nodes_list, node_psI_list


# 前処理として、年月の月間販売数の一日当たりの平均値を計算する
def calc_average_sales(monthly_sales, year):

    month_daily_average = [0] * 12

    for i, month_qty in enumerate(monthly_sales):

        month = i + 1

        days_in_month = calendar.monthrange(year, month)[1]

        month_daily_average[i] = monthly_sales[i] / days_in_month

    return month_daily_average


# ある年の月次販売数量を年月から年ISO週に変換する
def calc_weekly_sales(
    node,
    monthly_sales,
    year,
    year_month_daily_average,
    sales_by_iso_year,
    yyyyww_value,
    yyyyww_key,
):

    weekly_sales = [0] * 53

    for i, month_qty in enumerate(monthly_sales):

        # 開始月とリストの要素番号を整合
        month = i + 1

        # 月の日数を調べる
        days_in_month = calendar.monthrange(year, month)[1]

        # 月次販売の日平均
        avg_daily_sales = year_month_daily_average[year][i]  # i=month-1

        # 月の日毎の処理
        for day in range(1, days_in_month + 1):
            # その年の"年月日"を発生

            ## iso_week_noの確認 年月日でcheck その日がiso weekで第何週か
            # iso_week = datetime.date(year,month, day).isocalendar()[1]

            # ****************************
            # year month dayからiso_year, iso_weekに変換
            # ****************************
            dt = datetime.date(year, month, day)

            iso_year, iso_week, _ = dt.isocalendar()

            # 辞書に入れる場合
            sales_by_iso_year[iso_year][iso_week - 1] += avg_daily_sales

            # リストに入れる場合
            node_year_week_str = f"{node}{iso_year}{iso_week:02d}"

            if node_year_week_str not in yyyyww_key:

                yyyyww_key.append(node_year_week_str)

            pos = len(yyyyww_key) - 1

            yyyyww_value[pos] += avg_daily_sales

    return sales_by_iso_year[year]


# *******************************************************
# trans S from monthly to weekly
# *******************************************************
# 処理内容
# 入力ファイル: 拠点node別サプライチェーン需給tree
#               複数年別、1月-12月の需要数
#

# 処理        : iso_year+iso_weekをkeyにして、需要数を月間から週間に変換する

#               前処理で、各月の日数と月間販売数から、月毎の日平均値を求める
#               年月日からISO weekを判定し、
#               月間販売数の日平均値をISO weekの変数に加算、週間販売数を計算

#               ***** pointは「年月日からiso_year+iso_weekへの変換処理」 *****
#               dt = datetime.date(year, month, day)
#               iso_year, iso_week, _ = dt.isocalendar()

#               for nodeのループ下で、
#               YM_key_list.append(key)  ## keyをappendして
#               pos = len( YW_key_list ) ## YM_key_listの長さを位置にして
#               YW_value_list( pos ) += average_daily_value ## 値を+=加算

# 出力リスト  : node別 複数年のweekの需要 S_week


def trans_month2week(input_file, outputfile):

    # read monthly S
    # csvファイルの読み込み
    df = pd.read_csv(input_file)  # IN:      'S_month_data.csv'

    #    # mother plant capacity parameter
    #    demand_supply_ratio = 1.2  # demand_supply_ratio = ttl_supply / ttl_demand

    # initial setting of total demand and supply
    # total_demandは、各行のm1からm12までの列の合計値

    df_capa = pd.read_csv(input_file)

    df_capa["total_demand"] = df_capa.iloc[:, 3:].sum(axis=1)

    # yearでグループ化して、月次需要数の総和を計算
    df_capa_year = df_capa.groupby(["year"], as_index=False).sum()

    # リストに変換
    month_data_list = df.values.tolist()

    # node_nameをユニークなキーとしたリストを作成する
    node_list = df["node_name"].unique().tolist()

    # *********************************
    # write csv file header [prod-A,node_name,year.w0,w1,w2,w3,,,w51,w52,w53]
    # *********************************

    file_name_out = outputfile  # OUT:     'S_iso_week_data.csv'

    with open(file_name_out, mode="w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "product_name",
                "node_name",
                "year",
                "w1",
                "w2",
                "w3",
                "w4",
                "w5",
                "w6",
                "w7",
                "w8",
                "w9",
                "w10",
                "w11",
                "w12",
                "w13",
                "w14",
                "w15",
                "w16",
                "w17",
                "w18",
                "w19",
                "w20",
                "w21",
                "w22",
                "w23",
                "w24",
                "w25",
                "w26",
                "w27",
                "w28",
                "w29",
                "w30",
                "w31",
                "w32",
                "w33",
                "w34",
                "w35",
                "w36",
                "w37",
                "w38",
                "w39",
                "w40",
                "w41",
                "w42",
                "w43",
                "w44",
                "w45",
                "w46",
                "w47",
                "w48",
                "w49",
                "w50",
                "w51",
                "w52",
                "w53",
            ]
        )

    # *********************************
    # plan initial setting
    # *********************************

    # node別に、中期計画の3ヵ年、5ヵ年をiso_year+iso_week連番で並べたもの
    # node_lined_iso_week = { node-A+year+week: [iso_year+iso_week1,2,3,,,,,],   }
    # 例えば、2024W00, 2024W01, 2024W02,,, ,,,2028W51,2028W52,2028W53という5年間分

    node_lined_iso_week = {}

    node_yyyyww_value = []
    node_yyyyww_key = []

    for node in node_list:

        df_node = df[df["node_name"] == node]

        # リストに変換
        node_data_list = df_node.values.tolist()

        #
        # getting start_year and end_year
        #
        start_year = node_data_min = df_node["year"].min()
        end_year = node_data_max = df_node["year"].max()

        # S_month辞書の初期セット
        monthly_sales_data = {}

        # *********************************
        # plan initial setting
        # *********************************

        plan_year_st = start_year  # 2024  # plan開始年

        plan_range = end_year - start_year + 1  # 5     # 5ヵ年計画分のS計画

        plan_year_end = plan_year_st + plan_range

        #
        # an image of data "df_node"
        #
        # product_name	node_name	year	m1	m2	m3	m4	m5	m6	m7	m8	m9	m10	m11	m12
        # prod-A	CAN	2024	0	0	0	0	0	0	0	0	0	0	0	0
        # prod-A	CAN	2025	0	0	0	0	0	0	0	0	0	0	0	0
        # prod-A	CAN	2026	0	0	0	0	0	0	0	0	0	0	0	0
        # prod-A	CAN	2027	0	0	0	0	0	0	0	0	0	0	0	0
        # prod-A	CAN	2028	0	0	0	0	0	0	0	0	0	0	0	0
        # prod-A	CAN_D	2024	122	146	183	158	171	195	219	243	231	207	195	219
        # prod-A	CAN_D	2025	122	146	183	158	171	195	219	243	231	207	195	219

        # *********************************
        # by node    node_yyyyww = [ node-a, yyyy01, yyyy02,,,, ]
        # *********************************

        yyyyww_value = [0] * 53 * plan_range  # 5ヵ年plan_range=5

        yyyyww_key = []

        for data in node_data_list:

            # node別　3年～5年　月次需要予測値

            # 辞書形式{year: S_week_list, }でデータ定義する
            sales_by_iso_year = {}

            # 前後年付きの辞書 53週を初期セット
            # 空リストの初期設定
            # start and end setting from S_month data # 月次Sのデータからmin&max
            # 前年の52週が発生する可能性あり # 計画の前後の-1年 +1年を見る
            work_year = plan_year_st - 1

            for i in range(plan_range + 2):  # 計画の前後の-1年 +1年を見る

                year_sales = [0] * 53  # 53週分の要素を初期セット

                # 年の辞書に週次Sをセット
                sales_by_iso_year[work_year] = year_sales

                work_year += 1

            # *****************************************
            # initial setting end
            # *****************************************

            # *****************************************
            # start process
            # *****************************************

            # ********************************
            # generate weekly S from monthly S
            # ********************************

            # S_monthのcsv fileを読んでS_month_listを生成する
            # pandasでcsvからリストにして、node_nameをキーに順にM2W変換

            # ****************** year ****** Smonth_list ******
            monthly_sales_data[data[2]] = data[3:]

            # data[0] = prod-A
            # data[1] = node_name
            # data[2] = year

        # **************************************
        # 年月毎の販売数量の日平均を計算する
        # **************************************
        year_month_daily_average = {}

        for y in range(plan_year_st, plan_year_end):

            year_month_daily_average[y] = calc_average_sales(monthly_sales_data[y], y)

        # 販売数量を年月から年ISO週に変換する
        for y in range(plan_year_st, plan_year_end):

            sales_by_iso_year[y] = calc_weekly_sales(
                node,
                monthly_sales_data[y],
                y,
                year_month_daily_average,
                sales_by_iso_year,
                yyyyww_value,
                yyyyww_key,
            )

        work_yyyyww_value = [node] + yyyyww_value
        work_yyyyww_key = [node] + yyyyww_key

        node_yyyyww_value.append(work_yyyyww_value)
        node_yyyyww_key.append(work_yyyyww_key)

        # 複数年のiso週毎の販売数を出力する
        for y in range(plan_year_st, plan_year_end):

            rowX = ["product-X"] + [node] + [y] + sales_by_iso_year[y]

            with open(file_name_out, mode="a", newline="") as f:

                writer = csv.writer(f)

                writer.writerow(rowX)

    # **********************
    # リスト形式のS出力
    # **********************

    return node_yyyyww_value, node_yyyyww_key, plan_range, df_capa_year


# *********************
# END of week data generation
# node_yyyyww_value と node_yyyyww_keyに複数年の週次データがある
# *********************


# *******************************************************
# lot by lot PSI
# *******************************************************
def makeS(S_week, lot_size):  # Sの値をlot単位に変換してリスト化

    return [math.ceil(num / lot_size) for num in S_week]


# @230908 mark このlotid生成とセットするpsi_listを、直接tree上のpsiのS[0]に置く
# @230908 mark または、S_stackedというリストで返す


def make_lotid_stack(S_stack, node_name, Slot, node_yyyyww_list):

    for w, (lots_count, node_yyyyww) in enumerate(zip(Slot, node_yyyyww_list)):

        stack_list = []

        for i in range(lots_count):

            lot_id = str(node_yyyyww) + str(i)

            stack_list.append(lot_id)

        ## week 0="S"
        # psi_list[w][0] = stack_list

        S_stack[w] = stack_list

    return S_stack


def make_lotid_list_setS(psi_list, node_name, Slot, yyyyww_list):

    for w, (lots_count, yyyyww) in enumerate(zip(Slot, yyyyww_list)):

        stack_list = []

        for i in range(lots_count):

            lot_id = str(yyyyww) + str(i)

            stack_list.append(lot_id)

        psi_list[w][0] = stack_list

    return psi_list


# ************************************
# checking constraint to inactive week , that is "Long Vacation"
# ************************************
def check_lv_week_bw(const_lst, check_week):

    num = check_week

    if const_lst == []:

        pass

    else:

        while num in const_lst:

            num -= 1

    return num


def check_lv_week_fw(const_lst, check_week):

    num = check_week

    if const_lst == []:

        pass

    else:

        while num in const_lst:

            num += 1

    return num


def calcPS2I4demand(psiS2P):

    plan_len = len(psiS2P)

    for w in range(1, plan_len):  # starting_I = 0 = w-1 / ending_I = 53
        # for w in range(1,54): # starting_I = 0 = w-1 / ending_I = 53

        s = psiS2P[w][0]
        co = psiS2P[w][1]

        i0 = psiS2P[w - 1][2]
        i1 = psiS2P[w][2]

        p = psiS2P[w][3]

        # *********************
        # # I(n-1)+P(n)-S(n)
        # *********************

        work = i0 + p

        # memo ここで、期末の在庫、S出荷=売上を操作している
        # S出荷=売上を明示的にlogにして、price*qty=rev売上として記録し表示処理
        # 出荷されたS=売上、在庫I、未出荷COの集合を正しく表現する

        # **************************
        # モノがお金に代わる瞬間
        # **************************

        diff_list = [x for x in work if x not in s]  # I(n-1)+P(n)-S(n)

        psiS2P[w][2] = i1 = diff_list

    return psiS2P


def shiftS2P_LV(psiS, safety_stock_week, lv_week):  # LV:long vacations

    ss = safety_stock_week

    plan_len = len(psiS) - 1  # -1 for week list position

    for w in range(plan_len, ss, -1):  # backward planningで需要を降順でシフト

        # my_list = [1, 2, 3, 4, 5]
        # for i in range(2, len(my_list)):
        #    my_list[i] = my_list[i-1] + my_list[i-2]

        # 0:S
        # 1:CO
        # 2:I
        # 3:P

        eta_plan = w - ss  # ss:safty stock

        eta_shift = check_lv_week_bw(lv_week, eta_plan)  # ETA:Eatimate Time Arrival

        # リスト追加 extend
        # 安全在庫とカレンダ制約を考慮した着荷予定週Pに、w週Sからoffsetする

        psiS[eta_shift][3].extend(psiS[w][0])  # P made by shifting S with

    return psiS


def shiftS2P_LV_replace(psiS, safety_stock_week, lv_week):  # LV:long vacations

    ss = safety_stock_week

    plan_len = len(psiS) - 1  # -1 for week list position

    for w in range(plan_len):  # foreward planningでsupplyのp [w][3]を初期化

        # psiS[w][0] = [] # S active

        psiS[w][1] = []  # CO
        psiS[w][2] = []  # I
        psiS[w][3] = []  # P

    for w in range(plan_len, ss, -1):  # backward planningでsupplyを降順でシフト

        # my_list = [1, 2, 3, 4, 5]
        # for i in range(2, len(my_list)):
        #    my_list[i] = my_list[i-1] + my_list[i-2]

        # 0:S
        # 1:CO
        # 2:I
        # 3:P

        eta_plan = w - ss  # ss:safty stock

        eta_shift = check_lv_week_bw(lv_week, eta_plan)  # ETA:Eatimate Time Arrival

        # リスト追加 extend
        # 安全在庫とカレンダ制約を考慮した着荷予定週Pに、w週Sからoffsetする
        psiS[eta_shift][3].extend(psiS[w][0])  # P made by shifting S with

    return psiS


def shiftP2S_LV(psiP, safety_stock_week, lv_week):  # LV:long vacations

    ss = safety_stock_week

    plan_len = len(psiP) - 1  # -1 for week list position

    for w in range(plan_len - 1):  # forward planningで確定Pを確定Sにシフト

        # my_list = [1, 2, 3, 4, 5]
        # for i in range(2, len(my_list)):
        #    my_list[i] = my_list[i-1] + my_list[i-2]

        # 0:S
        # 1:CO
        # 2:I
        # 3:P

        etd_plan = w + ss  # ss:safty stock

        etd_shift = check_lv_week_fw(lv_week, etd_plan)  # ETD:Eatimate TimeDep
        # リスト追加 extend
        # 安全在庫とカレンダ制約を考慮した着荷予定週Pに、w週Sからoffsetする

        psiP[etd_shift][0] = psiP[w][3]  # S made by shifting P with

    return psiP


def make_S_lots(node_yyyyww_value, node_yyyyww_key, nodes):

    S_lots_dict = {}

    for i, node_val in enumerate(node_yyyyww_value):  # by nodeでrepeat処理

        node_name = node_val[0]
        S_week = node_val[1:]

        node = nodes[node_name]  # node_nameからnodeインスタンスを取得

        # node.lot_sizeを使う
        lot_size = node.lot_size  # Node()からセット

        # makeSでSlotを生成
        # ロット数に変換し、週リストで返す # lotidではない
        # return [math.ceil(num / lot_size) for num in S_week]

        # Slot = makeS(S_week, lot_size)
        Slot = [math.ceil(num / lot_size) for num in S_week]

        ## nodeに対応するpsi_list[w][0,1,2,3]を生成する
        # psi_list = [[[] for j in range(4)] for w in range( len(S_week) )]

        S_stack = [[] for w in range(len(S_week))]

        node_key = node_yyyyww_key[i]  # node_name + yyyyww

        ####node_name = node_key[0] # node_valと同じ

        yyyyww_list = node_key[1:]

        # lotidをリスト化 #  Slotの要素「ロット数」からlotidを付番してリスト化
        S_lots_dict[node.name] = make_lotid_stack(S_stack, node_name, Slot, yyyyww_list)

    return S_lots_dict


def make_node_psi_dict(node_yyyyww_value, node_yyyyww_key, nodes):

    node_psi_dict = {}  # node_psi辞書

    for i, node_val in enumerate(node_yyyyww_value):  # by nodeでrepeat処理

        node_name = node_val[0]
        S_week = node_val[1:]

        node = nodes[node_name]  # node_nameからnodeインスタンスを取得

        # node.lot_sizeを使う
        lot_size = node.lot_size  # Node()からセット

        # makeSでSlotを生成
        # ロット数に変換し、週リストで返す # lotidではない
        # return [math.ceil(num / lot_size) for num in S_week]

        Slot = makeS(S_week, lot_size)

        # nodeに対応するpsi_list[w][0,1,2,3]を生成する
        psi_list = [[[] for j in range(4)] for w in range(len(S_week))]

        node_key = node_yyyyww_key[i]  # node_name + yyyyww

        yyyyww_list = node_key[1:]

        # lotidをリスト化 #  Slotの要素「ロット数」からlotidを付番してリスト化
        psiS = make_lotid_list_setS(psi_list, node_name, Slot, yyyyww_list)

        node_psi_dict[node_name] = psiS  # 初期セットSを渡す。本来はleaf_nodeのみ

    return node_psi_dict


# ***************************************
# mother plant/self.nodeの確定Sから子nodeを分離
# ***************************************
def extract_node_conf(req_plan_node, S_confirmed_plan):

    node_list = list(itertools.chain.from_iterable(req_plan_node))

    extracted_list = []
    extracted_list.extend(S_confirmed_plan)

    # フラットなリストに展開する
    flattened_list = [item for sublist in extracted_list for item in sublist]

    # node_listとextracted_listを比較して要素の追加と削除を行う
    extracted_list = [
        [item for item in sublist if item in node_list] for sublist in extracted_list
    ]

    return extracted_list


def separated_node_plan(node_req_plans, S_confirmed_plan):

    shipping_plans = []

    for req_plan in node_req_plans:

        shipping_plan = extract_node_conf(req_plan, S_confirmed_plan)

        shipping_plans.append(shipping_plan)

    return shipping_plans


# *****************************
# @240422 memo EVAL
# ****************************
def eval_supply_chain(node):
    # preordering

    #    if node.children == []:
    #
    #        pass
    #
    #    else:
    #
    #        # counting Purchase Order
    #        # psi_listのPOは、psi_list[w][2]の中のlot_idのロット数=リスト長
    #        node.set_lot_counts()
    #
    #
    #        #@240423 EvalPlanSIP()の中でnode instanceに以下をセットする
    #        # self.profit, self.revenue, self.profit_ratio

    # @stop240508 二回evalしているので・・・
    # node.EvalPlanSIP()

    # print("Eval node profit revenue profit_ratio", node.name, node.profit, node.revenue, node.profit_ratio)

    # counting Purchase Order
    # psi_listのPOは、psi_list[w][2]の中のlot_idのロット数=リスト長
    node.set_lot_counts()

    # @240423 EvalPlanSIP()の中でnode instanceに以下をセットする
    # self.profit, self.revenue, self.profit_ratio
    node.EvalPlanSIP()

    # @240508
    print(
        "Eval node profit revenue profit_ratio",
        node.name,
        node.eval_profit,
        node.eval_revenue,
        node.eval_profit_ratio,
    )

    for child in node.children:

        eval_supply_chain(child)


# **********************************
# create tree
# **********************************
class Node:  # with "parent"

    def __init__(self, name):

        print("class Node init name", name)

        self.name = name
        self.children = []
        self.parent = None

        # application attribute # nodeをインスタンスした後、初期値セット
        self.psi4demand = None
        self.psi4supply = None

        self.psi4couple = None

        self.psi4accume = None

        self.plan_range = 1
        self.plan_year_st = 2020

        self.safety_stock_week = 0
        # self.safety_stock_week = 2

        # self.lv_week = []

        self.lot_size = 1  # defalt set

        # leadtimeとsafety_stock_weekは、ここでは同じ
        self.leadtime = 1  # defalt set  # 前提:SS=0

        self.long_vacation_weeks = []

        # evaluation
        self.decoupling_total_I = []  # total Inventory all over the plan

        # position
        self.longtitude = None
        self.latitude = None

        # **************************
        # BU_SC_node_profile     business_unit_supplychain_node
        # **************************

        # @240421 機械学習のフラグはstop
        ## **************************
        ## plan_basic_parameter ***sequencing is TEMPORARY
        ## **************************
        #        self.PlanningYear           = row['plan_year']
        #        self.plan_engine            = row['plan_engine']
        #        self.reward_sw              = row['reward_sw']

        # ***************************
        # business unit identify
        # ***************************

        # @240421 多段階PSIのフラグはstop
        #        self.product_name           = row['product_name']
        #        self.SC_tree_id             = row['SC_tree_id']
        #        self.node_from              = row['node_from']
        #        self.node_to                = row['node_to']

        # ***************************
        # "lot_counts" is the bridge PSI2EVAL
        # ***************************
        self.lot_counts = [0 for x in range(0, 53 * self.plan_range)]

        # ***************************
        # settinng for cost-profit evaluation parameter
        # ***************************
        self.LT_boat = 1  # row['LT_boat']

        self.SGMC_ratio = 0.1  # row['SGMC_ratio']
        self.Cash_Intrest = 0.1  # row['Cash_Intrest']
        self.LOT_SIZE = 1  # row['LOT_SIZE']
        self.REVENUE_RATIO = 0.1  # row['REVENUE_RATIO']

        print("set_plan parameter")
        print("self.LT_boat", self.LT_boat)
        print("self.SGMC_ratio", self.SGMC_ratio)
        print("self.Cash_Intrest", self.Cash_Intrest)
        print("self.LOT_SIZE", self.LOT_SIZE)
        print("self.REVENUE_RATIO", self.REVENUE_RATIO)

        # **************************
        # product_cost_profile
        # **************************
        self.PO_Mng_cost = 1  # row['PO_Mng_cost']
        self.Purchase_cost = 1  # row['Purchase_cost']
        self.WH_COST_RATIO = 0.1  # row['WH_COST_RATIO']
        self.weeks_year = 53 * 5  # row['weeks_year']
        self.WH_COST_RATIO_aWeek = 0.1  # row['WH_COST_RATIO_aWeek']

        print("product_cost_profile parameter")
        print("self.PO_Mng_cost", self.PO_Mng_cost)
        print("self.Purchase_cost", self.Purchase_cost)
        print("self.WH_COST_RATIO", self.WH_COST_RATIO)
        print("self.weeks_year", self.weeks_year)
        print("self.WH_COST_RATIO_aWeek", self.WH_COST_RATIO_aWeek)

        # **************************
        # distribution_condition
        # **************************
        self.Indivisual_Packing = 1  # row['Indivisual_Packing']
        self.Packing_Lot = 1  # row['Packing_Lot']
        self.Transport_Lot = 1  # row['Transport_Lot']
        self.planning_lot_size = 1  # row['planning_lot_size']
        self.Distriburion_Cost = 1  # row['Distriburion_Cost']
        self.SS_days = 7  # row['SS_days']

        print("distribution_condition parameter")
        print("self.Indivisual_Packing", self.Indivisual_Packing)
        print("self.Packing_Lot", self.Packing_Lot)
        print("self.Transport_Lot", self.Transport_Lot)
        print("self.planning_lot_size", self.planning_lot_size)
        print("self.Distriburion_Cost", self.Distriburion_Cost)
        print("self.SS_days", self.SS_days)

        #
        # ******************************
        # evaluation data initialise rewardsを計算の初期化
        # ******************************

        # ******************************
        # Profit_Ratio #float
        # ******************************
        self.eval_profit_ratio = Profit_Ratio = 0.6

        self.eval_profit = 0
        self.eval_revenue = 0

        self.eval_PO_cost = 0
        self.eval_P_cost = 0
        self.eval_WH_cost = 0
        self.eval_SGMC = 0
        self.eval_Dist_Cost = 0

        # ******************************
        # set_EVAL_cash_in_data #list for 53weeks * 5 years # 5年を想定
        # *******************************
        self.Profit = Profit = [0 for i in range(53 * self.plan_range)]
        self.Week_Intrest = Week_Intrest = [0 for i in range(53 * self.plan_range)]
        self.Cash_In = Cash_In = [0 for i in range(53 * self.plan_range)]
        self.Shipped_LOT = Shipped_LOT = [0 for i in range(53 * self.plan_range)]
        self.Shipped = Shipped = [0 for i in range(53 * self.plan_range)]

        # ******************************
        # set_EVAL_cash_out_data #list for 54 weeks
        # ******************************

        self.SGMC = SGMC = [0 for i in range(53 * self.plan_range)]
        self.PO_manage = PO_manage = [0 for i in range(53 * self.plan_range)]
        self.PO_cost = PO_cost = [0 for i in range(53 * self.plan_range)]
        self.P_unit = P_unit = [0 for i in range(53 * self.plan_range)]
        self.P_cost = P_cost = [0 for i in range(53 * self.plan_range)]

        self.I = I = [0 for i in range(53 * self.plan_range)]

        self.I_unit = I_unit = [0 for i in range(53 * self.plan_range)]
        self.WH_cost = WH_cost = [0 for i in range(53 * self.plan_range)]
        self.Dist_Cost = Dist_Cost = [0 for i in range(53 * self.plan_range)]

    def add_child(self, child):
        self.children.append(child)

    def set_parent(self):
        # def set_parent(self, node):

        # treeを辿りながら親ノードを探索
        if self.children == []:

            pass

        else:

            for child in self.children:

                child.parent = self
                # child.parent = node

    # ********************************
    # ココで属性をセット@240417
    # ********************************
    def set_attributes(self, row):
        # self.lot_size = int(row[3])
        # self.leadtime = int(row[4])  # 前提:SS=0
        # self.long_vacation_weeks = eval(row[5])

        self.lot_size = int(row["lot_size"])
        self.leadtime = int(row["leadtime"])  # 前提:SS=0
        self.long_vacation_weeks = eval(row["long_vacation_weeks"])

        # **************************
        # BU_SC_node_profile     business_unit_supplychain_node
        # **************************

        # @240421 機械学習のフラグはstop
        ## **************************
        ## plan_basic_parameter ***sequencing is TEMPORARY
        ## **************************
        #        self.PlanningYear           = row['plan_year']
        #        self.plan_engine            = row['plan_engine']
        #        self.reward_sw              = row['reward_sw']

        # @240421 多段階PSIのフラグはstop
        ## ***************************
        ## business unit identify
        ## ***************************
        #        self.product_name           = row['product_name']
        #        self.SC_tree_id             = row['SC_tree_id']
        #        self.node_from              = row['node_from']
        #        self.node_to                = row['node_to']

        # ***************************
        # ココからcost-profit evaluation 用の属性セット
        # ***************************
        self.LT_boat = float(row["LT_boat"])

        self.SGMC_ratio = float(row["SGMC_ratio"])
        self.Cash_Intrest = float(row["Cash_Intrest"])
        self.LOT_SIZE = float(row["LOT_SIZE"])
        self.REVENUE_RATIO = float(row["REVENUE_RATIO"])

        print("set_plan parameter")
        print("self.LT_boat", self.LT_boat)
        print("self.SGMC_ratio", self.SGMC_ratio)
        print("self.Cash_Intrest", self.Cash_Intrest)
        print("self.LOT_SIZE", self.LOT_SIZE)
        print("self.REVENUE_RATIO", self.REVENUE_RATIO)

        # **************************
        # product_cost_profile
        # **************************
        self.PO_Mng_cost = float(row["PO_Mng_cost"])
        self.Purchase_cost = float(row["Purchase_cost"])
        self.WH_COST_RATIO = float(row["WH_COST_RATIO"])
        self.weeks_year = float(row["weeks_year"])
        self.WH_COST_RATIO_aWeek = float(row["WH_COST_RATIO_aWeek"])

        print("product_cost_profile parameter")
        print("self.PO_Mng_cost", self.PO_Mng_cost)
        print("self.Purchase_cost", self.Purchase_cost)
        print("self.WH_COST_RATIO", self.WH_COST_RATIO)
        print("self.weeks_year", self.weeks_year)
        print("self.WH_COST_RATIO_aWeek", self.WH_COST_RATIO_aWeek)

        # **************************
        # distribution_condition
        # **************************
        self.Indivisual_Packing = float(row["Indivisual_Packing"])
        self.Packing_Lot = float(row["Packing_Lot"])
        self.Transport_Lot = float(row["Transport_Lot"])
        self.planning_lot_size = float(row["planning_lot_size"])
        self.Distriburion_Cost = float(row["Distriburion_Cost"])
        self.SS_days = float(row["SS_days"])

        print("distribution_condition parameter")
        print("self.Indivisual_Packing", self.Indivisual_Packing)
        print("self.Packing_Lot", self.Packing_Lot)
        print("self.Transport_Lot", self.Transport_Lot)
        print("self.planning_lot_size", self.planning_lot_size)
        print("self.Distriburion_Cost", self.Distriburion_Cost)
        print("self.SS_days", self.SS_days)

    # ********************************
    # setting profit-cost attributes@240417
    # ********************************

    ##@240417
    ## ココは、機械学習で使用したEVAL用のcost属性をセットする
    #

    def set_psi_list(self, psi_list):

        self.psi4demand = psi_list

    # supply_plan
    def set_psi_list4supply(self, psi_list):

        self.psi4supply = psi_list

    def get_set_childrenP2S2psi(self, plan_range):

        for child in self.children:

            for w in range(53 * plan_range):

                self.psi4demand[w][0].extend(child.psi4demand[w][3])

#@240526 stopping
#    def confirmedS2childrenP_by_lot(self, plan_range):
#
#        # マザープラントの確定したSを元に、
#        # demand_plan上のlot_idの状態にかかわらず、
#        # supply_planにおいては、
#        # 確定出荷confirmed_Sを元に、以下のpush planを実行する
#
#        # by lotidで一つずつ処理する。
#
#        # 親のconfSのlotidは、どの子nodeから来たのか、出荷先を特定する。
#        #  "demand_planのpsi_listのS" の中をサーチしてisin.listで特定する。
#        # search_node()では子node psiの中にlotidがあるかisinでcheck
#
#        # LT_shiftして、子nodeのPにplaceする。
#        # 親S_lotid => ETA=LT_shift() =>子P[ETA][3]
#
#        # 着荷PをSS_shiftして出荷Sをセット
#        # 子P=>SS_shift(P)=>子S
#
#        # Iの生成
#        # all_PS2I
#
#        # 親の確定出荷confirmedSをを子供の確定Pとして配分
#        # 最後に、conf_PをLT_shiftしてconf_Sにもセットする
#        # @230717このLT_shiftの中では、cpnf_Sはmoveする/extendしない
#
#        #
#        # def feedback_confirmedS2childrenP(self, plan_range):
#        #
#
#        self_confirmed_plan = [[] for _ in range(53 * plan_range)]
#
#        # ************************************
#        # setting mother_confirmed_plan
#        # ************************************
#        for w in range(53 * plan_range):
#
#            ## 親node自身のsupply_planのpsi_list[w][0]がconfirmed_S
#            # self_confirmed_plan[w].extend(self.psi4supply[w][0])
#
#            confirmed_S_lots = self.psi4supply[w][0]
#
#            for lot in confirmed_S_lots:
#
#                if lot == []:
#
#                    pass
#
#                else:
#
#                    # @230722 lot by lot operation
#
#                    ## in parent
#                    # get_lot()
#
#                    # in children
#                    # demand側で出荷先の子nodeを確認し
#                    node_to = check_lot_in_demand_plan(lot)
#
#                    # supply側で出荷先の子nodeに置く
#                    # LT_shiftしてset_P
#                    # SS_shiftしてset_S
#                    place_lot_in_supply_plan(node_to)
#
#        # end of
#        #    def confirmedS2childrenP_by_lot(self, plan_range):
#
#        node_req_plans = []
#        node_confirmed_plans = []
#
#        # ************************************
#        # setting node_req_plans 各nodeからの要求S(=P)
#        # ************************************
#        # 子nodeのdemand_planのpsi_list[w][3]のPがS_requestに相当する
#
#        # すべての子nodesから、S_reqをappendしてnode_req_plansを作る
#        for child in self.children:
#
#            child_S_req = [[] for _ in range(53 * plan_range)]
#
#            for w in range(53 * plan_range):
#
#                child_S_req[w].extend(child.psi4demand[w][3])  # setting P2S
#
#            node_req_plans.append(child_S_req)
#
#        # node_req_plans      子nodeのP=S要求計画planのリストplans
#        # self_confirmed_plan 自nodeの供給計画の確定S
#
#        # 出荷先ごとの出荷計画を求める
#        # node_req_plans = [req_plan_node_1, req_plan_node_2, req_plan_node_3]
#
#        # ***************************
#        # node 分離
#        # ***************************
#        node_confirmed_plans = []
#
#        node_confirmed_plans = separated_node_plan(node_req_plans, self_confirmed_plan)
#
#        for i, child in enumerate(self.children):
#
#            for w in range(53 * plan_range):
#
#                # 子nodeのsupply_planのPにmother_plantの確定Sをセット
#
#                child.psi4supply[w][3] = []  # clearing list
#
#                # i番目の子nodeの確定Sをsupply_planのPにextendでlot_idをcopy
#
#                child.psi4supply[w][3].extend(node_confirmed_plans[i][w])
#
#            # ココまででsupply planの子nodeにPがセットされたことになる。
#
#        # *******************************************
#        # supply_plan上で、PfixをSfixにPISでLT offsetする
#        # *******************************************
#
#        # **************************
#        # Safety Stock as LT shift
#        # **************************
#        safety_stock_week = self.leadtime
#
#        # 案-1:長い搬送工程はPSI処理の対象とし、self.LT=搬送LT(+SS=2)とする
#        #      生産工程は、self.LT=加工LT  SS=0
#        #      保管工程は、self.LT=入出庫LT SS=0
#
#        # 案-2:
#        # process_week = self.process_leadtime
#        # safety_stock_week = self.SS_leadtime
#
#        # demand plan : using "process_LT" + "safety_stock_LT" with backward planning
#        # supply plan : using "process_LT"                     with foreward planning
#
#        # **************************
#        # long vacation weeks
#        # **************************
#        lv_week = self.long_vacation_weeks
#
#        # P to S の計算処理
#        self.psi4supply = shiftP2S_LV(self.psi4supply, safety_stock_week, lv_week)
#
#        ## S to P の計算処理
#        # self.psi4demand = shiftS2P_LV(self.psi4demand, safety_stock_week, lv_week)
#
#



    def set_S2psi(self, pSi):
        # def set_S2psi(self, S_lots_list):
        # def set_Slots2psi4demand(self, S_lots_list):

        # S_lots_listが辞書で、node.psiにセットする

        for w in range(len(pSi)):
            self.psi4demand[w][0].extend(pSi[w])

    def feedback_confirmedS2childrenP(self, plan_range):

        # マザープラントの確定したSを元に、
        # demand_plan上のlot_idの状態にかかわらず、
        # supply_planにおいては、
        # 確定出荷confirmed_Sを元に、以下のpush planを実行する

        # by lotidで一つずつ処理する。

        # 親のconfSのlotidは、どの子nodeから来たのか?
        #  "demand_planのpsi_listのS" の中をサーチしてisin.listで特定する。
        # search_node()では子node psiの中にlotidがあるかisinでcheck

        # LT_shiftして、子nodeのPにplaceする。
        # 親S_lotid => ETA=LT_shift() =>子P[ETA][3]

        # 着荷PをSS_shiftして出荷Sをセット
        # 子P=>SS_shift(P)=>子S

        # Iの生成
        # all_PS2I

        # 親の確定出荷confirmedSをを子供の確定Pとして配分
        # 最後に、conf_PをLT_shiftしてconf_Sにもセットする
        # @230717このLT_shiftの中では、cpnf_Sはmoveする/extendしない

        #
        # def feedback_confirmedS2childrenP(self, plan_range):
        #
        node_req_plans = []
        node_confirmed_plans = []

        self_confirmed_plan = [[] for _ in range(53 * plan_range)]

        # ************************************
        # setting mother_confirmed_plan
        # ************************************
        for w in range(53 * plan_range):

            # 親node自身のsupply_planのpsi_list[w][0]がconfirmed_S
            self_confirmed_plan[w].extend(self.psi4supply[w][0])

        # ************************************
        # setting node_req_plans 各nodeからの要求S(=P)
        # ************************************
        # 子nodeのdemand_planのpsi_list[w][3]のPがS_requestに相当する

        # すべての子nodesから、S_reqをappendしてnode_req_plansを作る
        for child in self.children:

            child_S_req = [[] for _ in range(53 * plan_range)]

            for w in range(53 * plan_range):

                child_S_req[w].extend(child.psi4demand[w][3])  # setting P2S

            node_req_plans.append(child_S_req)

        # node_req_plans      子nodeのP=S要求計画planのリストplans
        # self_confirmed_plan 自nodeの供給計画の確定S

        # 出荷先ごとの出荷計画を求める
        # node_req_plans = [req_plan_node_1, req_plan_node_2, req_plan_node_3]

        # ***************************
        # node 分離
        # ***************************
        node_confirmed_plans = []

        node_confirmed_plans = separated_node_plan(node_req_plans, self_confirmed_plan)

        for i, child in enumerate(self.children):

            for w in range(53 * plan_range):

                # 子nodeのsupply_planのPにmother_plantの確定Sをセット

                child.psi4supply[w][3] = []  # clearing list

                # i番目の子nodeの確定Sをsupply_planのPにextendでlot_idをcopy

                child.psi4supply[w][3].extend(node_confirmed_plans[i][w])

            # ココまででsupply planの子nodeにPがセットされたことになる。

        # *******************************************
        # supply_plan上で、PfixをSfixにPISでLT offsetする
        # *******************************************

        # **************************
        # Safety Stock as LT shift
        # **************************
        safety_stock_week = self.leadtime

        # **************************
        # long vacation weeks
        # **************************
        lv_week = self.long_vacation_weeks

        # P to S の計算処理
        self.psi4supply = shiftP2S_LV(self.psi4supply, safety_stock_week, lv_week)

        ## S to P の計算処理
        # self.psi4demand = shiftS2P_LV(self.psi4demand, safety_stock_week, lv_week)

    def calcPS2I4demand(self):

        # psiS2P = self.psi4demand # copyせずに、直接さわる

        plan_len = 53 * self.plan_range
        # plan_len = len(self.psi4demand)

        for w in range(1, plan_len):  # starting_I = 0 = w-1 / ending_I =plan_len
            # for w in range(1,54): # starting_I = 0 = w-1 / ending_I = 53

            s = self.psi4demand[w][0]
            co = self.psi4demand[w][1]

            i0 = self.psi4demand[w - 1][2]
            i1 = self.psi4demand[w][2]

            p = self.psi4demand[w][3]

            # *********************
            # # I(n-1)+P(n)-S(n)
            # *********************

            work = i0 + p  # 前週在庫と当週着荷分 availables

            # ここで、期末の在庫、S出荷=売上を操作している
            # S出荷=売上を明示的にlogにして、売上として記録し、表示する処理
            # 出荷されたS=売上、在庫I、未出荷COの集合を正しく表現する

            # モノがお金に代わる瞬間

            diff_list = [x for x in work if x not in s]  # I(n-1)+P(n)-S(n)

            self.psi4demand[w][2] = i1 = diff_list

    def calcPS2I4supply(self):

        # psiS2P = self.psi4demand # copyせずに、直接さわる

        plan_len = 53 * self.plan_range
        # plan_len = len(self.psi4supply)

        for w in range(1, plan_len):  # starting_I = 0 = w-1 / ending_I =plan_len
            # for w in range(1,54): # starting_I = 0 = w-1 / ending_I = 53

            s = self.psi4supply[w][0]
            co = self.psi4supply[w][1]

            i0 = self.psi4supply[w - 1][2]
            i1 = self.psi4supply[w][2]

            p = self.psi4supply[w][3]

            # *********************
            # # I(n-1)+P(n)-S(n)
            # *********************

            work = i0 + p  # 前週在庫と当週着荷分 availables

            # memo ここで、期末の在庫、S出荷=売上を操作している
            # S出荷=売上を明示的にlogにして、売上として記録し、表示する処理
            # 出荷されたS=売上、在庫I、未出荷COの集合を正しく表現する

            # モノがお金に代わる瞬間

            diff_list = [x for x in work if x not in s]  # I(n-1)+P(n)-S(n)

            self.psi4supply[w][2] = i1 = diff_list

    def calcPS2I_decouple4supply(self):

        # psiS2P = self.psi4demand # copyせずに、直接さわる

        plan_len = 53 * self.plan_range
        # plan_len = len(self.psi4supply)

        # demand planのSを出荷指示情報=PULL SIGNALとして、supply planSにセット

        for w in range(0, plan_len):
            # for w in range(1,plan_len):

            # pointer参照していないか? 明示的にデータを渡すには?

            self.psi4supply[w][0] = self.psi4demand[w][
                0
            ].copy()  # copy data using copy() method
            # self.psi4supply[w][0]    = self.psi4demand[w][0] # PULL replaced

            # checking pull data
            # show_psi_graph(root_node_outbound,"supply", "HAM", 0, 300 )
            # show_psi_graph(root_node_outbound,"supply", node_show, 0, 300 )

        for w in range(1, plan_len):  # starting_I = 0 = w-1 / ending_I =plan_len

            # demand planSをsupplySにコピー済み
            s = self.psi4supply[w][0]  # PUSH supply S

            co = self.psi4supply[w][1]

            i0 = self.psi4supply[w - 1][2]
            i1 = self.psi4supply[w][2]

            p = self.psi4supply[w][3]

            # *********************
            # # I(n-1)+P(n)-S(n)
            # *********************

            work = i0 + p  # 前週在庫と当週着荷分 availables

            # memo ここで、期末の在庫、S出荷=売上を操作している
            # S出荷=売上を明示的にlogにして、売上として記録し、表示する処理
            # 出荷されたS=売上、在庫I、未出荷COの集合を正しく表現する

            # モノがお金に代わる瞬間

            diff_list = [x for x in work if x not in s]  # I(n-1)+P(n)-S(n)

            self.psi4supply[w][2] = i1 = diff_list

    def calcS2P(self):

        # **************************
        # Safety Stock as LT shift
        # **************************
        # leadtimeとsafety_stock_weekは、ここでは同じ
        safety_stock_week = self.leadtime

        # **************************
        # long vacation weeks
        # **************************
        lv_week = self.long_vacation_weeks

        # S to P の計算処理
        self.psi4demand = shiftS2P_LV(self.psi4demand, safety_stock_week, lv_week)

        pass

    def calcS2P_4supply(self):

        # **************************
        # Safety Stock as LT shift
        # **************************
        # leadtimeとsafety_stock_weekは、ここでは同じ
        safety_stock_week = self.leadtime

        # **************************
        # long vacation weeks
        # **************************
        lv_week = self.long_vacation_weeks

        # S to P の計算処理
        self.psi4supply = shiftS2P_LV_replace(
            self.psi4supply, safety_stock_week, lv_week
        )

        pass

    def set_plan_range_lot_counts(self, plan_range):

        print("node.plan_range", self.name, self.plan_range)

        self.plan_range = plan_range
        self.lot_counts = [0 for x in range(0, 53 * self.plan_range)]

        # ******************************
        # set_EVAL_cash_in_data #list for 53weeks * 5 years # 5年を想定
        # *******************************
        self.Profit = Profit = [0 for i in range(53 * self.plan_range)]
        self.Week_Intrest = Week_Intrest = [0 for i in range(53 * self.plan_range)]
        self.Cash_In = Cash_In = [0 for i in range(53 * self.plan_range)]
        self.Shipped_LOT = Shipped_LOT = [0 for i in range(53 * self.plan_range)]
        self.Shipped = Shipped = [0 for i in range(53 * self.plan_range)]

        # ******************************
        # set_EVAL_cash_out_data #list for 54 weeks
        # ******************************

        self.SGMC = SGMC = [0 for i in range(53 * self.plan_range)]
        self.PO_manage = PO_manage = [0 for i in range(53 * self.plan_range)]
        self.PO_cost = PO_cost = [0 for i in range(53 * self.plan_range)]
        self.P_unit = P_unit = [0 for i in range(53 * self.plan_range)]
        self.P_cost = P_cost = [0 for i in range(53 * self.plan_range)]

        self.I = I = [0 for i in range(53 * self.plan_range)]

        self.I_unit = I_unit = [0 for i in range(53 * self.plan_range)]
        self.WH_cost = WH_cost = [0 for i in range(53 * self.plan_range)]
        self.Dist_Cost = Dist_Cost = [0 for i in range(53 * self.plan_range)]

        for child in self.children:

            child.set_plan_range_lot_counts(plan_range)

    ## 初期値がplan_range=1のままなので、M2Wの後にplan_rangeをRESETする
    # def reset_plan_range_related_attributes(self):

    # ******************************
    # evaluation 操作
    # ******************************
    # ******************************
    # EvalPlanSIP  rewardを計算
    # ******************************

    def set_lot_counts(self):

        plan_len = 53 * self.plan_range

        print("plan_len", plan_len)

        # plan_len = len(self.psi4supply)

        # print("plan_len = 53 * self.plan_range", plan_len, self.plan_range)

        for w in range(0, plan_len):  ### 以下のi+1で1週スタート = W1,W2,W3,,

            # self.lot_counts[]はPurchaseの数 supply sideのconfirmed Pを適用
            # target PSIのPのロットID数をリスト長len()でセット

            # @240423 "i+1"では最後にcount overするのでは???
            # self.lot_counts[i+1] = len(self.psi4supply[i+1][3])  #psi[w][3]=PO

            # ココは素直に"w"でcountしてみる

            print("BEFORE self.lot_counts[w]", w, self.lot_counts[w])
            print("self.psi4supply[w][3]", self.psi4supply[w][3])

            self.lot_counts[w] = len(self.psi4supply[w][3])  # psi[w][3]=PO

            print("AFTER  self.lot_counts[w]", w, self.lot_counts[w])
            print("self.psi4supply[w][3]", self.psi4supply[w][3])

    def EvalPlanSIP(self):

        plan_len = 53 * self.plan_range
        # plan_len = len(self.psi4supply)

        # for i in range( 0 , plan_len ): ###以下のi+1で1週スタート = W1,W2,W3,,

        # @240423 初期versionは"i+1"だったが、psi4supply[][]では"i"で良い? =>良くない
        # @240423 plan_len - 1 でout of rangeを回避
        # for i in range( 0 , plan_len - 1 ): ###以下のi+1で1週スタート = W1,W2,W3
        # 240424 i+1をiにして、start endをキレイにしたい・・・
        # for i in range( 0 , plan_len ): ###以下のi+1で1週スタート = W1,W2,W3,,

        # 240424 53*self.plan_rangeでfull spanで計算

        for i in range(
            0, 53 * self.plan_range
        ):  ###以下のi+1で1週スタート = W1,W2,W3,,,

            # calc PO_manage 各週の(梱包単位)LOT数をカウントし輸送ロットで丸め
            # =IF(SUM(G104:G205)=0,0,QUOTIENT(SUM(G104:G205),$C$17)+1)

            ### i+1 = W1,W2,W3,,,

            if self.lot_counts[i] == 0:  ## ロットが発生しない週の分母=0対応
                self.PO_manage[i] = 0
            else:
                self.PO_manage[i] = self.lot_counts[i] // self.Transport_Lot + 1

            # Distribution Cost =$C$19*G12
            self.Dist_Cost[i] = self.Distriburion_Cost * self.PO_manage[i]

            # 在庫self.I_year[w] <=> 在庫self.psi4supply[w][2]
            # Inventory UNIT =G97/$C$7
            # self.I_unit[i]  = self.I_year[i] / self.planning_lot_size

            # print("EvalPlanSIP len(self.psi4supply[i][2])", self.name, len(self.psi4supply[i][2]), self.psi4supply[i][2], self.planning_lot_size )

            self.I_unit[i] = len(self.psi4supply[i][2]) / float(self.planning_lot_size)

            # WH_COST by WEEK =G19*$C$11*$C$8
            self.WH_cost[i] = (
                float(self.I_unit[i]) * self.WH_COST_RATIO * self.REVENUE_RATIO
            )

            # Purchase by UNIT =G98/$C$7
            # self.P_unit[i]    = self.P_year[i] / self.planning_lot_size

            self.P_unit[i] = len(self.psi4supply[i][3]) / float(self.planning_lot_size)

            # Purchase Cost =G15*$C$8*$C$10
            self.P_cost[i] = (
                float(self.P_unit[i]) * self.Purchase_cost * self.REVENUE_RATIO
            )

            # PO manage cost =G15*$C$9*$C$8 ### PO_manage => P_unit
            ### self.PO_manage[i] = self.PO_manage[i] ###
            self.PO_cost[i] = self.P_unit[i] * self.PO_Mng_cost * self.REVENUE_RATIO
            #            # PO manage cost =G12*$C$9*$C$8
            #            self.PO_cost[i]   = self.PO_manage[i] * self.PO_Mng_cost * self.REVENUE_RATIO

            # =MIN(G95+G96,G97+G98) shipped
            # self.Shipped[i] = min( self.S_year[i] + self.CO_year[i] , self.I_year[i] + self.IP_year[i] )

            self.Shipped[i] = min(
                len(self.psi4supply[i][0]) + len(self.psi4supply[i][1]),
                len(self.psi4supply[i][2]) + len(self.psi4supply[i][3]),
            )

            # =G9/$C$7 shipped UNIT
            # self.Shipped_LOT[i] = self.Shipped[i] / self.planning_lot_size

            # @240424 memo すでにlot_sizeでの丸めは処理されているハズ
            self.Shipped_LOT[i] = self.Shipped[i]  ###**/ self.planning_lot_size

            # =$C$8*G8 Cach In
            self.Cash_In[i] = self.REVENUE_RATIO * self.Shipped_LOT[i]

            # =$C$6*(52-INT(RIGHT(G94,LEN(G94)-1)))/52 Week_Intrest Cash =5%/52
            self.Week_Intrest[i] = self.Cash_Intrest * (52 - (i)) / 52

            # =G7*$C$5 Sales and General managnt cost
            self.SGMC[i] = self.Cash_In[i] * self.SGMC_ratio

            # =G7*(1+G6)-G13-G16-G20-G5-G21 Profit
            self.Profit[i] = (
                self.Cash_In[i] * (1 + self.Week_Intrest[i])
                - self.PO_cost[i]
                - self.P_cost[i]
                - self.WH_cost[i]
                - self.SGMC[i]
                - self.Dist_Cost[i]
            )

        # ******************************
        # reward 切り替え
        # ******************************
        # =SUM(H4:BH4)/SUM(H7:BH7) profit_ratio
        if sum(self.Cash_In[1:]) == 0:
            self.eval_profit_ratio = 0
        else:

            # ********************************
            # 売切る商売や高級店ではPROFITをrewardに使うことが想定される
            # ********************************
            self.eval_profit = sum(self.Profit[1:])  # *** PROFIT

            # ********************************
            # 前線の小売りの場合、revenueをrewardに使うことが想定される
            # ********************************
            self.eval_revenue = sum(self.Cash_In[1:])  # *** REVENUE

            self.eval_PO_cost = sum(self.PO_cost[1:])
            self.eval_P_cost = sum(self.P_cost[1:])
            self.eval_WH_cost = sum(self.WH_cost[1:])
            self.eval_SGMC = sum(self.SGMC[1:])
            self.eval_Dist_Cost = sum(self.Dist_Cost[1:])

            # ********************************
            # 一般的にはprofit ratioをrewardに使うことが想定される
            # ********************************
            self.eval_profit_ratio = sum(self.Profit[1:]) / sum(self.Cash_In[1:])

        print("")
        print("Eval node", self.name)
        print("profit       ", self.eval_profit)

        print("PO_cost      ", self.eval_PO_cost)
        print("P_cost       ", self.eval_P_cost)
        print("WH_cost      ", self.eval_WH_cost)
        print("SGMC         ", self.eval_SGMC)
        print("Dist_Cost    ", self.eval_Dist_Cost)

        print("revenue      ", self.eval_revenue)
        print("profit_ratio ", self.eval_profit_ratio)

        # @240423 returnしないで、self.xxxxでnode instanceの中にセット
        # return( self.profit, self.revenue, self.profit_ratio )

    # *****************************
    # @240121 ここでCPU_LOTsを抽出する
    # *****************************
    def extract_CPU(self, csv_writer):

        plan_len = 53 * self.plan_range  # 計画長をセット

        # w=1から抽出処理

        # starting_I = 0 = w-1 / ending_I=plan_len
        for w in range(1, plan_len):

            # for w in range(1,54):   #starting_I = 0 = w-1 / ending_I = 53

            s = self.psi4supply[w][0]

            co = self.psi4supply[w][1]

            i0 = self.psi4supply[w - 1][2]
            i1 = self.psi4supply[w][2]

            p = self.psi4supply[w][3]

            # ***************************
            # write CPU
            # ***************************
            #
            # ISO_week_no,
            # CPU_lot_id,
            # S-I-P区分,
            # node座標(longitude, latitude),
            # step(高さ=何段目),
            # lot_size
            # ***************************

            # ***************************
            # write "s" CPU
            # ***************************
            for step_no, lot_id in enumerate(s):

                # lot_idを計画週YYYYWWでユニークにする
                lot_id_yyyyww = lot_id + str(self.plan_year_st) + str(w).zfill(3)

                CPU_row = [
                    w,
                    lot_id_yyyyww,
                    "s",
                    self.name,
                    self.longtude,
                    self.latitude,
                    step_no,
                    self.lot_size,
                ]

                csv_writer.writerow(CPU_row)

            # ***************************
            # write "i1" CPU
            # ***************************
            for step_no, lot_id in enumerate(i1):

                # lot_idを計画週YYYYWWでユニークにする
                lot_id_yyyyww = lot_id + str(self.plan_year_st) + str(w).zfill(3)

                CPU_row = [
                    w,
                    lot_id_yyyyww,
                    "i1",
                    self.name,
                    self.longtude,
                    self.latitude,
                    step_no,
                    self.lot_size,
                ]

                csv_writer.writerow(CPU_row)

            # ***************************
            # write "p" CPU
            # ***************************
            for step_no, lot_id in enumerate(p):

                # lot_idを計画週YYYYWWでユニークにする
                lot_id_yyyyww = lot_id + str(self.plan_year_st) + str(w).zfill(3)

                CPU_row = [
                    w,
                    lot_id_yyyyww,
                    "p",
                    self.name,
                    self.longtude,
                    self.latitude,
                    step_no,
                    self.lot_size,
                ]

                csv_writer.writerow(CPU_row)

            ## *********************
            ## s checking demand_lot_idのリスト
            ## *********************
            # if w == 100:
            #
            #    print('checking w100 s',s)


# ****************************
# supply chain tree creation
# ****************************
def create_tree_set_attribute(file_name):

    root_node_name = ""  # init setting

    with open(file_name, "r", encoding="utf-8-sig") as f:

        reader = csv.DictReader(f)  # header行の項目名をkeyにして、辞書生成

        # for row in reader:

        # デフォルトでヘッダー行はスキップされている
        # next(reader)  # ヘッダー行をスキップ

        # nodeインスタンスの辞書を作り、親子の定義に使う
        # nodes = {row[2]: Node(row[2]) for row in reader}
        nodes = {row["child_node_name"]: Node(row["child_node_name"]) for row in reader}

        f.seek(0)  # ファイルを先頭に戻す #@240421 大丈夫か?

        next(reader)  # ヘッダー行をスキップ

        ## next(reader)  # root行をスキップしないでloopに入る
        #
        ## readerの一行目root rawのroot_node_nameを取得する

        for row in reader:

            # if row[0] == "root":
            if row["Parent_node"] == "root":

                # root_node_name = row[1]
                root_node_name = row["Child_node"]

            else:

                print("row['Parent_node'] ", row["Parent_node"])

                # parent = nodes[row[0]]
                parent = nodes[row["Parent_node"]]

                # child = nodes[row[1]]
                child = nodes[row["Child_node"]]

                parent.add_child(child)

                # 子ノードにアプリケーション属性セット

                print("row", row)

                child.set_attributes(row)

    return nodes, root_node_name  # すべてのインスタンス・ポインタを返して使う
    # return nodes['JPN']   # "JPN"のインスタンス・ポインタ


def create_tree(csv_file):

    root_node_name = ""  # init setting

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        next(reader)  # ヘッダー行をスキップ

        # nodeインスタンスの辞書を作り、親子の定義に使う
        nodes = {row[2]: Node(row[2]) for row in reader}

        f.seek(0)  # ファイルを先頭に戻す

        next(reader)  # ヘッダー行をスキップ

        # next(reader)  # root行をスキップしないでloopに入る

        # readerの一行目root rawのroot_node_nameを取得する

        for row in reader:

            if row[0] == "root":

                root_node_name = row[1]

            else:

                parent = nodes[row[0]]

                child = nodes[row[1]]

                parent.add_child(child)

                # 子ノードにアプリケーション属性セット
                child.set_attributes(row)

    return nodes, root_node_name  # すべてのインスタンス・ポインタを返して使う
    # return nodes['JPN']   # "JPN"のインスタンス・ポインタ


def set_psi_lists(node, node_psi_dict):
    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []:  # 子nodeがないleaf nodeの場合

        node.set_psi_list(node_psi_dict.get(node.name))

    else:

        node.get_set_childrenP2S2psi(node.plan_range)

    for child in node.children:

        set_psi_lists(child, node_psi_dict)


def set_Slots2psi4OtDm(node, S_lots_list):

    for child in node.children:

        set_Slots2psi4OtDm(child, S_lots_list)

    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []:  # 子nodeがないleaf nodeの場合

        # S_lots_listが辞書で、node.psiにセットする
        node.set_S2psi(S_lots_list)

        # shifting S2P
        node.calcS2P()  # backward plan with postordering

    else:

        # gathering S and Setting S
        node.get_set_childrenP2S2psi(node.plan_range)
        # node.get_set_childrenS2psi(plan_range)

        # shifting S2P
        node.calcS2P()  # backward plan with postordering


# _dictで渡す
def set_Slots2psi4demand(node, S_lots_dict):
    # def set_psi_lists_postorder(node, node_psi_dict):

    for child in node.children:

        set_Slots2psi4demand(child, S_lots_dict)

    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []:  # 子nodeがないleaf nodeの場合

        # 辞書のgetメソッドでキーから値を取得。キーが存在しない場合はNone
        pSi = S_lots_dict.get(node.name)

        node.set_S2psi(pSi)

        # shifting S2P
        node.calcS2P()  # backward plan with postordering

    else:

        # gathering S and Setting S
        node.get_set_childrenP2S2psi(node.plan_range)

        # shifting S2P
        node.calcS2P()  # backward plan with postordering


def set_psi_lists_postorder(node, node_psi_dict):

    for child in node.children:

        set_psi_lists_postorder(child, node_psi_dict)

    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []:  # 子nodeがないleaf nodeの場合

        # 辞書のgetメソッドでキーから値を取得。キーが存在しない場合はNone
        node.set_psi_list(node_psi_dict.get(node.name))

        # shifting S2P
        node.calcS2P()  # backward plan with postordering

    else:

        # gathering S and Setting S
        node.get_set_childrenP2S2psi(node.plan_range)
        # node.get_set_childrenS2psi(plan_range)

        # shifting S2P
        node.calcS2P()  # backward plan with postordering


def make_psi4supply(node, node_psi_dict):

    plan_range = node.plan_range

    node_psi_dict[node.name] = [[[] for j in range(4)] for w in range(53 * plan_range)]

    for child in node.children:

        make_psi4supply(child, node_psi_dict)

    return node_psi_dict


def set_psi_lists4supply(node, node_psi_dict):

    node.set_psi_list4supply(node_psi_dict.get(node.name))

    for child in node.children:

        set_psi_lists4supply(child, node_psi_dict)


def find_path_to_leaf_with_parent(node, leaf_node, current_path=[]):

    current_path.append(leaf_node.name)

    if node.name == leaf_node.name:

        return current_path

    else:

        parent = leaf_node.parent

        path = find_path_to_leaf_with_parent(node, parent, current_path.copy())

    return path


#        if path:
#
#            return path


def find_path_to_leaf(node, leaf_node, current_path=[]):

    current_path.append(node.name)

    if node.name == leaf_node.name:

        return current_path

    for child in node.children:

        path = find_path_to_leaf(child, leaf_node, current_path.copy())

        if path:

            return path


def flatten_list(data_list):
    for item in data_list:
        if isinstance(item, list):
            yield from flatten_list(item)
        else:
            yield item


def children_nested_list(data_list):

    flat_list = set(flatten_list(data_list))

    return flat_list


def extract_node_name(stringA):
    # 右側の数字部分を除外してnode名を取得

    index = len(stringA) - 1

    while index >= 0 and stringA[index].isdigit():

        index -= 1

    node_name = stringA[: index + 1]

    return node_name


def place_P_in_supply(w, child, lot):  # lot LT_shift on P

    # *******************************************
    # supply_plan上で、PfixをSfixにPISでLT offsetする
    # *******************************************

    # **************************
    # Safety Stock as LT shift
    # **************************

    # leadtimeとsafety_stock_weekは、ここでは同じ
    # safety_stock_week = child.leadtime
    LT_SS_week = child.leadtime

    # **************************
    # long vacation weeks
    # **************************
    lv_week = child.long_vacation_weeks

    ## P to S の計算処理
    # self.psi4supply = shiftP2S_LV(self.psi4supply, safety_stock_week, lv_week)

    ### S to P の計算処理
    ##self.psi4demand = shiftS2P_LV(self.psi4demand, safety_stock_week, lv_week)

    # my_list = [1, 2, 3, 4, 5]
    # for i in range(2, len(my_list)):
    #    my_list[i] = my_list[i-1] + my_list[i-2]

    # 0:S
    # 1:CO
    # 2:I
    # 3:P

    # @230723モデルの基本的な定義について
    # このモデルの前提では、輸送工程をPSI計算しているので、
    # ETD=ETAとなっている。。。不自然???

    # LT:leadtime SS:safty stockは1つ
    # foreward planで、「親confirmed_S出荷=子confirmed_P着荷」と表現
    eta_plan = w + LT_SS_week  # ETA=ETDなので、+LTすると次のETAとなる

    # etd_plan = w + ss # ss:safty stock
    # eta_plan = w - ss # ss:safty stock

    # *********************
    # 着荷週が事業所nodeの非稼働週の場合 +1次週の着荷とする
    # *********************
    eta_shift = check_lv_week_fw(lv_week, eta_plan)  # ETD:Eatimate TimeDept.

    # リスト追加 extend
    # 安全在庫とカレンダ制約を考慮した着荷予定週Pに、w週Sからoffsetする

    # lot by lot operation
    # confirmed_P made by shifting parent_conf_S

    # ***********************
    # place_lot_supply_plan
    # ***********************

    # ここは、"REPLACE lot"するので、appendの前にchild psiをzero clearしてから

    # 今回のmodelでは、輸送工程もpsi nodeと同等に扱っている(=POではない)ので
    # 親のconfSを「そのままのWで」子のconfPに置く place_lotする
    child.psi4supply[w][3].append(lot)

    # 親のconfSを「輸送LT=0, 加工LT+SSでwをshiftして」子confSにplace_lotする
    child.psi4supply[eta_shift][0].append(lot)


def set_parent_all(node):
    # preordering

    if node.children == []:

        pass

    else:

        node.set_parent()  # この中で子nodeを見て親を教える。
        # def set_parent(self)

    for child in node.children:

        set_parent_all(child)


def print_parent_all(node):
    # preordering

    if node.children == []:

        pass

    else:

        print("node.parent and children", node.name, node.children)

    for child in node.children:

        print("child and parent", child.name, node.name)

        print_parent_all(child)


# position座標のセット
def set_position_all(node, node_position_dic):
    # preordering

    node.longtude = node_position_dic[node.name][0]
    node.latitude = node_position_dic[node.name][1]

    if node.children == []:

        pass

    else:

        for child in node.children:

            set_position_all(child, node_position_dic)


# def place_S_in_supply(child, lot): # lot SS shift on S


# 確定Pのセット

# replace lotするために、事前に、
# 出荷先となるすべてのchildren nodesのS[w][0]とP[w][3]をクリアしてplace lot
# ※出荷先ship2nodeを特定してからreplaceするのは難しいので、


def ship_lots2market(node, nodes):

    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []:  # 子nodeがないleaf nodeの場合

        pass

    else:

        # returnせずに子nodeのpsiのPに返す child.psi4demand[w][3]に直接セット
        # feedback_confirmedS2childrenP(node_req_plans, S_confirmed_plan)

        # ************************************
        # clearing children P[w][3] and S[w][0]
        # ************************************
        # replace lotするために、事前に、
        # 出荷先となるすべてのchildren nodesのS[w][0]とP[w][3]をクリア

        for child in node.children:

            for w in range(53 * node.plan_range):

                child.psi4supply[w][0] = []
                child.psi4supply[w][3] = []

        # lotidから、leaf_nodeを特定し、出荷先ship2nodeに出荷することは、
        # すべての子nodeに出荷することになる

        # ************************************
        # setting mother_confirmed_S
        # ************************************
        # このnode内での子nodeへの展開
        for w in range(53 * node.plan_range):

            # ある拠点の週次 生産出荷予定lots_list

            confirmed_S_lots = node.psi4supply[w][0]  # 親の確定出荷confS lot

            # 出荷先nodeを特定して

            # 一般には、下記のLT shiftだが・・・・・
            # 出荷先の ETA = LT_shift(ETD) でP place_lot
            # 工程中の ETA = SS_shift(ETD) でS place_lot

            # 本モデルでは、輸送工程 = modal_nodeを想定して・・・・・
            # 出荷先の ETA = 出荷元ETD        でP place_lot
            # 工程中の ETA = LT&SS_shift(ETD) でS place_lot
            # というイビツなモデル定義・・・・・

            # 直感的なPO=INVOICEという考え方に戻すべきかも・・・・・
            #
            # modal shiftのmodelingをLT_shiftとの拡張で考える???
            # modal = BOAT/AIR/QURIE
            # LT_shift(modal, w, ,,,,

            for lot in confirmed_S_lots:

                if lot == []:

                    pass

                else:

                    # lotidからleaf_nodeのpointerを返す
                    leaf_node_name = extract_node_name(lot)

                    leaf_node = nodes[leaf_node_name]

                    # 末端からあるnodeAまでleaf_nodeまでのnode_listをpathで返す

                    current_path = []
                    path = []

                    path = find_path_to_leaf_with_parent(node, leaf_node, current_path)

                    # nodes_listを逆にひっくり返す
                    path.reverse()

                    # あるnodeAから末端のleaf_nodeまでのnode_listをpathで返す
                    # path = find_path_to_leaf(node, leaf_node,current_path)

                    # 出荷先nodeはnodeAの次node、path[1]になる
                    ship2node_name = path[1]

                    ship2node = nodes[ship2node_name]

                    # ここでsupply planを更新している
                    # 出荷先nodeのPSIのPとSに、confirmed_S中のlotをby lotで置く
                    place_P_in_supply(w, ship2node, lot)

    for child in node.children:

        ship_lots2market(child, nodes)


def feedback_psi_lists(node, node_psi_dict, nodes):

    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []:  # 子nodeがないleaf nodeの場合

        pass

    else:

        # ************************************
        # clearing children P[w][3] and S[w][0]
        # ************************************
        # replace lotするために、事前に、
        # 出荷先となるすべてのchildren nodesのS[w][0]とP[w][3]をクリア

        for child in node.children:

            for w in range(53 * node.plan_range):

                child.psi4supply[w][0] = []
                child.psi4supply[w][3] = []

        # lotidから、leaf_nodeを特定し、出荷先ship2nodeに出荷することは、
        # すべての子nodeに出荷することになる

        # ************************************
        # setting mother_confirmed_S
        # ************************************
        # このnode内での子nodeへの展開
        for w in range(53 * node.plan_range):

            confirmed_S_lots = node.psi4supply[w][0]  # 親の確定出荷confS lot

            # 出荷先nodeを特定して

            # 一般には、下記のLT shiftだが・・・・・
            # 出荷先の ETA = LT_shift(ETD) でP place_lot
            # 工程中の ETA = SS_shift(ETD) でS place_lot

            # 本モデルでは、輸送工程 = modal_nodeを想定して・・・・・
            # 出荷先の ETA = 出荷元ETD        でP place_lot
            # 工程中の ETA = LT&SS_shift(ETD) でS place_lot
            # というイビツなモデル定義・・・・・

            # 直感的なPO=INVOICEという考え方に戻すべきかも・・・・・
            #
            # modal shiftのmodelingをLT_shiftとの拡張で考える???
            # modal = BOAT/AIR/QURIE
            # LT_shift(modal, w, ,,,,

            for lot in confirmed_S_lots:

                if lot == []:

                    pass

                else:

                    # *********************************************************
                    # child#ship2node = find_node_to_ship(node, lot)
                    # lotidからleaf_nodeのpointerを返す
                    leaf_node_name = extract_node_name(lot)

                    leaf_node = nodes[leaf_node_name]

                    # 末端からあるnodeAまでleaf_nodeまでのnode_listをpathで返す

                    current_path = []
                    path = []

                    path = find_path_to_leaf_with_parent(node, leaf_node, current_path)

                    # nodes_listを逆にひっくり返す
                    path.reverse()

                    # 出荷先nodeはnodeAの次node、path[1]になる
                    ship2node_name = path[1]

                    ship2node = nodes[ship2node_name]

                    # ここでsupply planを更新している
                    # 出荷先nodeのPSIのPとSに、confirmed_S中のlotをby lotで置く
                    place_P_in_supply(w, ship2node, lot)

    for child in node.children:

        feedback_psi_lists(child, node_psi_dict, nodes)


def get_all_psi4demand(node, node_all_psi):

    node_all_psi[node.name] = node.psi4demand

    for child in node.children:

        get_all_psi4demand(child, node_all_psi)

    return node_all_psi


def get_all_psi4demand_postorder(node, node_all_psi):

    node_all_psi[node.name] = node.psi4demand

    for child in node.children:

        get_all_psi4demand_postorder(child, node_all_psi)

    return node_all_psi


def get_all_psi4supply(node, node_all_psi):

    node_all_psi[node.name] = node.psi4supply

    for child in node.children:

        get_all_psi4supply(child, node_all_psi)

    return node_all_psi


#@240526 stopping
#def set_all_I4bullwhip(node):
#
#    for child in node.children:
#
#        set_all_I4bullwhip(child)
#
#    # node辞書に時系列set
#    # node.set_I4bullwhip()
#
#    I_hi_len = []  # 在庫の高さ=リストの長さ
#
#    for w in range(len(node.psi4demand)):
#
#        I_hi_len.append(len(node.psi4demand[w][2]))
#
#    node_I4bullwhip[node.name] = I_hi_len
#
#    return node_I4bullwhip


def calc_all_psi2i4demand(node):

    # node_search.append(node)

    node.calcPS2I4demand()

    for child in node.children:

        calc_all_psi2i4demand(child)


def calcPS2I4demand2dict(node, node_psi_dict_In4Dm):

    plan_len = 53 * node.plan_range

    for w in range(1, plan_len):  # starting_I = 0 = w-1 / ending_I =plan_len

        s = node.psi4demand[w][0]
        co = node.psi4demand[w][1]

        i0 = node.psi4demand[w - 1][2]
        i1 = node.psi4demand[w][2]

        p = node.psi4demand[w][3]

        # *********************
        # # I(n-1)+P(n)-S(n)
        # *********************

        work = i0 + p  # 前週在庫と当週着荷分 availables

        # ここで、期末の在庫、S出荷=売上を操作している
        # S出荷=売上を明示的にlogにして、売上として記録し、表示する処理
        # 出荷されたS=売上、在庫I、未出荷COの集合を正しく表現する

        # モノがお金に代わる瞬間

        diff_list = [x for x in work if x not in s]  # I(n-1)+P(n)-S(n)

        node.psi4demand[w][2] = i1 = diff_list

    node_psi_dict_In4Dm[node.name] = node.psi4demand

    return node_psi_dict_In4Dm


# ********************
# inbound demand PS2I
# ********************


def calc_all_psi2i4demand_postorder(node, node_psi_dict_In4Dm):

    for child in node.children:

        calc_all_psi2i4demand_postorder(child, node_psi_dict_In4Dm)

    node_psi_dict_In4Dm = calcPS2I4demand2dict(node, node_psi_dict_In4Dm)

    node.psi4demand = node_psi_dict_In4Dm[node.name]  # 辞書をインスタンスに戻す


def calc_all_psi2i4supply(node):

    node.calcPS2I4supply()

    for child in node.children:

        calc_all_psi2i4supply(child)


def calc_all_psi2i_decouple4supply(
    node, nodes_decouple, decouple_flag, node_psi_dict_Ot4Dm, nodes_outbound
):

    # ********************************
    if node.name in nodes_decouple:

        decouple_flag = "ON"
    # ********************************

    if decouple_flag == "OFF":

        node.calcPS2I4supply()  # calc_psi with PUSH_S

    elif decouple_flag == "ON":

        # decouple nodeの場合は、psi処理後のsupply plan Sを出荷先nodeに展開する
        #
        # demand plan Sをsupply plan Sにcopyし、psi処理後に、supply plan Sを
        # PULL S / confirmed Sとして以降nodeのsupply planのSを更新する

        # ********************************

        if node.name in nodes_decouple:

            # 明示的に.copyする。
            plan_len = 53 * node.plan_range

            for w in range(0, plan_len):

                node.psi4supply[w][0] = node.psi4demand[w][0].copy()

            node.calcPS2I4supply()  # calc_psi with PULL_S

            # *******************************************
            # decouple nodeは、pull_Sで出荷指示する
            # *******************************************
            ship_lots2market(node, nodes_outbound)

        else:

            #
            # decouple から先のnodeのpsi計算
            #

            # 明示的に.copyする。
            plan_len = 53 * node.plan_range

            for w in range(0, plan_len):

                node.psi4supply[w][0] = node.psi4demand[w][0].copy()  # @230728

            node.calcPS2I4supply()  # calc_psi with PULL_S

    else:

        print("error node decouple process " + node.name + " and " + nodes_decouple)

    for child in node.children:

        calc_all_psi2i_decouple4supply(
            child, nodes_decouple, decouple_flag, node_psi_dict_Ot4Dm, nodes_outbound
        )


def calc_all_psi2i_postorder(node):

    for child in node.children:

        calc_all_psi2i_postorder(child)

    node.calcPS2I4demand()  # backward plan with postordering


def calc_all_psiS2P_postorder(node):

    for child in node.children:

        calc_all_psiS2P_postorder(child)

    node.calcS2P()  # backward plan with postordering


# nodeを手繰りながらnode_psi_dict辞書を初期化する
def make_psi_space_dict(node, node_psi_dict, plan_range):

    psi_list = [[[] for j in range(4)] for w in range(53 * plan_range)]

    node_psi_dict[node.name] = psi_list  # 新しいdictにpsiをセット

    for child in node.children:

        make_psi_space_dict(child, node_psi_dict, plan_range)

    return node_psi_dict


# nodeを手繰りながらnode_psi_dict辞書を初期化する
def make_psi_space_zero_dict(node, node_psi_dict, plan_range):

    psi_list = [[0 for j in range(4)] for w in range(53 * plan_range)]

    node_psi_dict[node.name] = psi_list  # 新しいdictにpsiをセット

    for child in node.children:

        make_psi_space_zero_dict(child, node_psi_dict, plan_range)

    return node_psi_dict


# ****************************
# 辞書をinbound tree nodeのdemand listに接続する
# ****************************


def set_dict2tree_psi(node, attr_name, node_psi_dict):

    setattr(node, attr_name, node_psi_dict.get(node.name))

    # node.psi4supply = node_psi_dict.get(node.name)

    for child in node.children:

        set_dict2tree_psi(child, attr_name, node_psi_dict)


def set_dict2tree_InOt4AC(node, node_psi_dict):

    node.psi4accume = node_psi_dict.get(node.name)

    for child in node.children:

        set_dict2tree_InOt4AC(child, node_psi_dict)


def set_dict2tree_In4Dm(node, node_psi_dict):

    node.psi4demand = node_psi_dict.get(node.name)

    for child in node.children:

        set_dict2tree_In4Dm(child, node_psi_dict)


def set_dict2tree_In4Sp(node, node_psi_dict):

    node.psi4supply = node_psi_dict.get(node.name)

    for child in node.children:

        set_dict2tree_In4Sp(child, node_psi_dict)


def set_plan_range(node, plan_range):

    node.plan_range = plan_range

    print("node.plan_range", node.name, node.plan_range)

    for child in node.children:

        set_plan_range(child, plan_range)


# **********************************
# 多次元リストの要素数をcount
# **********************************
def multi_len(l):
    count = 0
    if isinstance(l, list):
        for v in l:
            count += multi_len(v)
        return count
    else:
        return 1


# a way of leveling
#
#      supply           demand
# ***********************************
# *                *                *
# * carry_over_out *                *
# *                *   S_lot        *
# *** capa_ceil ****   get_S_lot    *
# *                *                *
# *  S_confirmed   *                *
# *                *                *
# *                ******************
# *                *  carry_over_in *
# ***********************************

#
# carry_over_out = ( carry_over_in + S_lot ) - capa
#


def leveling_operation(carry_over_in, S_lot, capa_ceil):

    demand_side = []

    demand_side.extend(carry_over_in)

    demand_side.extend(S_lot)

    if len(demand_side) <= capa_ceil:

        S_confirmed = demand_side

        carry_over_out = []  # 繰り越し無し

    else:

        S_confirmed = demand_side[:capa_ceil]  # 能力内を確定する

        carry_over_out = demand_side[capa_ceil:]  # 能力を超えた分を繰り越す

    return S_confirmed, carry_over_out


# **************************
# leveling production
# **************************
def confirm_S(S_lots_list, prod_capa_limit, plan_range):

    S_confirm_list = [[] for i in range(53 * plan_range)]  # [[],[],,,,[]]

    carry_over_in = []

    week_no = 53 * plan_range - 1

    for w in range(week_no, -1, -1):  # 6,5,4,3,2,1,0

        S_lot = S_lots_list[w]
        capa_ceil = prod_capa_limit[w]

        S_confirmed, carry_over_out = leveling_operation(
            carry_over_in, S_lot, capa_ceil
        )

        carry_over_in = carry_over_out

        S_confirm_list[w] = S_confirmed

    return S_confirm_list

    # *********************************
    # visualise with 3D bar graph
    # *********************************


def show_inbound_demand(root_node_inbound):

    nodes_list, node_psI_list = extract_nodes_psI4demand(root_node_inbound)

    fig = visualise_psi_label(node_psI_list, nodes_list)

    offline.plot(fig, filename="inbound_demand_plan_010.html")


def connect_outbound2inbound(root_node_outbound, root_node_inbound):

    # ***************************************
    # setting root node OUTBOUND to INBOUND
    # ***************************************

    plan_range = root_node_outbound.plan_range

    for w in range(53 * plan_range):

        root_node_inbound.psi4demand[w][0] = root_node_outbound.psi4supply[w][0].copy()
        root_node_inbound.psi4demand[w][1] = root_node_outbound.psi4supply[w][1].copy()
        root_node_inbound.psi4demand[w][2] = root_node_outbound.psi4supply[w][2].copy()
        root_node_inbound.psi4demand[w][3] = root_node_outbound.psi4supply[w][3].copy()

        root_node_inbound.psi4supply[w][0] = root_node_outbound.psi4supply[w][0].copy()
        root_node_inbound.psi4supply[w][1] = root_node_outbound.psi4supply[w][1].copy()
        root_node_inbound.psi4supply[w][2] = root_node_outbound.psi4supply[w][2].copy()
        root_node_inbound.psi4supply[w][3] = root_node_outbound.psi4supply[w][3].copy()


#  class NodeのメソッドcalcS2Pと同じだが、node_psiの辞書を更新してreturn
def calc_bwd_inbound_si2p(node, node_psi_dict_In4Dm):

    # **************************
    # Safety Stock as LT shift
    # **************************
    # leadtimeとsafety_stock_weekは、ここでは同じ
    safety_stock_week = node.leadtime

    # **************************
    # long vacation weeks
    # **************************
    lv_week = node.long_vacation_weeks

    # S to P の計算処理  # dictに入れればself.psi4supplyから接続して見える
    node_psi_dict_In4Dm[node.name] = shiftS2P_LV(
        node.psi4demand, safety_stock_week, lv_week
    )

    return node_psi_dict_In4Dm


def calc_bwd_inbound_all_si2p(node, node_psi_dict_In4Dm):

    plan_range = node.plan_range

    # ********************************
    # inboundは、親nodeのSをそのままPに、shift S2Pして、node_spi_dictを更新
    # ********************************
    #    S2P # dictにlistセット
    node_psi_dict_In4Dm = calc_bwd_inbound_si2p(node, node_psi_dict_In4Dm)

    # *********************************
    # 子nodeがあればP2_child.S
    # *********************************

    if node.children == []:

        pass

    else:

        # inboundの場合には、dict=[]でセット済　代入する[]になる
        # 辞書のgetメソッドでキーnameから値listを取得。
        # キーが存在しない場合はNone
        # self.psi4demand = node_psi_dict_In4Dm.get(self.name)

        for child in node.children:

            for w in range(53 * plan_range):

                # move_lot P2S
                child.psi4demand[w][0] = node.psi4demand[w][3].copy()

    for child in node.children:

        calc_bwd_inbound_all_si2p(child, node_psi_dict_In4Dm)

    # stop 返さなくても、self.psi4demand[w][3]でPを参照できる。
    return node_psi_dict_In4Dm


# ************************
# sankey
# ************************
def make_outbound_sankey_nodes_preorder(
    week, node, nodes_all, all_source, all_target, all_value_acc
):

    for child in node.children:

        # 子nodeが特定したタイミングで親nodeと一緒にセット

        # source = node(from)のnodes_allのindexで返す
        # target = child(to)のnodes_allのindexで返す
        # value  = S: psi4supply[w][0]を取り出す

        all_source[week].append(nodes_all.index(str(node.name)))
        all_target[week].append(nodes_all.index(str(child.name)))

        if len(child.psi4demand[week][3]) == 0:

            work = 0  # dummy link
            # work = 0.1 # dummy link

        else:

            # child.をvalueとする
            work = len(child.psi4supply[week][3])

        value_acc = child.psi4accume[week][3] = child.psi4accume[week - 1][3] + work

        # accを[]にして、tree nodes listに戻してからvalueをセットする
        all_value_acc[week].append(value_acc)  # これも同じ辞書+リスト構造に

        make_outbound_sankey_nodes_preorder(
            week, child, nodes_all, all_source, all_target, all_value_acc
        )

    return all_source, all_target, all_value_acc


def make_inbound_sankey_nodes_postorder(
    week, node, nodes_all, all_source, all_target, all_value_acc
):

    for child in node.children:

        make_inbound_sankey_nodes_postorder(
            week, child, nodes_all, all_source, all_target, all_value_acc
        )

        # 子nodeが特定したタイミングで親nodeと一緒にセット

        # source = node(from)のnodes_allのindexで返す
        # target = child(to)のnodes_allのindexで返す
        # value  = S: psi4supply[w][0]を取り出す

        # ***********************
        # source_target_reverse
        # ***********************
        all_target[week].append(nodes_all.index(str(node.name)))
        all_source[week].append(nodes_all.index(str(child.name)))

        # all_source[week].append( nodes_all.index( str(node.name)  ) )
        # all_target[week].append( nodes_all.index( str(child.name) ) )

        if len(child.psi4demand[week][3]) == 0:

            # pass
            work = 0  # ==0でもlinkが見えるようにdummyで与える
            # work = 0.1  # ==0でもlinkが見えるようにdummyで与える

        else:

            # inboundのvalueは、子node数で割ることで親の数字と合わせる
            work = len(child.psi4demand[week][3]) / len(node.children)

        # @230610
        value_acc = child.psi4accume[week][3] = child.psi4accume[week - 1][3] + work

        all_value_acc[week].append(value_acc)
        # all_value[week].append( work )

        # all_value[week].append( len( child.psi4demand[week][3] ) )

    return all_source, all_target, all_value_acc

    # ********************************
    # end2end supply chain accumed plan
    # ********************************


def visualise_e2e_supply_chain_plan(root_node_outbound, root_node_inbound):

    # ************************
    # sankey
    # ************************

    nodes_outbound = []
    nodes_inbound = []
    node_psI_list = []

    nodes_outbound, node_psI_list = extract_nodes_psI4demand(root_node_outbound)

    nodes_inbound, node_psI_list = extract_nodes_psI4demand_postorder(root_node_inbound)

    nodes_all = []
    nodes_all = nodes_inbound + nodes_outbound[1:]

    all_source = {}  # [0,1,1,0,2,3,3] #sourceは出発元のnode
    all_target = {}  # [2,2,3,3,4,4,5] #targetは到着先のnode
    all_value = {}  # [8,1,3,2,9,3,2] #値
    all_value_acc = {}  # [8,1,3,2,9,3,2] #値

    plan_range = root_node_outbound.plan_range

    for week in range(1, plan_range * 53):

        all_source[week] = []
        all_target[week] = []
        all_value[week] = []
        all_value_acc[week] = []

        all_source, all_target, all_value_acc = make_outbound_sankey_nodes_preorder(
            week, root_node_outbound, nodes_all, all_source, all_target, all_value_acc
        )

        all_source, all_target, all_value_acc = make_inbound_sankey_nodes_postorder(
            week, root_node_inbound, nodes_all, all_source, all_target, all_value_acc
        )

    # init setting week
    week = 50

    data = dict(
        type="sankey",
        arrangement="fixed",  # node fixing option
        node=dict(
            pad=100,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes_all,  # 各nodeを作成
            # color = ["blue", "blue", "green", "green", "yellow", "yellow"] #色を指定します。
        ),
        link=dict(
            source=all_source[week],  # [0,1,1,0,2,3,3], #sourceは出発元のnode
            target=all_target[week],  # [2,2,3,3,4,4,5], #targetは到着先のnode
            value=all_value_acc[week],  # [8,1,3,2,9,3,2]   #流量
        ),
    )

    layout = dict(title="global weekly supply chain Sankey Diagram", font=dict(size=10))

    # **********************
    # frames 2 animation
    # **********************

    # フレームを保存するリスト
    frames = []

    ## プロットを保存するリスト
    # data = []
    # x = np.linspace(0, 1, 53*self.plan_range)

    # プロットの作成
    # 0, 0.1, ... , 5までのプロットを作成する
    # for step in np.linspace(0, 5, 51):

    week_len = 53 * plan_range

    # for step in np.linspace(0, week_len, week_len+1):

    for week in range(40, 53 * plan_range):

        frame_data = dict(
            type="sankey",
            arrangement="fixed",  # node fixing option
            node=dict(
                pad=100,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes_all,  # 各nodeを作成
                ##color = ["blue", "blue", "green", "green", "yellow", "yellow"],
            ),
            link=dict(
                source=all_source[week],  # [0,1,1,0,2,3,3], #sourceは出発元のnode
                target=all_target[week],  # [2,2,3,3,4,4,5], #targetは到着先のnode
                value=all_value_acc[week],  # [8,1,3,2,9,3,2] #数量
            ),
        )

        frame_layout = dict(
            title="global weekly supply chain Week_No:" + str(week), font=dict(size=10)
        )

        frame = go.Frame(data=frame_data, layout=frame_layout)

        frames.append(frame)

        # ********************************
        # ココでpng出力
        # ********************************
        fig_temp = go.Figure(data=frame_data, layout=frame_layout)

        # ゼロ埋め
        # num = 12
        # f文字列：Python 3.6以降
        # s = f'{num:04}'  # 0埋めで4文字
        ##print(s)  # 0012

        zfill3_w = f"{week:03}"  # type is string

        temp_file_name = zfill3_w + ".png"

        pio.write_image(fig_temp, temp_file_name)  # write png

    fig = go.Figure(data=data, layout=layout, frames=frames)

    offline.plot(fig, filename="end2end_supply_chain_accumed_plan.html")


def map_psi_lots2df(node, psi_type, psi_lots):

    # preordering

    # psi4xxxx[w][0,1,2,3]で、node内のpsiをdfにcopy

    #    plan_len = 53 * node.plan_range
    #
    #    for w in range(1,plan_len):
    #
    #        s   = node.psi4demand[w][0]
    #        co  = node.psi4demand[w][1]
    #
    #        i0  = node.psi4demand[w-1][2]
    #        i1  = node.psi4demand[w][2]
    #
    #        p   = node.psi4demand[w][3]

    if psi_type == "demand":

        matrix = node.psi4demand

    elif psi_type == "supply":

        matrix = node.psi4supply

    else:

        print("error: wrong psi_type is defined")

    ## マッピングするデータのリスト
    #    psi_lots = []

    # マトリクスの各要素と位置をマッピング
    for week, row in enumerate(matrix):  # week

        for scoip, lots in enumerate(row):  # scoip

            for step_no, lot_id in enumerate(lots):

                psi_lots.append([node.name, week, scoip, step_no, lot_id])

    for child in node.children:

        map_psi_lots2df(child, psi_type, psi_lots)

    # DataFrameのカラム名
    # columns = ["step", "Element", "Position"]  # pos=(week,s-co-i-p)
    columns = ["node_name", "week", "s-co-i-p", "step_no", "lot_id"]

    # DataFrameの作成
    df = pd.DataFrame(psi_lots, columns=columns)

    return df


# *************************
# mapping psi tree2df    showing psi with plotly
# *************************
def show_psi_graph(root_node, D_S_flag, node_name, week_start, week_end):
    # def show_psi_graph(root_node_outbound,"demand","CAN_I",0,300):

    # show_psi_graph(
    #    root_node_outbound or root_node_inbound,  # out or in
    #    "demand"or "supply" ,                     # psi plan
    #    node_name,                                #"CAN_I" ,
    #    display_week_start,                       # 0 ,
    #    display_week_end,                         # 300 ,
    #    )

    # ********************************
    # map_psi_lots2df
    # ********************************

    # set_dataframe(root_node_outbound, root_node_inbound)

    if D_S_flag == "demand":

        psi_lots = []  # 空リストを持ってtreeの中に入る

        # tree中で、リストにpsiを入れ
        # DataFrameの作成して、dfを返している
        #     df = pd.DataFrame(psi_lots, columns=columns)

        df_demand_plan = map_psi_lots2df(root_node, D_S_flag, psi_lots)

    elif D_S_flag == "supply":

        psi_lots = []  # 空リストを持ってtreeの中に入る

        df_supply_plan = map_psi_lots2df(root_node, D_S_flag, psi_lots)

    else:

        print("error: combination  root_node==in/out  psi_plan==demand/supply")

    # **********************
    # select PSI
    # **********************

    if D_S_flag == "demand":

        df_init = df_demand_plan

    elif D_S_flag == "supply":

        df_init = df_supply_plan

    else:

        print("error: D_S_flag should be demand/sopply")

    # node指定
    node_show = node_name
    # node_show = "Platform"
    # node_show = "JPN"
    # node_show = "TrBJPN2HAM"
    # node_show = "HAM"
    # node_show = "MUC"
    # node_show = "MUC_D"
    # node_show = "SHA_D"
    # node_show = "SHA"
    # node_show = "CAN_I"

    ## 条件1: "node_name"の値がnode_show
    condition1 = df_init["node_name"] == node_show

    ## 条件2: "week"の値が50以上53以下
    # week_start, week_end

    condition2 = (df_init["week"] >= week_start) & (df_init["week"] <= week_end)
    # condition2 = (df_init["week"] >= 0) & (df_init["week"] <= 53 )
    # condition2 = (df_init["week"] >= 53) & (df_init["week"] <= 53+13 )
    # condition2 = (df_init["week"] >= 53) & (df_init["week"] <= 106)
    # condition2 = (df_init["week"] >= 0) & (df_init["week"] <= 300)

    ## 条件1のみでデータを抽出
    # df = df_init[condition1]

    ## 条件2のみでデータを抽出
    # df = df_init[condition2]

    ## 条件1と条件2のAND演算でデータを抽出
    df = df_init[condition1 & condition2]

    #    # 列名 "s-co-i-p" の値が 0 または 3 の行のみを抽出
    line_data_2I = df[df["s-co-i-p"].isin([2])]

    #    line_data_0 = df[df["s-co-i-p"].isin([0])]
    #    line_data_3 = df[df["s-co-i-p"].isin([3])]

    # 列名 "s-co-i-p" の値が 0 の行のみを抽出
    bar_data_0S = df[df["s-co-i-p"] == 0]

    # 列名 "s-co-i-p" の値が 3 の行のみを抽出
    bar_data_3P = df[df["s-co-i-p"] == 3]

    ## 列名 "s-co-i-p" の値が 2 の行のみを抽出
    # bar_data_2I = df[df["s-co-i-p"] == 2]

    # 折れ線グラフ用のデータを作成
    # 累積'cumsum'ではなく、'count'
    line_plot_data_2I = line_data_2I.groupby("week")["lot_id"].count()  ####.cumsum()

    #    line_plot_data_0 = line_data_0.groupby("week")["lot_id"].count().cumsum()
    #    line_plot_data_3 = line_data_3.groupby("week")["lot_id"].count().cumsum()

    # 積み上げ棒グラフ用のデータを作成
    bar_plot_data_3P = bar_data_3P.groupby("week")["lot_id"].count()
    bar_plot_data_0S = bar_data_0S.groupby("week")["lot_id"].count()

    # 積み上げ棒グラフのhovertemplate用のテキストデータを作成
    bar_hover_text_3P = (
        bar_data_3P.groupby("week")["lot_id"]
        .apply(lambda x: "<br>".join(x))
        .reset_index()
    )

    bar_hover_text_3P = bar_hover_text_3P["lot_id"].tolist()

    # 積み上げ棒グラフのhovertemplate用のテキストデータを作成
    bar_hover_text_0S = (
        bar_data_0S.groupby("week")["lot_id"]
        .apply(lambda x: "<br>".join(x))
        .reset_index()
    )
    bar_hover_text_0S = bar_hover_text_0S["lot_id"].tolist()

    # **************************
    # making graph
    # **************************
    # グラフの作成
    # fig = go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    #    # 折れ線グラフの追加
    #    fig.add_trace(go.Scatter(x=line_plot_data_0.index,
    #                             y=line_plot_data_0.values,
    #                             mode='lines', name='Cumulative Count 0 S'),
    #        secondary_y=False )
    #
    #    fig.add_trace(go.Scatter(x=line_plot_data_3.index,
    #                             y=line_plot_data_3.values,
    #                             mode='lines', name='Cumulative Count 3 P'),
    #        secondary_y=False )

    # 積み上げ棒グラフの追加

    fig.add_trace(
        go.Bar(
            x=bar_plot_data_3P.index,
            y=bar_plot_data_3P.values,
            name="node 3_P: " + node_show,
            # name='Individual Count'+"3_P",
            text=bar_hover_text_3P,
            texttemplate="%{text}",
            textposition="inside",
            hovertemplate="Lot ID: %{x}<br>Count: %{y}",
        ),
        # hovertemplate='Lot ID: %{x}<br>Count: %{y}')
        # )
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=bar_plot_data_0S.index,
            y=bar_plot_data_0S.values,
            name="node 0_S: " + node_show,
            # name='Individual Count'+"0_S",
            text=bar_hover_text_0S,
            texttemplate="%{text}",
            textposition="inside",
            hovertemplate="Lot ID: %{x}<br>Count: %{y}",
        ),
        # hovertemplate='Lot ID: %{x}<br>Count: %{y}')
        # )
        secondary_y=False,
    )

    # 折れ線グラフの追加
    fig.add_trace(
        go.Scatter(
            x=line_plot_data_2I.index,
            y=line_plot_data_2I.values,
            mode="lines",
            name="node 2_I: " + node_show,
        ),
        # name='Inventory 2I'),
        secondary_y=True,
    )

    # 軸ラベルの設定
    fig.update_xaxes(title_text="week")
    fig.update_yaxes(title_text="I by lots", secondary_y=True)

    # グラフの表示
    fig.show()


# *******************
# 生産平準化の前処理　ロット・カウント
# *******************
def count_lots_yyyy(psi_list, yyyy_str):

    matrix = psi_list

    # 共通の文字列をカウントするための変数を初期化
    count_common_string = 0

    # Step 1: マトリクス内の各要素の文字列をループで調べる
    for row in matrix:

        for element in row:

            # Step 2: 各要素内の文字列が "2023" を含むかどうかを判定
            if yyyy_str in element:

                # Step 3: 含む場合はカウンターを増やす
                count_common_string += 1

    return count_common_string


def is_52_or_53_week_year(year):
    # 指定された年の12月31日を取得
    last_day_of_year = datetime.date(year, 12, 31)

    # 12月31日のISO週番号を取得 (isocalendar()メソッドはタプルで[ISO年, ISO週番号, ISO曜日]を返す)
    _, iso_week, _ = last_day_of_year.isocalendar()

    # ISO週番号が1の場合は前年の最後の週なので、52週と判定
    if iso_week == 1:
        return 52
    else:
        return iso_week


def find_depth(node):
    if not node.parent:
        return 0
    else:
        return find_depth(node.parent) + 1


def find_all_leaves(node, leaves, depth=0):
    if not node.children:
        leaves.append((node, depth))  # (leafノード, 深さ) のタプルを追加
    else:
        for child in node.children:
            find_all_leaves(child, leaves, depth + 1)


def make_nodes_decouple_all(node):
    # def main():
    #
    #    root_node = build_tree()
    #    set_parent(root_node)

    #    leaves = []
    #    find_all_leaves(root_node, leaves)
    #    pickup_list = leaves[::-1]  # 階層の深い順に並べる

    leaves = []
    leaves_name = []

    nodes_decouple = []

    find_all_leaves(node, leaves)
    # find_all_leaves(root_node, leaves)
    pickup_list = sorted(leaves, key=lambda x: x[1], reverse=True)
    pickup_list = [leaf[0] for leaf in pickup_list]  # 深さ情報を取り除く

    # こうすることで、leaf nodeを階層の深い順に並べ替えた pickup_list が得られます。
    # 先に深さ情報を含めて並べ替え、最後に深さ情報を取り除くという流れになります。

    # 初期処理として、pickup_listをnodes_decoupleにcopy
    # pickup_listは使いまわしで、pop / insert or append / removeを繰り返す
    for nd in pickup_list:
        nodes_decouple.append(nd.name)

    nodes_decouple_all = []

    while len(pickup_list) > 0:

        # listのcopyを要素として追加
        nodes_decouple_all.append(nodes_decouple.copy())

        current_node = pickup_list.pop(0)
        del nodes_decouple[0]  # 並走するnode.nameの処理

        parent_node = current_node.parent

        if parent_node is None:
            break

        # 親ノードをpick up対象としてpickup_listに追加
        if current_node.parent:

            #    pickup_list.append(current_node.parent)
            #    nodes_decouple.append(current_node.parent.name)

            # if parent_node not in pickup_list:  # 重複追加を防ぐ

            # 親ノードの深さを見て、ソート順にpickup_listに追加
            depth = find_depth(parent_node)
            inserted = False

            for idx, node in enumerate(pickup_list):

                if find_depth(node) <= depth:

                    pickup_list.insert(idx, parent_node)
                    nodes_decouple.insert(idx, parent_node.name)

                    inserted = True
                    break

            if not inserted:
                pickup_list.append(parent_node)
                nodes_decouple.append(parent_node.name)

            # 親ノードから見た子ノードをpickup_listから削除
            for child in parent_node.children:

                if child in pickup_list:

                    pickup_list.remove(child)
                    nodes_decouple.remove(child.name)

        else:

            print("error: node dupplicated", parent_node.name)

    return nodes_decouple_all


def evaluate_inventory_all(node, total_I, node_eval_I, nodes_decouple):

    total_I_work = []

    # if node.name in nodes_decouple:

    # デカップル拠点nodeの在庫psi4xxx[w][2]をworkにappendする
    for w in range(node.plan_range * 53):
        total_I_work.append(len(node.psi4supply[w][2]))

    node_eval_I[node.name] = total_I_work

    # node.decoupling_total_I.extend( total_I_work )
    ##node.decoupling_total_I = total_I_work
    #
    # node_eval_I[node.name] = node.decoupling_total_I

    total_I += sum(total_I_work)  # sumをとる

    # デカップル拠点nodeのmax在庫をとる
    # max_I = max( max_I, max(total_I_work) ) # maxをとる

    # デカップル拠点nodeのmax在庫の累計をとる
    # total_I += max(total_I_work)

    # else:
    #
    #    pass

    if node.children == []:

        pass

    else:

        for child in node.children:

            total_I, node_eval_I = evaluate_inventory_all(
                child, total_I, node_eval_I, nodes_decouple
            )

    return total_I, node_eval_I


def evaluate_inventory(node, total_I, node_eval_I, nodes_decouple):

    total_I_work = []

    if node.name in nodes_decouple:

        # デカップル拠点nodeの在庫psi4xxx[w][2]をworkにappendする
        for w in range(node.plan_range * 53):
            total_I_work.append(len(node.psi4supply[w][2]))
            # total_I_work +=  len( node.psi4supply[w][2] )

        node_eval_I[node.name] = total_I_work

        # node.decoupling_total_I.extend( total_I_work )
        ##node.decoupling_total_I = total_I_work
        #
        # node_eval_I[node.name] = node.decoupling_total_I

        # デカップル拠点nodeのmax在庫の累計をとる
        total_I += max(total_I_work)
        # total_I = max( total_I, max(total_I_work) )

    else:

        pass

    if node.children == []:

        pass

    else:

        for child in node.children:

            total_I, node_eval_I = evaluate_inventory(
                child, total_I, node_eval_I, nodes_decouple
            )

    return total_I, node_eval_I


def show_subplots_set_y_axies(node_eval_I, nodes_decouple):

    nodes_decouple_text = ""

    for node_name in nodes_decouple:
        work_text = node_name + " "
        nodes_decouple_text += work_text

    # 各グラフのy軸の最大値を計算
    max_value = max(max(values) for values in node_eval_I.values())

    # サブプロットを作成
    fig = make_subplots(
        rows=len(node_eval_I),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f"{key} (Max: {max(values)}, Sum: {sum(values)})"
            for key, values in node_eval_I.items()
        ],
    )

    # 各データをプロット
    row = 1
    for key, values in node_eval_I.items():

        max_sum_text = key + " max=" + str(max(values)) + " sum=" + str(sum(values))

        trace = go.Scatter(
            x=list(range(len(values))),
            y=values,
            fill="tozeroy",
            mode="none",
            name=max_sum_text,
        )

        # trace = go.Scatter(x=list(range(len(values))), y=values, fill='tozeroy', mode='none',  name = key )

        fig.add_trace(trace, row=row, col=1)
        row += 1

    # グラフのy軸の範囲を設定
    for i in range(1, len(node_eval_I) + 1):
        fig.update_yaxes(range=[0, max_value], row=i, col=1)

    # グラフレイアウトを設定
    fig.update_layout(
        title="デカップリング・ポイントの在庫推移" + nodes_decouple_text,
        # title='デカップリング・ポイントの在庫推移',
        xaxis_title="Week",
        yaxis_title="Lots",
        showlegend=False,  # 凡例を非表示
    )

    # グラフを表示
    fig.show()


def show_subplots_bar_decouple(node_eval_I, nodes_decouple):

    nodes_decouple_text = ""
    for node_name in nodes_decouple:
        work_text = node_name + " "
        nodes_decouple_text += work_text

    fig = make_subplots(
        rows=len(node_eval_I),
        cols=1,
        shared_xaxes=True,
        subplot_titles=list(node_eval_I.keys()),
    )

    row = 1
    for key, values in node_eval_I.items():
        trace = go.Scatter(
            x=list(range(len(values))), y=values, fill="tozeroy", mode="lines", name=key
        )
        fig.add_trace(trace, row=row, col=1)
        row += 1

    fig.update_layout(
        title="デカップリング・ポイントの在庫推移" + nodes_decouple_text,
        xaxis_title="week",
        yaxis_title="lots",
    )

    fig.show()


def show_subplots_bar(node_eval_I):

    fig = make_subplots(
        rows=len(node_eval_I),
        cols=1,
        shared_xaxes=True,
        subplot_titles=list(node_eval_I.keys()),
    )

    row = 1
    for key, values in node_eval_I.items():
        trace = go.Scatter(
            x=list(range(len(values))), y=values, fill="tozeroy", mode="lines", name=key
        )
        fig.add_trace(trace, row=row, col=1)
        row += 1

    fig.update_layout(
        title="デカップリング・ポイントの在庫推移",
        xaxis_title="week",
        yaxis_title="lots",
    )

    fig.show()


# A = {
#    'NodeA': [10, 15, 8, 12, 20],
#    'NodeB': [5, 8, 6, 10, 12],
#    'NodeC': [2, 5, 3, 6, 8]
# }
#
# show_subplots_bar(A)


def show_node_eval_I(node_eval_I):

    ## サンプルの辞書A（キーがノード名、値が時系列データのリストと仮定）
    # A = {
    #    'NodeA': [10, 15, 8, 12, 20],
    #    'NodeB': [5, 8, 6, 10, 12],
    #    'NodeC': [2, 5, 3, 6, 8]
    # }

    # グラフ描画
    fig = px.line()
    for key, values in node_eval_I.items():
        fig.add_scatter(x=list(range(len(values))), y=values, mode="lines", name=key)

    fig.update_layout(
        title="デカップリング・ポイントの在庫推移",
        xaxis_title="week",
        yaxis_title="lots",
    )

    fig.show()


# *******************************************
# 流動曲線で表示　show_flow_curve
# *******************************************


def show_flow_curve(df_init, node_show):

    # 条件1: "node_name"の値がnode_show
    condition1 = df_init["node_name"] == node_show

    ## 条件2: "week"の値が50以上53以下
    # condition2 = (df_init["week"] >= 0) & (df_init["week"] <= 53 )
    # condition2 = (df_init["week"] >= 53) & (df_init["week"] <= 53+13 )
    # condition2 = (df_init["week"] >= 53) & (df_init["week"] <= 106)

    # 条件1のみでデータを抽出
    df = df_init[condition1]

    ## 条件1と条件2のAND演算でデータを抽出
    # df = df_init[condition1 & condition2]

    # df_init = df_init[condition1 & condition2]
    # df_init = df_init[df_init['node_name']==node_show]

    # グループ化して小計"count"の計算
    df = df.groupby(["node_name", "week", "s-co-i-p"]).size().reset_index(name="count")

    # 累積値"count_accum"の計算
    df["count_accum"] = df.groupby(["node_name", "s-co-i-p"])["count"].cumsum()

    # 折れ線グラフの作成
    line_df_0 = df[df["s-co-i-p"].isin([0])]
    # s-co-i-pの値が0の行を抽出

    # 折れ線グラフの作成
    line_df_3 = df[df["s-co-i-p"].isin([3])]
    # s-co-i-pの値が3の行を抽出

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=line_df_0["week"],
            y=line_df_0["count_accum"],
            mode="lines",
            name="Demand S " + node_show,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=line_df_3["week"],
            y=line_df_3["count_accum"],
            mode="lines",
            name="Supply P " + node_show,
        ),
        secondary_y=False,
    )

    # 棒グラフの作成
    bar_df = df[df["s-co-i-p"] == 2]  # s-co-i-pの値が2の行を抽出

    fig.add_trace(
        go.Bar(x=bar_df["week"], y=bar_df["count"], name="Inventory "),
        # go.Bar(x=bar_df['week'], y=bar_df['step_no'], name='棒グラフ'),
        secondary_y=True,
    )

    # 軸ラベルの設定
    fig.update_xaxes(title_text="week")
    fig.update_yaxes(title_text="S and P", secondary_y=False)
    fig.update_yaxes(title_text="count_accum", secondary_y=True)
    # fig.update_yaxes(title_text='step_no', secondary_y=True)

    # グラフの表示
    fig.show()


# *******************************************
# tree handling parts
# *******************************************


def print_tree_bfs(root):
    queue = deque([(root, 0)])

    while queue:
        node, depth = queue.popleft()
        print("  " * depth + node.name)
        queue.extend((child, depth + 1) for child in node.children)


def print_tree_dfs(node, depth=0):
    print("  " * depth + node.name)
    for child in node.children:
        print_tree_dfs(child, depth + 1)


# *******************************************
# extract_CPU_tree_preorder          PREORDER / OUTBOUND
# *******************************************
def extract_CPU_tree_preorder(node, csv_writer):

    print("extracting  " + node.name)

    node.extract_CPU(csv_writer)

    if node.children == []:  # 子nodeがないleaf nodeの場合

        pass

    else:

        for child in node.children:

            extract_CPU_tree_preorder(child, csv_writer)


def feedback_psi_lists(node, node_psi_dict, nodes):

    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。

    if node.children == []:  # 子nodeがないleaf nodeの場合

        pass

    else:

        # ************************************
        # clearing children P[w][3] and S[w][0]
        # ************************************
        # replace lotするために、事前に、
        # 出荷先となるすべてのchildren nodesのS[w][0]とP[w][3]をクリア

        for child in node.children:

            for w in range(53 * node.plan_range):

                child.psi4supply[w][0] = []
                child.psi4supply[w][3] = []

        # lotidから、leaf_nodeを特定し、出荷先ship2nodeに出荷することは、
        # すべての子nodeに出荷することになる

        # ************************************
        # setting mother_confirmed_S
        # ************************************
        # このnode内での子nodeへの展開
        for w in range(53 * node.plan_range):

            confirmed_S_lots = node.psi4supply[w][0]  # 親の確定出荷confS lot

            # 出荷先nodeを特定して

            # 一般には、下記のLT shiftだが・・・・・
            # 出荷先の ETA = LT_shift(ETD) でP place_lot
            # 工程中の ETA = SS_shift(ETD) でS place_lot

            # 本モデルでは、輸送工程 = modal_nodeを想定して・・・・・
            # 出荷先の ETA = 出荷元ETD        でP place_lot
            # 工程中の ETA = LT&SS_shift(ETD) でS place_lot
            # というイビツなモデル定義・・・・・

            # 直感的なPO=INVOICEという考え方に戻すべきかも・・・・・
            #
            # modal shiftのmodelingをLT_shiftとの拡張で考える???
            # modal = BOAT/AIR/QURIE
            # LT_shift(modal, w, ,,,,

            for lot in confirmed_S_lots:

                if lot == []:

                    pass

                else:

                    # *********************************************************
                    # child#ship2node = find_node_to_ship(node, lot)
                    # lotidからleaf_nodeのpointerを返す
                    leaf_node_name = extract_node_name(lot)

                    leaf_node = nodes[leaf_node_name]

                    # 末端からあるnodeAまでleaf_nodeまでのnode_listをpathで返す

                    current_path = []
                    path = []

                    path = find_path_to_leaf_with_parent(node, leaf_node, current_path)

                    # nodes_listを逆にひっくり返す
                    path.reverse()

                    # 出荷先nodeはnodeAの次node、path[1]になる
                    ship2node_name = path[1]

                    ship2node = nodes[ship2node_name]

                    # ここでsupply planを更新している
                    # 出荷先nodeのPSIのPとSに、confirmed_S中のlotをby lotで置く
                    place_P_in_supply(w, ship2node, lot)

    for child in node.children:

        feedback_psi_lists(child, node_psi_dict, nodes)


# *******************************************
# make node_name_list for OUTBOUND with POST-ORDER
# *******************************************
def make_node_post_order(node, node_seq_list):

    if node.children == []:  # 子nodeがないleaf nodeの場合

        pass

    else:

        for child in node.children:

            make_node_post_order(child, node_seq_list)

    node_seq_list.append(node.name)

    return node_seq_list


# *******************************************
# make node_name_list for INBOUND with PRE-ORDER
# *******************************************
def make_node_pre_order(node, node_seq_list):

    node_seq_list.append(node.name)

    if node.children == []:  # 子nodeがないleaf nodeの場合

        pass

    else:

        for child in node.children:

            make_node_pre_order(child, node_seq_list)

    return node_seq_list


def make_lot(node_sequence, nodes, lot_ID_list):

    # def make_lot_post_order(node_sequence, nodes, lot_ID_list_out,lot_ID_names_out):

    print("lot_ID_list ", lot_ID_list)

    for pos, node_name in enumerate(node_sequence):

        node = nodes[node_name]

        print("pos", pos)

        print("node_sequence[pos]", node_sequence[pos])

        print("node.psi4supply", node.psi4supply)

        print("lot_ID_list[pos][0][:]", lot_ID_list[pos][0][:])
        # print('lot_ID_list[:][0][pos]',lot_ID_list[:][0][pos])

        # @240429
        #  node.psi4supply[week][PSI][lot_list]

        list_B = []

        for w in range(53 * node.plan_range):

            # list_B.append( node.psi4supply[w][0] ) # Sをweek分append

            # @240429 memo イメージはコレだが、これではpointerのcopy???
            # lot_ID_list[pos][0][w] = node.psi4supply[w][0][:]

            # posは、node_sequence[pos]で示すnode
            # lot_ID_listのデータ・イメージ
            # lot_ID_list[nodeのpos][P-CO-S-I][w0-w53*plan_range]

            lot_ID_list[pos][0].insert(w, node.psi4supply[w][0])
            lot_ID_list[pos][1].insert(w, node.psi4supply[w][1])
            lot_ID_list[pos][2].insert(w, node.psi4supply[w][2])
            lot_ID_list[pos][3].insert(w, node.psi4supply[w][3])

            # 見本 w週がindexとなり、lot_listをinsertする
            ## インデックス w の位置に B の要素を挿入
            # A.insert(w, B)

    print("lot_ID_list LOT_list形式", lot_ID_list)

    return lot_ID_list


## *******************************************
## make lot_ID_list for INBOUND with PRE-ORDER
## *******************************************

#    lot_ID_list_out = make_lot_post_order(root_node_outbound, lot_ID_list_out, lot_ID_names)

#    lot_ID_list_in  = make_lot_pre_order(root_node_inbound , lot_ID_list_in, lot_ID_names )


# *******************************************
# lots visualise on 3D plot
# *******************************************


def show_lots_status_in_nodes_PSI_W318(
    node_sequence, PSI_locations, PSI_location_colors, weeks, lot_ID
):

    # Create a 3D plot using plotly
    fig = go.Figure()

    # Add traces for each node and location

    #@240526 define as figure clearly
    location_index = 0

    for node_index, node in enumerate(node_sequence):

        for location_index, location in enumerate(PSI_locations):

            fig.add_trace(
                go.Scatter3d(
                    x=[node_index * len(PSI_locations) + location_index] * len(weeks),
                    y=weeks,
                    z=lot_ID[node_index, location_index],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=PSI_location_colors[
                            location
                        ],  # set color to location color
                        opacity=0.8,
                    ),
                    name=f"{node} {location}",
                )
            )

    # Create and add slider
    steps = []
    for i in range(len(weeks)):
        step = dict(
            method="update",
            args=[
                {
                    "visible": [False]
                    * len(weeks)
                    * len(node_sequence)
                    * len(PSI_locations)
                }
            ],
            label=f"Week {i}",
        )
        for j in range(len(node_sequence) * len(PSI_locations)):
            step["args"][0]["visible"][i + j * len(weeks)] = True
            # Toggle i-th trace to "visible"
        steps.append(step)

    sliders = [
        dict(active=0, currentvalue={"prefix": "Week: "}, pad={"t": 50}, steps=steps)
    ]

    fig.update_layout(
        sliders=sliders,
        scene=dict(
            xaxis=dict(
                title="Node Location",
                tickvals=list(range(len(node_sequence))),
                ticktext=node_sequence,
            ),
            yaxis=dict(title="Weeks"),
            zaxis=dict(title="Lot ID"),
        ),
        title="3D Plot Graph with Time Series Slider",
    )

    # Show the plot
    fig.show()




# *******************************************
# lots visualise on 3D plot
# *******************************************

#@240526 stopping
#def show_lots_status_in_nodes_PSI_W318_list(
#    node_sequence, PSI_locations, PSI_location_colors, weeks, lot_ID_list):
#
#    # Create a 3D plot using plotly
#    fig = go.Figure()
#
#    # Add traces for each node and location
#
#    ## Define lot_ID_list (replace with actual data)
#    # lot_ID_list = [
#    #    [['LotA', 'LotB'], [], ['LotC'], ['LotD']],
#    #    [['LotE'], [], ['LotF', 'LotG'], ['LotH']],
#    #    [['LotI'], [], ['LotJ'], ['LotK', 'LotL']],
#    #    [['LotM', 'LotN', 'LotO'], [], ['LotP'], []],
#    #    [['LotQ', 'LotR'], [], ['LotS'], ['LotT']]
#    # ]
#    #
#    ## Define the nodes and their locations
#    # node_sequence = ['nodeA', 'nodeB', 'nodeC', 'nodeD', 'nodeE']
#    # PSI_locations = ['Sales', 'Carry Over', 'Inventory', 'Purchase']
#
#    # ********************************
#    # by GPT
#    # ********************************
#    # Initialize an empty 2D array for z values
#    z_val_matrix = np.empty((len(node_sequence), len(PSI_locations)), dtype=object)
#
#    #    # Populate z values based on lot_ID_list
#    #    for node_index, node in enumerate(node_sequence):
#    #
#    #        for psi_index, psi_location in enumerate(PSI_locations):
#    #
#    #            lot_ID_lots = lot_ID_list[node_index][psi_index]
#    #
#    #
#
#    # ********************************
#    # STOP もう一つ下でlot_list中のpositionを見る
#    # ********************************
#    # z_val_matrix[node_index, location_index] = len(lot_ID_lots) + 1
#
#    # Now you can use z_val_matrix in your 3D plot
#    # ...
#    # (Your existing code for creating the 3D plot goes here)
#    # ...
#
#    # Don't forget to add hovertext using lot_ID_names[node_index][location_index]
#    # ...
#    # (Add hovertext to your existing code)
#    # ...
#
#    for node_index, node in enumerate(node_sequence):
#
#        for psi_index, psi_location in enumerate(PSI_locations):
#
#            # 週内のlot_IDのリスト
#            # lot_ID_lots = lot_ID_list[node_index, location_index] # np.array
#
#            lot_ID_lots = lot_ID_list[node_index][location_index]  # list
#
#            for pos, lot in enumerate(lot_ID_lots):  # リスト位置が高さ
#
#                x_val = [node_index * len(PSI_locations) + psi_index] * len(weeks)
#                y_val = (weeks,)
#
#                # listの長さではなく、list中の位置=pos+1をセット
#                z_val_matrix[node_index, psi_index] = pos + 1
#
#                # STOP
#                # z_val_matrix[node_index, location_index] =len(lot_ID_lots)+1 #
#                #z_val = [pos], # listで渡す???
#
#                # print('3D plot X Y Z',x_val,y_val,z_val)
#
#                fig.add_trace(
#                    go.Scatter3d(
#                        x=x_val,
#                        # x=[node_index * len(PSI_locations) + psi_index] * len(weeks),
#                        y=y_val,
#                        # y=weeks,
#                        z=z_val_matrix,
#                        # z = [pos], # listで渡す???
#                        # z = pos, #数値ではNG
#                        # z=lot_ID[node_index, location_index],
#                        mode="markers",
#                        marker=dict(
#                            size=5,
#                            color=PSI_location_colors[
#                                psi_location
#                            ],  # set color to location color
#                            opacity=0.8,
#                        ),
#                        # name=f'{node} {location}{lot}'
#                        name=f"{node} {location}",
#                    )
#                )
#
#    # Create and add slider
#    steps = []
#    for i in range(len(weeks)):
#        step = dict(
#            method="update",
#            args=[
#                {
#                    "visible": [False]
#                    * len(weeks)
#                    * len(node_sequence)
#                    * len(PSI_locations)
#                }
#            ],
#            label=f"Week {i}",
#        )
#        for j in range(len(node_sequence) * len(PSI_locations)):
#            step["args"][0]["visible"][i + j * len(weeks)] = True
#            # Toggle i-th trace to "visible"
#        steps.append(step)
#
#    sliders = [
#        dict(active=0, currentvalue={"prefix": "Week: "}, pad={"t": 50}, steps=steps)
#    ]
#
#    fig.update_layout(
#        sliders=sliders,
#        scene=dict(
#            xaxis=dict(
#                title="Node Location",
#                tickvals=list(range(len(node_sequence))),
#                ticktext=node_sequence,
#            ),
#            yaxis=dict(title="Weeks"),
#            zaxis=dict(title="Lot ID"),
#        ),
#        title="3D Plot Graph with Time Series Slider",
#    )
#
#    # Show the plot
#    fig.show()


# *******************************************
# location_sequence[] = PSI 0/1/2/3 X node
# *******************************************
def make_loc_seq(node_sequence, psi_seq):

    location_sequence = []

    for node in node_sequence:

        for i, psi in enumerate(psi_seq):
            # 0,1,2,3 == S, CO, I, P

            locat = str(i) + node  # "0"+node

            location_sequence.append(locat)

    return location_sequence


# *******************************************
# make_loc_dic
# *******************************************
def make_loc_dic(location_sequence, nodes):

    location_dic = {}  ####dictionary

    for loc in location_sequence:

        node_name = loc[1:]

        node = nodes[node_name]

        # location_dic[node_name] = node.eval_revenue

        # location_dic[node_name] = "REVENUE" + str(node.eval_revenue)

        location_dic[loc] = [
            node.eval_revenue,
            node.eval_profit,
            node.eval_profit_ratio,
        ]

    return location_dic


# *******************************************
# make_node_dic
# *******************************************
def make_node_dic(location_sequence, nodes):

    node_dic = {}  ####dictionary

    for loc in location_sequence:

        node_name = loc[1:]

        node = nodes[node_name]

        node_dic[node_name] = node.eval_revenue

    return node_dic


# *******************************************
# write_CPU_lot2list
# *******************************************
def make_CPU_lot2list(node, CPU_lot_list):

    # pre_ordering search

    # targets are "lot_ID", "step" in node.psi4supply

    for week in range(53 * node.plan_range):

        for psi in range(0, 4):  # 0:Sales 1:Carry Over 2:Inventory 3:Purchase

            CPU_lots = node.psi4supply[week][psi]

            if CPU_lots == []:

                pass

            else:

                for step, lot_ID in enumerate(CPU_lots):

                    # flatなCPU_lot_listにセット
                    cpu_lot_row = []  # rowの初期化

                    # a target "data layout "image of df

                    # df4PSI_visualize
                    # product, node_PSI_location, node_name, PSI_name, week, lot_ID, step

                    #
                    # an image of sample data4 PSI_visualize
                    # "AAAAAAA", "0"+"HAM_N", "HAM_N", "Sales", "W26", 'HAM_N2024390', 2

                    ####cpu_lot_row.append(node.product_name)

                    node_PSI_location = str(psi) + node.name  # PSI+node X-axis
                    cpu_lot_row.append(node_PSI_location)  # PSI+node

                    cpu_lot_row.append(node.name)

                    cpu_lot_row.append(psi)

                    cpu_lot_row.append(week)  # week Y-axis
                    cpu_lot_row.append(lot_ID)
                    cpu_lot_row.append(step)  # step Z-axis

                    # *******************
                    # add row to list
                    # *******************
                    CPU_lot_list.append(cpu_lot_row)

    if node.children == []:  # leaf node

        pass

    else:

        for child in node.children:

            make_CPU_lot2list(child, CPU_lot_list)

    # end

    return CPU_lot_list


#                    write row to
#
#            for lot in CPU_lots:
#
#            lot_ID_list[pos][0].insert(w, node.psi4supply[w][0])
#            lot_ID_list[pos][1].insert(w, node.psi4supply[w][1])
#            lot_ID_list[pos][2].insert(w, node.psi4supply[w][2])
#            lot_ID_list[pos][3].insert(w, node.psi4supply[w][3])


# *******************************************
# 頭一桁の数字を取って整数値に変換
# *******************************************
def extract_first_digit(locat):

    return int(locat[0])


# *******************************************
# lots visualise on 3D plot
# *******************************************
def show_lots_status_in_nodes_PSI_list_matrix(
    node_sequence, PSI_locations, PSI_location_colors, weeks, lot_ID_list
):

    # Create a 3D plot using plotly
    fig = go.Figure()

    # Add traces for each node and location

    # *********************************
    # z_val_matrixの中に積み上げロットの高さをセットする
    # *********************************
    #                # listの長さではなく、list中の位置=pos+1をセット
    #                z_val_matrix[locat_index, week] = hight_pos + 1

    location_sequence = []
    psi_seq = [0, 1, 2, 3]

    location_sequence = make_loc_seq(node_sequence, psi_seq)
    # "0"+node_A, "1"+node_A, "2"+node_A, "3"+node_A

    print("location_sequence", location_sequence)

    # ********************************
    # by GPT
    # ********************************
    # Initialize an empty 2D array for z values

    # X軸はlocation = node+psi Y軸は week

    # Z軸はロットの高さ
    max_lot_count = 100

    z_val_matrix = np.empty(
        (len(location_sequence), max_lot_count, len(weeks)), dtype=object
    )

    # z_val_matrix = np.empty((len(node_sequence)*len(PSI_locations), len(weeks)), dtype=object)

    # "z_val_matrix = np.empty((len(node_sequence), len(PSI_locations)), dtype=object)

    #    # Populate z values based on lot_ID_list
    #    for node_index, node in enumerate(node_sequence):
    #
    #        for psi_index, psi_location in enumerate(PSI_locations):
    #
    #            lot_ID_lots = lot_ID_list[node_index][location_index]
    #
    #

    for locat_index, locat in enumerate(location_sequence):
        # for node_index, node in enumerate(node_sequence):

        # node_indexは、len(location_sequence)をlen(PSI_locations)=4で割った商

        # list位置の計算なので、len(list)-1で計算
        node_index = (len(location_sequence) - 1) // len(PSI_locations)

        # "0"+node 頭1桁がpsi 0/1/2/3なので、psi_index = eval("0")

        psi_index = extract_first_digit(locat)
        # location_index = extract_first_digit( locat )

        # STOP
        # node_index, location_index = trans_node(locat,node_sequence, PSI_locations)

        # locat = "0"+node 頭1桁がpsi 0/1/2/3 + node_name

        # STOP
        # node_sequenceの中のnodeを見て、position=node_indexを返す

        # STOP
        # location_sequenceの中のpsiを見て、location_indexを返す

        for week in weeks:

            # 週内のlot_IDのリスト
            # lot_ID_lots = lot_ID_list[node_index, location_index] # np.array

            # print('node_index, location_index',node_index, location_index)

            lot_ID_lots = lot_ID_list[node_index][psi_index][week]  # list

            for hight_pos, lot in enumerate(lot_ID_lots):  # リスト位置が高さ

                # PSIの色辞書引のためのpsi名、salesなどをここではlocationと定義
                psi_location = PSI_locations[psi_index]

                x_val = [locat_index] * len(weeks)
                # x_val=[node_index *len(PSI_locations) + psi_index] *len(weeks)

                y_val = (weeks,)

                # listの長さではなく、list中の位置=posに1をセット
                z_val_matrix[locat_index, hight_pos, week] = 1
                # z_val_matrix[locat_index, hight_pos, week] = hight_pos + 1

    # *****************************************************************
    # STOP ココまでを流して、z_val_matrixを作り終える
    # *****************************************************************

    # *****************************************************************
    # 最初のloopの中で、z_val_matrixをセットしてから、直ぐにグラフ化する
    # 3D plotする時には、z=lot_ID[locat_index, week], で参照するのみ
    #
    # *****************************************************************

    for locat_index, locat in enumerate(location_sequence):
        # for node_index, node in enumerate(node_sequence):

        # node_indexは、len(location_sequence)をlen(PSI_locations)=4で割った商

        # list位置の計算なので、len(list)-1で計算
        node_index = (len(location_sequence) - 1) // len(PSI_locations)

        # "0"+node 頭1桁がpsi 0/1/2/3なので、psi_index = eval("0")

        psi_index = extract_first_digit(locat)
        # location_index = extract_first_digit( locat )

        for pos in range(0, max_lot_count):
            # for week in weeks:

            #            # 週内のlot_IDのリスト
            #            #lot_ID_lots = lot_ID_list[node_index, location_index] # np.array
            #
            #            #print('node_index, location_index',node_index, location_index)
            #
            #            lot_ID_lots = lot_ID_list[node_index][psi_index][week] # list
            #
            #
            #
            #            for hight_pos, lot in enumerate(lot_ID_lots): # リスト位置が高さ
            #
            #

            # PSIの色辞書引のためのpsi名、salesなどをここではlocationと定義
            psi_location = PSI_locations[psi_index]

            #            x_val=[ locat_index ] * len(weeks)
            #            #x_val=[node_index *len(PSI_locations) + psi_index] *len(weeks)
            #
            #            y_val=weeks,

            #            # listの長さではなく、list中の位置=pos+1をセット
            #            z_val=z_val_matrix[locat_index, week],

            z_val = z_val_matrix[locat_index, pos]

            print("z_val", z_val)

            #            if z_val == 0:
            #
            #                pass
            #
            #            else:

            fig.add_trace(
                go.Scatter3d(
                    x=[locat_index] * len(weeks),
                    # x = x_val,
                    # x=[node_index * len(PSI_locations) + location_index] * len(weeks),
                    # y = y_val,
                    y=weeks,
                    z=z_val_matrix[locat_index, pos],
                    # z = z_val_matrix[locat_index, week],
                    # z = [pos], # listで渡す???
                    # z = pos, #数値ではNG
                    # z=lot_ID[node_index, location_index],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=PSI_location_colors[
                            psi_location
                        ],  # set color to location color
                        opacity=0.8,
                    ),
                    # name=f'{node} {psi_location}{lot}'
                    # name=f'{node} {psi_location}'
                    name=f"{locat} {lot}",  # locatは、psi"0/1/2/3"+node_name
                )
            )

    # Create and add slider
    steps = []
    for i in range(len(weeks)):
        step = dict(
            method="update",
            args=[
                {
                    "visible": [False]
                    * len(weeks)
                    * len(node_sequence)
                    * len(PSI_locations)
                }
            ],
            label=f"Week {i}",
        )
        for j in range(len(node_sequence) * len(PSI_locations)):
            step["args"][0]["visible"][i + j * len(weeks)] = True
            # Toggle i-th trace to "visible"
        steps.append(step)

    sliders = [
        dict(active=0, currentvalue={"prefix": "Week: "}, pad={"t": 50}, steps=steps)
    ]

    fig.update_layout(
        sliders=sliders,
        scene=dict(
            xaxis=dict(
                title="Node Location",
                tickvals=list(range(len(node_sequence))),
                ticktext=node_sequence,
            ),
            yaxis=dict(title="Weeks"),
            zaxis=dict(title="Lot ID"),
        ),
        title="3D Plot Graph with Time Series Slider",
    )

    # Show the plot
    fig.show()


def show_PSI(CPU_lot_list):

    # 提供されたデータをDataFrameに変換
    CPU_lot_list_header = [
        "node_PSI_location",
        "node_name",
        "PSI_name",
        "week",
        "lot_ID",
        "step",
    ]

    df = pd.DataFrame(CPU_lot_list, columns=CPU_lot_list_header)

    # PSI_nameごとに色を指定
    PSI_name_colors = {0: "lightblue", 1: "darkblue", 2: "brown", 3: "yellow"}
    df["color"] = df["PSI_name"].map(PSI_name_colors)

    # 3D散布図を作成
    fig = px.scatter_3d(
        df,
        x="node_PSI_location",
        y="week",
        z="step",
        color="color",
        hover_data=["lot_ID"],
    )

    # グラフを表示
    fig.show()


def show_PSI_seq(CPU_lot_list, location_sequence):

    # データフレームを作成
    df = pd.DataFrame(
        CPU_lot_list,
        columns=["node_PSI_location", "node_name", "PSI_no", "week", "lot_ID", "step"],
    )

    # 辞書psi_no_nameを使って"PSI_no"を"PSI_NAME"に変換
    psi_no_name = {0: "S", 1: "CO", 2: "I", 3: "P"}

    df["PSI_name"] = df["PSI_no"].map(psi_no_name)

    # “node_PSI_location”: ノードのPSIの場所
    # “node_no”: ノード番号
    # “PSI_no”: PSI番号
    # “week”: 週
    # “lot_ID”: ロットID
    # “step”: ステップ
    # “PSI_NAME”: PSI名（"PSI_no"に対応する名前）

    #    # 提供されたデータをDataFrameに変換
    #    CPU_lot_list_header = ["node_PSI_location", "node_name", "PSI_no", "week", "lot_ID", "step"]
    #
    #
    #    # データフレームを作成
    #    df = pd.DataFrame(CPU_lot_list, columns=["node_PSI_location", "node_no", "PSI_name", "week", "lot_ID", "step"])

    # PSI_name別の色指定
    PSI_name_colors = {"S": "blue", "CO": "darkblue", "I": "brown", "P": "yellow"}
    # PSI_name_colors = {"S": 'lightblue', "CO": 'darkblue', "I": 'brown', "P": 'yellow'}

    #    # PSI_name別の色指定
    #    PSI_name_colors = {0: 'lightblue', 1: 'darkblue', 2: 'brown', 3: 'yellow'}

    # X軸上の表示順を指定するリスト
    # location_sequence = ['0HAM_N', '1HAM_N', '2HAM_N', '3HAM_N', '0HAM_D', '1HAM_D', '2HAM_D', '3HAM_D',       '0HAM', '1HAM', '2HAM', '3HAM', '0JPN-OUT', '1JPN-OUT', '2JPN-OUT', '3JPN-OUT']

    # 3D散布図を作成
    fig = px.scatter_3d(
        df,
        x="node_PSI_location",
        y="week",
        z="step",
        color="PSI_name",
        color_discrete_map=PSI_name_colors,
        hover_data=["lot_ID"],
        category_orders={"node_PSI_location": location_sequence},
    )

    # グラフを表示
    fig.show()


def show_PSI_seq_dic(CPU_lot_list, location_sequence, location_dic):

    # データフレームを作成
    df = pd.DataFrame(
        CPU_lot_list,
        columns=["node_PSI_location", "node_name", "PSI_no", "week", "lot_ID", "step"],
    )

    # 辞書psi_no_nameを使って"PSI_no"を"PSI_NAME"に変換
    psi_no_name = {0: "S", 1: "CO", 2: "I", 3: "P"}

    df["PSI_name"] = df["PSI_no"].map(psi_no_name)

    # “node_PSI_location”: ノードのPSIの場所
    # “node_no”: ノード番号
    # “PSI_no”: PSI番号
    # “week”: 週
    # “lot_ID”: ロットID
    # “step”: ステップ
    # “PSI_NAME”: PSI名（"PSI_no"に対応する名前）

    # PSI_name別の色指定
    PSI_name_colors = {"S": "blue", "CO": "darkblue", "I": "brown", "P": "yellow"}

    # X軸上の表示順を指定するリスト
    # location_sequence = ['0HAM_N', '1HAM_N', '2HAM_N', '3HAM_N', '0HAM_D', '1HAM_D', '2HAM_D', '3HAM_D',       '0HAM', '1HAM', '2HAM', '3HAM', '0JPN-OUT', '1JPN-OUT', '2JPN-OUT', '3JPN-OUT']

    # Create a custom legend using annotations
    annotations = []
    for loc in location_sequence:
        value = location_dic.get(loc, "")
        annotations.append(
            dict(
                xref="paper",
                yref="paper",
                x=0.75,
                y=1 - 0.03 * len(annotations),
                xanchor="left",
                yanchor="top",
                text=f"{loc}: {value}",
                font=dict(size=12),
                showarrow=False,
            )
        )

    # 3D散布図を作成
    fig = px.scatter_3d(
        df,
        x="node_PSI_location",
        y="week",
        z="step",
        color="PSI_name",
        color_discrete_map=PSI_name_colors,
        hover_data=["lot_ID"],
        category_orders={"node_PSI_location": location_sequence},
    )

    # Add custom legend annotations
    fig.update_layout(annotations=annotations)

    # グラフを表示
    fig.show()


def show_PSI_node_dic(CPU_lot_list, location_sequence, node_sequence, node_dic):
    # def show_PSI_seq_dic(CPU_lot_list, location_sequence, location_dic):

    # データフレームを作成
    df = pd.DataFrame(
        CPU_lot_list,
        columns=["node_PSI_location", "node_name", "PSI_no", "week", "lot_ID", "step"],
    )

    # 辞書psi_no_nameを使って"PSI_no"を"PSI_NAME"に変換
    psi_no_name = {0: "S", 1: "CO", 2: "I", 3: "P"}

    df["PSI_name"] = df["PSI_no"].map(psi_no_name)

    # “node_PSI_location”: ノードのPSIの場所
    # “node_no”: ノード番号
    # “PSI_no”: PSI番号
    # “week”: 週
    # “lot_ID”: ロットID
    # “step”: ステップ
    # “PSI_NAME”: PSI名（"PSI_no"に対応する名前）

    # PSI_name別の色指定
    PSI_name_colors = {"S": "blue", "CO": "darkblue", "I": "brown", "P": "yellow"}

    # X軸上の表示順を指定するリスト
    # location_sequence = ['0HAM_N', '1HAM_N', '2HAM_N', '3HAM_N', '0HAM_D', '1HAM_D', '2HAM_D', '3HAM_D',       '0HAM', '1HAM', '2HAM', '3HAM', '0JPN-OUT', '1JPN-OUT', '2JPN-OUT', '3JPN-OUT']

    # Create a custom legend using annotations
    annotations = []
    for nd in node_sequence:
        value = node_dic.get(nd, "")
        annotations.append(
            dict(
                xref="paper",
                yref="paper",
                x=0.75,
                y=1 - 0.03 * len(annotations),
                xanchor="left",
                yanchor="top",
                text=f"{nd}:Revenue {value}",
                font=dict(size=12),
                showarrow=False,
            )
        )

    # 3D散布図を作成
    fig = px.scatter_3d(
        df,
        x="node_PSI_location",
        y="week",
        z="step",
        color="PSI_name",
        color_discrete_map=PSI_name_colors,
        hover_data=["lot_ID"],
        category_orders={"node_PSI_location": location_sequence},
    )

    # Add custom legend annotations
    fig.update_layout(annotations=annotations)

    # グラフを表示
    fig.show()


# *******************************************
# start main
# *******************************************


def main():

    # 最初にmonth2week変換をして、plan_rangeをセットする

    # trans_month2week
    # ***************************

    # in_file    = "S_month_data.csv"
    # in_file    = "S_month_data_prev_year_sssmall_JPN.csv"

    in_file = "S_month_data_prev_year_JPN.csv"

    out_file = "S_iso_week_data.csv"

    # print('BEFORE plan_range = root_node_outbound.plan_range',plan_range,root_node_outbound.plan_range)

    # STOP
    # plan_range = root_node_outbound.plan_range
    # START
    plan_range = 2  #### 計画期間=2年は、計画前年+計画年=2年で初期設定値

    # STOP
    # print('AFTER1  plan_range = root_node_outbound.plan_range',plan_range,root_node_outbound.plan_range)

    print("AFTER1  plan_range", plan_range)

    # month 2 week 変換でplan_rangeのN年が判定されて返される。
    node_yyyyww_value, node_yyyyww_key, plan_range, df_capa_year = trans_month2week(
        in_file, out_file
    )

    # STOP
    # print('AFTER2  plan_range = root_node_outbound.plan_range',plan_range,root_node_outbound.plan_range)

    print("AFTER2  plan_range", plan_range)

    # ***************************
    # Supply Chain planning 全体の流れ
    # ***************************
    #
    # 1. leaf_nodeのweekly Sの生成
    # 2. leaf_nodeのSをbwdで、mother_plantにセット
    #
    # option => 1)capaを増減 2)先行生産する、3)demandを待たせる
    #
    # 3. mother_plant上のSを(ここでは先行生産で)confirmed_Sに確定
    #
    # 4. mother_plantからdecoupling pointまで、PUSH operetion
    # 5. ??decoupling pointから先はleaf nodeからのPULLに該当するdemandで処理??
    #
    # 6. outbound and inbound CONNECTOR
    # 7. inbound PULL operotor

    # ***************************
    # build_psi_core
    # ***************************
    # build_psi_core()

    # create_tree

    # init_psi_dict # 辞書形式の3種類のpsiデータの初期化
    # demand plan, supply plan, decouple plan

    # init_set_psi_dict2tree # 辞書psiをtreeに初期セット

    # 以上で、psi_treeの骨格が完成
    # 後は、アプリケーション属性を追加して、操作と評価を繰り返す

    # ***************************
    # tree definition initialise
    # ***************************

    node_I4bullwhip = {}


    # ***************************
    # create outbound tree
    # ***************************

    # @240421 for csv2dic
    outbound_tree_file = "profile_test_outbound.csv"

    ##@240421 for csv2dic
    # profile_name = "profile_test_outbound.csv"

    # nodes_xxxは、すべてのnodeインスタンスをnode_nameで引出せる辞書
    nodes_outbound = {}
    # nodes_outbound, root_node_name = create_tree(outbound_tree_file)
    nodes_outbound, root_node_name = create_tree_set_attribute(outbound_tree_file)

    # rootのインスタンスを取得
    root_node_outbound = nodes_outbound[root_node_name]

    # root_node_outbound = nodes_outbound['JPN']
    # root_node_outbound = nodes_outbound['JPN_OUT']

    # 子nodeに親nodeをセットする
    set_parent_all(root_node_outbound)

    # ***************************
    # rean and set node_position2dic
    # ***************************
    # CSVファイルを読み込む
    df = pd.read_csv("node_position.csv")

    # DataFrameを辞書に変換する
    node_position_dic = df.set_index("node_name")[["longtude", "latitude"]].T.to_dict(
        "list"
    )

    print(node_position_dic)
    # print(node_position_dic["JPN"])

    # すべてのnodeにposition座標 経度と緯度をセットする
    set_position_all(root_node_outbound, node_position_dic)

    # ***************************
    # create inbound tree
    # ***************************

    # inbound_tree_file = "supply_chain_tree_inbound_attributes_JPN.csv"

    # @240421 for csv2dic
    inbound_tree_file = "profile_test_inbound.csv"

    nodes_inbound = {}

    # nodes_inbound, root_node_name = create_tree(inbound_tree_file)
    nodes_inbound, root_node_name = create_tree_set_attribute(inbound_tree_file)

    root_node_inbound = nodes_inbound[root_node_name]

    # inboundの親子ホインタはセットしていない

    # treeが出来たので、rootから順に、
    # Nodeのインスタンスにself.で明示的にplan_rangeをセットする。
    root_node_outbound.set_plan_range_lot_counts(plan_range)
    root_node_inbound.set_plan_range_lot_counts(plan_range)

    # set_plan_range(root_node_outbound, plan_range)
    # set_plan_range(root_node_inbound, plan_range)

    # plan_range RESET for lot_counts / psi4supply / EVAL_WORK_AREA
    # root_node_outbound.reset_plan_range_related_attributes(plan_range)
    # root_node_inbound.reset_plan_range_related_attributes(plan_range)

    # ***************************

    # an image of data
    #
    # for node_val in node_yyyyww_value:
    #   #print( node_val )
    #
    ##['SHA_N', 22.580645161290324, 22.580645161290324, 22.580645161290324, 22.5    80645161290324, 26.22914349276974, 28.96551724137931, 28.96551724137931, 28.    96551724137931, 31.067853170189103, 33.87096774193549, 33.87096774193549, 33    .87096774193549, 33.87096774193549, 30.33333333333333, 30.33333333333333, 30    .33333333333333, 30.33333333333333, 31.247311827956988, 31.612903225806452,

    # node_yyyyww_key [['CAN', 'CAN202401', 'CAN202402', 'CAN202403', 'CAN20240    4', 'CAN202405', 'CAN202406', 'CAN202407', 'CAN202408', 'CAN202409', 'CAN202    410', 'CAN202411', 'CAN202412', 'CAN202413', 'CAN202414', 'CAN202415', 'CAN2    02416', 'CAN202417', 'CAN202418', 'CAN202419',

    # ********************************
    # make_node_psi_dict
    # ********************************
    # 1. treeを生成して、nodes[node_name]辞書で、各nodeのinstanceを操作する
    # 2. 週次S yyyywwの値valueを月次Sから変換、
    #    週次のlotの数Slotとlot_keyを生成、
    # 3. ロット単位=lot_idとするリストSlot_id_listを生成しながらpsi_list生成
    # 4. node_psi_dict=[node1: psi_list1,,,]を生成、treeのnode.psi4demandに接続する

    S_week = []

    # *************************************************
    # initialise node_psi_dict
    # *************************************************
    node_psi_dict = {}  # 変数 node_psi辞書

    # ***************************
    # outbound psi_dic
    # ***************************
    node_psi_dict_Ot4Dm = {}  # node_psi辞書4demand plan
    node_psi_dict_Ot4Sp = {}  # node_psi辞書4supply plan

    # coupling psi
    node_psi_dict_Ot4Cl = {}  # node_psi辞書4couple plan

    # accume psi
    node_psi_dict_Ot4Ac = {}  # node_psi辞書outbound4accume plan

    # ***************************
    # inbound psi_dic
    # ***************************
    node_psi_dict_In4Dm = {}  # node_psi辞書inbound4demand plan
    node_psi_dict_In4Sp = {}  # node_psi辞書inbound4supply plan

    # coupling psi
    node_psi_dict_In4Cl = {}  # node_psi辞書inbound4couple plan

    # accume psi
    node_psi_dict_In4Ac = {}  # node_psi辞書inbound4accume plan

    # rootからtree nodeをpreorder順に検索 node_psi辞書を空リストを作る
    node_psi_dict_Ot4Dm = make_psi_space_dict(
        root_node_outbound, node_psi_dict_Ot4Dm, plan_range
    )
    node_psi_dict_Ot4Sp = make_psi_space_dict(
        root_node_outbound, node_psi_dict_Ot4Sp, plan_range
    )
    node_psi_dict_Ot4Cl = make_psi_space_dict(
        root_node_outbound, node_psi_dict_Ot4Cl, plan_range
    )
    node_psi_dict_Ot4Ac = make_psi_space_dict(
        root_node_outbound, node_psi_dict_Ot4Ac, plan_range
    )

    node_psi_dict_In4Dm = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Dm, plan_range
    )
    node_psi_dict_In4Sp = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Sp, plan_range
    )
    node_psi_dict_In4Cl = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Cl, plan_range
    )
    node_psi_dict_In4Ac = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Ac, plan_range
    )

    # ***********************************
    # set_dict2tree
    # ***********************************
    # rootからtreeをpreorder順に検索 node_psi辞書をnodeにset
    set_dict2tree_psi(root_node_outbound, "psi4demand", node_psi_dict_Ot4Dm)
    set_dict2tree_psi(root_node_outbound, "psi4supply", node_psi_dict_Ot4Sp)
    set_dict2tree_psi(root_node_outbound, "psi4couple", node_psi_dict_Ot4Cl)
    set_dict2tree_psi(root_node_outbound, "psi4accume", node_psi_dict_Ot4Ac)

    set_dict2tree_psi(root_node_inbound, "psi4demand", node_psi_dict_In4Dm)
    set_dict2tree_psi(root_node_inbound, "psi4supply", node_psi_dict_In4Sp)
    set_dict2tree_psi(root_node_inbound, "psi4couple", node_psi_dict_In4Cl)
    set_dict2tree_psi(root_node_inbound, "psi4accume", node_psi_dict_In4Ac)

    # *********************************
    # inbound data initial setting
    # *********************************

    # *********************************
    # node_psi辞書を作成して、node.psiにセットする
    # *********************************
    node_psi_dict_In4Dm = {}  # node_psi辞書を定義 # Inbound for Demand
    node_psi_dict_In4Sp = {}  # node_psi辞書を定義 # Inbound for Supply

    # rootからtree nodeをinbound4demand=preorder順に検索 node_psi辞書をmake
    node_psi_dict_In4Dm = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Dm, plan_range
    )
    node_psi_dict_In4Sp = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Sp, plan_range
    )

    # ***********************************
    # set_dict2tree
    # ***********************************
    # rootからtreeをinbound4demand=preorder順に検索 node_psi辞書をnodeにset
    set_dict2tree_In4Dm(root_node_inbound, node_psi_dict_In4Dm)
    set_dict2tree_In4Sp(root_node_inbound, node_psi_dict_In4Sp)

    # node_nameを先頭に、ISO week順にSのリストで持つ
    # leaf_nodeにはISO week Sが入っているが、
    # leaf以外のnode値=0 (需要シフト時に生成される)

    S_lots_dict = make_S_lots(node_yyyyww_value, node_yyyyww_key, nodes_outbound)

    set_Slots2psi4demand(root_node_outbound, S_lots_dict)

    show_sw = 0  # 1 or 0

    # node指定
    # node_show = "Platform"
    # node_show = "JPN"
    # node_show = "TrBJPN2HAM"

    node_show = "HAM"

    # node_show = "MUC"
    # node_show = "MUC_D"
    # node_show = "SHA"
    # node_show = "SHA_D"
    # node_show = "CAN_I"

    if show_sw == 1:
        show_psi_graph(root_node_outbound, "demand", node_show, 0, 300)

    # demand planのS Pの生成

    # ***************************************
    # you can see root_node_outbound with "mplot3d" if you want
    # ****************************************
    # show_psi_3D_graph_node(root_node_outbound)

    # @240422 memo *** this is "PUSH operotor for outbound"

    # ***************************************
    # OUT / FW / psi2i
    # ***************************************
    # calc_all_psi2i
    # ***************************************
    # SP2I計算はpreorderingでForeward     Planningする
    calc_all_psi2i4demand(root_node_outbound)

    if show_sw == 1:
        show_psi_graph(root_node_outbound, "demand", node_show, 0, 300)

    # *********************************
    # mother plant capacity parameter
    # *********************************

    demand_supply_ratio = 3  # demand_supply_ratio = ttl_supply / ttl_demand

    # ********************
    # common_plan_unit_lot_size
    # OR
    # lot_size on root( = mother plant )
    # ********************
    plant_lot_size = 0

    # mother plantのlot_size定義を取るのはやめて、
    # common plant unitとして一つのlot_sizeを使う

    common_plan_unit_lot_size = 1  # 100 #24 #50 # 100  # 100   # 3 , 10, etc
    # common_plan_unit_lot_size = 100 #24 #50 # 100  # 100   # 3 , 10, etc

    plant_lot_size = common_plan_unit_lot_size

    # plant_lot_size     = root_node_outbound.lot_size # parameter master file

    # ********************
    # 辞書 year key: total_demand
    # ********************

    # 切り捨ては、a//b
    # 切り上げは、(a+b-1)//b

    plant_capa_vol = {}
    plant_capa_lot = {}

    week_vol = 0

    for i, row in df_capa_year.iterrows():

        plant_capa_vol[row["year"]] = row["total_demand"]

        # plant_capa_lot[row['year']] = (row['total_demand']+plant_lot_size -1)//     plant_lot_size # 切り上げ

        week_vol = row["total_demand"] * demand_supply_ratio // 52

        plant_capa_lot[row["year"]] = (week_vol + plant_lot_size - 1) // plant_lot_size

        # plant_capa_lot[row['year']] = ((row['total_demand']+52-1 // 52)+plant_lot_size-1) // plant_lot_size
        # plant_capa_lot[row['year']] = row['total_demand'] // plant_lot_size

    # **********************
    # ISO weekが年によって52と53がある
    # ここでは、53*self.plan_rangeの年別53週のaverage_capaとして定義
    # **********************

    # 53*self.plan_range
    #

    year_st = 2020
    year_end = 2021

    year_st = df_capa_year["year"].min()
    year_end = df_capa_year["year"].max()

    week_capa = []
    week_capa_w = []

    for year in range(year_st, year_end + 1):  # 5_years

        week_capa_w = [plant_capa_lot[year]] * 53
        # week_capa_w = [ (plant_capa_lot[year] + 53 - 1) // 53 ] * 53

        week_capa += week_capa_w

    leveling_S_in = []

    leveling_S_in = root_node_outbound.psi4demand

    # calendar　先行生産によるキャパ対応、

    # *****************************
    # mother plan leveling    setting initial data
    # *****************************

    # a sample data setting

    week_no = 53 * plan_range

    S_confirm = 15

    S_lots = []
    S_lots_list = []

    for w in range(53 * plan_range):

        S_lots_list.append(leveling_S_in[w][0])

    prod_capa_limit = week_capa

    # ******************
    # initial setting
    # ******************

    capa_ceil = 50
    # capa_ceil = 100
    # capa_ceil = 10

    S_confirm_list = confirm_S(S_lots_list, prod_capa_limit, plan_range)

    # **********************************
    # 多次元リストの要素数をcountして、confirm処理の前後の要素数を比較check
    # **********************************
    S_lots_list_element = multi_len(S_lots_list)

    S_confirm_list_element = multi_len(S_confirm_list)

    # *********************************
    # initial setting
    # *********************************
    node_psi_dict_Ot4Sp = {}  # node_psi_dict_Ot4Spの初期セット

    node_psi_dict_Ot4Sp = make_psi4supply(root_node_outbound, node_psi_dict_Ot4Sp)

    #
    # node_psi_dict_Ot4Dmでは、末端市場のleafnodeのみセット
    #
    # root_nodeのS psi_list[w][0]に、levelingされた確定出荷S_confirm_listをセッ    ト

    # 年間の総需要(総lots)をN週先行で生産する。
    # 例えば、３ヶ月先行は13週先行生産として、年間総需要を週平均にする。

    # S出荷で平準化して、confirmedS-I-P
    # conf_Sからconf_Pを生成して、conf_P-S-I  PUSH and PULL

    S_list = []
    S_allocated = []

    year_lots_list = []
    year_week_list = []

    leveling_S_in = []

    leveling_S_in = root_node_outbound.psi4demand

    # psi_listからS_listを生成する
    for psi in leveling_S_in:

        S_list.append(psi[0])

    # 開始年を取得する
    plan_year_st = year_st  # 開始年のセット in main()要修正

    for yyyy in range(plan_year_st, plan_year_st + plan_range + 1):

        year_lots = count_lots_yyyy(S_list, str(yyyy))

        year_lots_list.append(year_lots)

    #        # 結果を出力
    #       #print(yyyy, " year carrying lots:", year_lots)
    #
    #    # 結果を出力
    #   #print(" year_lots_list:", year_lots_list)

    # an image of sample data
    #
    # 2023  year carrying lots: 0
    # 2024  year carrying lots: 2919
    # 2025  year carrying lots: 2914
    # 2026  year carrying lots: 2986
    # 2027  year carrying lots: 2942
    # 2028  year carrying lots: 2913
    # 2029  year carrying lots: 0
    #
    # year_lots_list: [0, 2919, 2914, 2986, 2942, 2913, 0]

    year_list = []

    for yyyy in range(plan_year_st, plan_year_st + plan_range + 1):

        year_list.append(yyyy)

        # テスト用の年を指定
        year_to_check = yyyy

        # 指定された年のISO週数を取得
        week_count = is_52_or_53_week_year(year_to_check)

        year_week_list.append(week_count)

    #        # 結果を出力
    #       #print(year_to_check, " year has week_count:", week_count)
    #
    #    # 結果を出力
    #   #print(" year_week_list:", year_week_list)

    # print("year_list", year_list)

    # an image of sample data
    #
    # 2023  year has week_count: 52
    # 2024  year has week_count: 52
    # 2025  year has week_count: 52
    # 2026  year has week_count: 53
    # 2027  year has week_count: 52
    # 2028  year has week_count: 52
    # 2029  year has week_count: 52
    # year_week_list: [52, 52, 52, 53, 52, 52, 52]

    # *****************************
    # 生産平準化のための年間の週平均生産量(ロット数単位)
    # *****************************

    # *****************************
    # make_year_average_lots
    # *****************************
    # year_list     = [2023,2024,2025,2026,2027,2028,2029]

    # year_lots_list = [0, 2919, 2914, 2986, 2942, 2913, 0]
    # year_week_list = [52, 52, 52, 53, 52, 52, 52]

    year_average_lots_list = []

    for lots, weeks in zip(year_lots_list, year_week_list):
        average_lots_per_week = math.ceil(lots / weeks)
        year_average_lots_list.append(average_lots_per_week)

    # print("year_average_lots_list", year_average_lots_list)
    #
    # an image of sample data
    #
    # year_average_lots_list [0, 57, 57, 57, 57, 57, 0]

    # 年間の総需要(総lots)をN週先行で生産する。
    # 例えば、３ヶ月先行は13週先行生産として、年間総需要を週平均にする。

    #
    # 入力データの前提
    #
    # leveling_S_in[w][0] == S_listは、outboundのdemand_planで、
    # マザープラントの出荷ポジションのSで、
    # 5年分 週次 最終市場におけるlot_idリストが
    # LT offsetされた状態で入っている
    #
    # year_list     = [2023,2024,2025,2026,2027,2028,2029]

    # year_lots_list = [0, 2919, 2914, 2986, 2942, 2913, 0]
    # year_week_list = [52, 52, 52, 53, 52, 52, 52]
    # year_average_lots_list [0, 57, 57, 57, 57, 57, 0]

    # ********************************
    # 先行生産の週数
    # ********************************
    # precedence_production_week =13

    # pre_prod_week =26 # 26週=6か月の先行生産をセット
    # pre_prod_week =13 # 13週=3か月の先行生産をセット
    pre_prod_week = 6  # 6週=1.5か月の先行生産をセット

    # ********************************
    # 先行生産の開始週を求める
    # ********************************
    # 市場投入の前年において i= 0  year_list[i]           # 2023
    # 市場投入の前年のISO週の数 year_week_list[i]         # 52

    # 先行生産の開始週は、市場投入の前年のISO週の数 - 先行生産週

    pre_prod_start_week = 0

    i = 0

    pre_prod_start_week = year_week_list[i] - pre_prod_week

    # スタート週の前週まで、[]リストで埋めておく
    for i in range(pre_prod_start_week):
        S_allocated.append([])

    # ********************************
    # 最終市場からのLT offsetされた出荷要求lot_idリストを
    # Allocate demand to mother plant weekly slots
    # ********************************

    # S_listの週別lot_idリストを一直線のlot_idリストに変換する
    # mother plant weekly slots

    # 空リストを無視して、一直線のlot_idリストに変換

    # 空リストを除外して一つのリストに結合する処理
    S_one_list = [item for sublist in S_list if sublist for item in sublist]

    ## 結果表示
    ##print(S_one_list)

    # to be defined 毎年の定数でのlot_idの切り出し

    # listBの各要素で指定された数だけlistAから要素を切り出して
    # 新しいリストlistCを作成

    listA = S_one_list  # 5年分のlot_idリスト

    listB = year_lots_list  # 毎年毎の総ロット数

    listC = []  # 毎年のlot_idリスト

    start_idx = 0

    for i, num in enumerate(listB):

        end_idx = start_idx + num

        # original sample
        # listC.append(listA[start_idx:end_idx])

        # **********************************
        # "slice" and "allocate" at once
        # **********************************
        sliced_lots = listA[start_idx:end_idx]

        # 毎週の生産枠は、year_average_lots_listの平均値を取得する。
        N = year_average_lots_list[i]

        if N == 0:

            pass

        else:

            # その年の週次の出荷予定数が生成される。
            S_alloc_a_year = [
                sliced_lots[j : j + N] for j in range(0, len(sliced_lots), N)
            ]

            S_allocated.extend(S_alloc_a_year)
            # S_allocated.append(S_alloc_a_year)

        start_idx = end_idx

    ## 結果表示
    # print("S_allocated", S_allocated)

    # set psi on outbound supply

    # "JPN-OUT"
    #

    node_name = root_node_outbound.name  # Nodeからnode_nameを取出す

    # for w, pSi in enumerate( S_allocated ):
    #
    #    node_psi_dict_Ot4Sp[node_name][w][0] = pSi

    for w in range(53 * plan_range):

        if w <= len(S_allocated) - 1:  # index=0 start

            node_psi_dict_Ot4Sp[node_name][w][0] = S_allocated[w]

        else:

            node_psi_dict_Ot4Sp[node_name][w][0] = []

    # supply_plan用のnode_psi_dictをtree構造のNodeに接続する
    # Sをnode.psi4supplyにset  # psi_listをclass Nodeに接続

    set_psi_lists4supply(root_node_outbound, node_psi_dict_Ot4Sp)

    # この後、
    # process_1 : mother plantとして、S2I_fixed2Pで、出荷から生産を確定
    # process_2 : 子nodeに、calc_S2P2psI()でfeedbackする。

    # if show_sw == 1:
    #    show_psi_graph(root_node_outbound, "supply", node_show, 0, 300)

    # @240422 memo *** this is "mother plant's confirmedS2P and PS2I"

    # demand planからsupply planの初期状態を生成

    # *************************
    # process_1 : mother plantとして、S2I_fixed2Pで、出荷から生産を確定
    # *************************

    # calcS2fixedI2P
    # psi4supplyを対象にする。
    # psi操作の結果Pは、S2PをextendでなくS2Pでreplaceする

    root_node_outbound.calcS2P_4supply()  # mother plantのconfirm S=> P

    root_node_outbound.calcPS2I4supply()  # mother plantのPS=>I

    # show_psi_graph(root_node_outbound,"supply", node_show, 0, 300 )
    if show_sw == 1:
        show_psi_graph(root_node_outbound, "supply", "JPN", 0, 300)

    # mother plantのみのleveling生産平準化処理
    # mother plant="JPN"のsupply planを、年間需要の週平均値でlot数を平均化

    # *************************
    # process_2 : 子nodeに、calc_S2P2psI()でfeedbackする。
    # *************************

    #
    # 接続して、mother_plant確定Sを生成し、tree中の子nodeに確定lotをfeedback
    #

    ##print('feedback node_psi_dict_Ot4Sp',node_psi_dict_Ot4Sp)

    # ***************************************
    # その3　都度のparent searchを実行
    # ***************************************
    feedback_psi_lists(root_node_outbound, node_psi_dict_Ot4Sp, nodes_outbound)

    #
    # lot by lotのサーチ　その1 遅い
    #
    # ***************************************
    # S_confirm_list: mother planの出荷計画を平準化、確定した出荷計画を
    # children_node.P_request  : すべての子nodeの出荷要求数のリストと比較して、
    # children_node.P_confirmed: それぞれの子nodeの出荷確定数を生成する
    # ***************************************

    #
    # lot by lotのサーチ その2 少し遅い
    #
    # ***************************************
    # tree上の子nodeをサーチして、子nodeのSに、lotかあるか(=出荷先nodeか)
    # ***************************************

    #
    # lot by lotのサーチ その3 少し早い
    #
    # ***************************************
    # lot処理の都度、以下のサーチを実行
    # lot_idが持つleaf_nodeの情報から、parent_nodeをサーチ、出荷先nodeを特定
    # ***************************************

    # lot by lotのサーチ その4 早いハズ
    #
    # ***************************************
    # creat_treeの後、leaf_nodeの辞書に、reverse(leaf_root_list)を作り、
    # lot_idが持つleaf_node情報から、leaf_root辞書から出荷先nodeを特定
    # root_leaf_list中の「指定したnodeの次」list[index(node)+1]を取り出す
    # ***************************************

    # if show_sw == 1:
    #    show_psi_graph(root_node_outbound, "supply", node_show, 0, 300)

    # *********************************
    # make visualise data for 3D bar graph
    # *********************************
    #    visualise_inventory4demand_3d_bar(root_node_outbound, 'demand_I_bar.html')

    # ***************************************
    # decouple nodeを判定して、
    # calc_all_psi2iのSをPUSHからPULLに切り替える
    # ***************************************

    # nodes_decouple_all = [] # nodes_decoupleのすべてのパターンをリストアップ
    #
    # decoupleパターンを計算・評価
    # for i, nodes_decouple in enumerate(nodes_decouple_all):
    #    calc_xxx(root_node, , , , )
    #    eval_xxx(root_node, eval)

    nodes_decouple_all = make_nodes_decouple_all(root_node_outbound)

    for i, nodes_decouple in enumerate(nodes_decouple_all):

        decouple_flag = "OFF"

        calc_all_psi2i_decouple4supply(
            root_node_outbound,
            nodes_decouple,
            decouple_flag,
            node_psi_dict_Ot4Dm,
            nodes_outbound,
        )

        # outbound supplyのIをsummary
        # setting on "node.decoupling_total_I"

        total_I = 0

        node_eval_I = {}

        # decoupleだけでなく、tree all nodeでグラフ表示
        total_I, node_eval_I = evaluate_inventory_all(
            root_node_outbound, total_I, node_eval_I, nodes_decouple
        )

        show_subplots_set_y_axies(node_eval_I, nodes_decouple)

    # show_psi_graph(root_node_outbound,"demand", node_show, 0, 300 ) #
    # show_psi_graph(root_node_outbound,"supply", node_show, 0, 300 ) #

    if show_sw == 2:

        show_psi_graph(root_node_outbound, "supply", "JPN", 0, 300)  #
        show_psi_graph(root_node_outbound, "supply", "TrBJPN2HAM", 0, 300)  #
        show_psi_graph(root_node_outbound, "supply", "HAM", 0, 300)  #
        show_psi_graph(root_node_outbound, "supply", "MUC", 0, 300)  #
        show_psi_graph(root_node_outbound, "supply", "MUC_D", 0, 300)  #

        show_psi_graph(root_node_outbound, "supply", "HAM_D", 0, 300)  #
        show_psi_graph(root_node_outbound, "supply", "HAM_I", 0, 300)  #
        show_psi_graph(root_node_outbound, "supply", "HAM_N", 0, 300)  #

    # show_psi_graph(root_node_outbound, "supply", "JPN", 0, 300)  #

    # *********************************
    # make visualise data for 3D bar graph
    # *********************************
    #    visualise_inventory4supply_3d_bar(root_node_outbound, 'supply_I_bar.html')

    # *********************************
    # psi4accume  accume_psi initial setting on Inbound and Outbound
    # *********************************

    # *********************************
    # node_psi辞書を作成して、node.psiにセットする
    # *********************************
    node_psi_dict_In4Ac = {}  # node_psi辞書を定義 # Inbound for Accume
    node_psi_dict_Ot4Ac = {}  # node_psi辞書を定義 # Outbound for Accume

    # *********************************
    # make dict from tree getting node_name and setting [[]*53*self.plan_range]
    # *********************************
    # inboundとoutboundのtreeをrootからpreorder順に検索 node_psi辞書をmake

    node_psi_dict_Ot4Ac = make_psi_space_zero_dict(
        root_node_outbound, node_psi_dict_Ot4Ac, plan_range
    )

    node_psi_dict_In4Ac = make_psi_space_zero_dict(
        root_node_inbound, node_psi_dict_In4Ac, plan_range
    )

    # ***********************************
    # set_dict2tree
    # ***********************************
    # rootから in&out treeをpreorder順に検索 node_psi辞書をnodeにset

    # psi4accumeは、inbound outbound共通
    set_dict2tree_InOt4AC(root_node_outbound, node_psi_dict_Ot4Ac)
    set_dict2tree_InOt4AC(root_node_inbound, node_psi_dict_In4Ac)

    # class Nodeのnode.psi4accumeにセット
    # node.psi4accume = node_psi_dict.get(node.name)

    # *********************************
    # inbound data initial setting
    # *********************************

    # *********************************
    # node_psi辞書を作成して、node.psiにセットする
    # *********************************
    node_psi_dict_In4Dm = {}  # node_psi辞書を定義 # Inbound for Demand
    node_psi_dict_In4Sp = {}  # node_psi辞書を定義 # Inbound for Supply

    # rootからtree nodeをinbound4demand=preorder順に検索 node_psi辞書をmake
    node_psi_dict_In4Dm = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Dm, plan_range
    )
    node_psi_dict_In4Sp = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Sp, plan_range
    )

    # set_dict2tree
    # rootからtreeをinbound4demand=preorder順に検索 node_psi辞書をnodeにset
    set_dict2tree_In4Dm(root_node_inbound, node_psi_dict_In4Dm)
    set_dict2tree_In4Sp(root_node_inbound, node_psi_dict_In4Sp)

    # @240422 memo *** this is "outbound and inbound connector"
    # ここで、outboundとinboundを接続している
    connect_outbound2inbound(root_node_outbound, root_node_inbound)

    # @240422 memo *** this is "inbound PULL operator"

    # Backward / Inbound
    # calc_bwd_inbound_all_si2p
    # S2P

    node_psi_dict_In4Dm = calc_bwd_inbound_all_si2p(
        root_node_inbound, node_psi_dict_In4Dm
    )

    ## ***********************************
    ## evaluation process    setting revenue-cost-profit
    ## ***********************************
    eval_supply_chain(root_node_outbound)
    eval_supply_chain(root_node_inbound)

    # **********************************
    # transrate fron PSI list structure to a pandas dataframe for 3D plot
    # **********************************

    # a target "data layout "image of df
    # df4PSI_visualize = product, node_PSI_location, node_name, PSI_name, week, lot_ID, step
    #
    # an image of sample data4 PSI_visualize
    # "AAAAAAA", "0"+"HAM_N", "HAM_N", "Sales", "W26", 'HAM_N2024390', 2

    # node_PSI_location = "PSI_index" + "node_name"
    #
    # lot_ID_list = ['HAM_N2024390', 'HAM_N2024391']
    #

    #
    # X = a position of sequence(node_PSI_location) # node+PSIの表示順
    # Y = a week of weeks
    # Z(X,Y) = step

    CPU_lot_list = []

    CPU_lot_list = make_CPU_lot2list(root_node_outbound, CPU_lot_list)

    # print('CPU_lot_list4df',CPU_lot_list)

    show_PSI(CPU_lot_list)

    # *******************************************
    # Nodeの並びをLOVE-Mに合わせて、最終市場からサプライヤーへ
    # *******************************************
    node_seq_out = []
    node_seq_in = []
    node_seq = []

    node_sequence_out = make_node_post_order(root_node_outbound, node_seq_out)
    node_sequence_in = make_node_pre_order(root_node_inbound, node_seq_in)

    print("node_sequence_out", node_sequence_out)
    print("node_sequence_in ", node_sequence_in)

    # STOP
    # node_sequence = node_sequence_out + node_sequence_in

    # set OUT without IN
    node_sequence = node_sequence_out

    print("node_sequence    ", node_sequence)

    location_sequence = []
    location_dic = []
    psi_seq = [0, 1, 2, 3]

    location_sequence = make_loc_seq(node_sequence, psi_seq)
    # "0"+node_A, "1"+node_A, "2"+node_A, "3"+node_A

    print("location_sequence", location_sequence)

    node_dic = make_node_dic(location_sequence, nodes_outbound)  # outboundのみ

    location_dic = make_loc_dic(location_sequence, nodes_outbound)  # outboundのみ

    print("location_dic", location_dic)

    show_PSI_seq(CPU_lot_list, location_sequence)

    # @240508 各nodeの評価値の売上、利益、利益率を表示する
    show_PSI_seq_dic(CPU_lot_list, location_sequence, location_dic)

    show_PSI_node_dic(CPU_lot_list, location_sequence, node_sequence, node_dic)

    ##@240427 STOP
    # node_sequence = ['nodeA', 'nodeB', 'nodeC', 'nodeD', 'nodeE']

    PSI_locations = ["Sales", "Carry Over", "Inventory", "Purchase"]

    PSI_location_colors = {
        "Sales": "lightblue",
        "Carry Over": "darkblue",
        "Inventory": "brown",
        "Purchase": "yellow",
    }

    lot_ID_list_out = []
    lot_ID_list_in = []

    ####lot_ID_list         = [] # STOP

    ##@240428 イキ lot_ID_listの初期化
    # lot_ID_listのリスト構造は、以下の三次元 WEEK x PSI x NODEで構成されている

    lot_ID_list = [
        [[[] for i in range(53 * plan_range)] for j in range(len(PSI_locations))]
        for k in range(len(node_sequence))
    ]

    # lot_ID_list =[[[[] for i in range(318)] for j in range(len(PSI_locations))] for k in range(len(node_sequence))]

    ####lot_ID_list =[[[0 for i in range(318)] for j in range(len(PSI_locations))] for k in range(len(node_sequence))]

    nodes_all_dic = {}
    nodes_all_dic = nodes_outbound | nodes_inbound  # 辞書の統合

    lot_ID_list = make_lot(node_sequence, nodes_all_dic, lot_ID_list)

    print("lot_ID_listこの積上LOT_ID listを入力とする", lot_ID_list)

    # Define the weeks (Y-axis)
    # weeks = np.arange(0, 318) # 0 to 317 weeks

    weeks = np.arange(0, 53 * plan_range)  # from inputfile prev_month_S

    # STOP STARTrandom test data
    #
    lot_ID = np.random.randint(
        1, 100, size=(len(node_sequence), len(PSI_locations), 318)
    )

    ##@240429 STOP sample 3D plot graph-1
    #    show_lots_status_in_nodes_PSI_W318(node_sequence, PSI_locations, PSI_location_colors, weeks, lot_ID)

    # 入力データとして週別の「積上ロットIDのリスト」LOT_listを与えて3D plot graph
    #
    # @240429 START sample 3D plot graph-2  with LOT_list in weeks

    # @240508
    # show_lots_status_in_nodes_PSI_list_matrix(node_sequence, PSI_locations, PSI_location_colors, weeks, lot_ID_list)

    # STOP
    # show_lots_status_in_nodes_PSI_W318_list(node_sequence, PSI_locations, PSI_location_colors, weeks, lot_ID_list)

    print_tree_dfs(root_node_outbound, depth=0)
    print_tree_bfs(root_node_outbound)
    print("extructing CPU_lots")

    # 共通ロットCPU_lotの出力
    filename = "CPU_OUT_FW_plan010.csv"

    # csvファイルを書き込みモード「追記」で開く
    with open(filename, "w", newline="") as csvfile:

        # csv.writerオブジェクトを作成する
        csv_writer = csv.writer(csvfile)

        # treeの各ノードをpreordering search
        extract_CPU_tree_preorder(root_node_outbound, csv_writer)

    print("end of process")


if __name__ == "__main__":
    main()

