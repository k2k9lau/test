"""
交易數據分析系統 (Trading Analysis System)
支持大規模交易數據的處理與分析
包含兩個標籤頁：整體數據概覽、個別客戶探查
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from io import BytesIO

# ==================== 頁面配置 ====================
st.set_page_config(
    page_title="交易數據分析系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 欄位映射配置 ====================
COLUMN_MAP = {
    'execution_time': 'Execution Time\n交易时间',
    'open_time': 'Open Time\n开仓时间',
    'aid': 'AID\n用户账号',
    'closed_pl': 'Closed P/L\n平仓盈亏',
    'commission': 'Commission\n手续费',
    'swap': 'Swap\n隔夜利息',
    'instrument': 'Instrument\n交易品种',
    'business_type': 'Business Type\n业务类型',
    'action': 'Action\n交易类型',
    'volume': 'Volume\n开仓数量',
    'side': 'Side\n交易方向'
}


# ==================== 數據載入與預處理 ====================
@st.cache_data(show_spinner=False)
def load_and_preprocess(uploaded_files):
    """
    載入並預處理交易數據
    - 支持多檔合併
    - 自動欄位對齊
    - 計算 Net_PL 和 Hold_Seconds
    """
    dfs = []
    
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            dfs.append(df)
        except Exception as e:
            st.error(f"讀取檔案 {uploaded_file.name} 時發生錯誤: {e}")
            continue
    
    if not dfs:
        return None
    
    # 合併所有數據
    df = pd.concat(dfs, ignore_index=True)
    
    # 移除 Total 行（如果存在）
    exec_col = COLUMN_MAP['execution_time']
    if exec_col in df.columns:
        df = df[df[exec_col] != 'Total'].copy()
    
    # 去重
    df = df.drop_duplicates()
    
    # 轉換時間欄位
    for col in ['execution_time', 'open_time']:
        if COLUMN_MAP[col] in df.columns:
            df[COLUMN_MAP[col]] = pd.to_datetime(df[COLUMN_MAP[col]], errors='coerce')
    
    # 填充空值（盈虧與費用）
    for col in ['closed_pl', 'commission', 'swap']:
        if COLUMN_MAP[col] in df.columns:
            df[COLUMN_MAP[col]] = df[COLUMN_MAP[col]].fillna(0)
    
    # 計算 Net_PL = Closed P/L + Commission + Swap
    df['Net_PL'] = (
        df[COLUMN_MAP['closed_pl']] + 
        df[COLUMN_MAP['commission']] + 
        df[COLUMN_MAP['swap']]
    )
    
    # 計算 Hold_Seconds = (Execution Time - Open Time).dt.total_seconds()
    exec_time = df[COLUMN_MAP['execution_time']]
    open_time = df[COLUMN_MAP['open_time']]
    
    # 只在兩個時間都有效時計算
    df['Hold_Seconds'] = np.where(
        pd.notna(exec_time) & pd.notna(open_time),
        (exec_time - open_time).dt.total_seconds(),
        np.nan
    )
    
    # 確保 AID 為純數字字串（移除浮點數的 .0 後綴，不帶千分位逗號）
    if COLUMN_MAP['aid'] in df.columns:
        df[COLUMN_MAP['aid']] = (
            df[COLUMN_MAP['aid']]
            .astype(str)
            .str.replace(r'\.0$', '', regex=True)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
    
    return df


def filter_closing_trades(df):
    """篩選已平倉交易（CLOSING）"""
    action_col = COLUMN_MAP['action']
    if action_col in df.columns:
        return df[df[action_col] == 'CLOSING'].copy()
    return df


# ==================== MDD 計算函數 ====================
@st.cache_data(show_spinner=False)
def calculate_mdd(equity_series, initial_balance=0):
    """
    計算最大回撤 (Maximum Drawdown)
    
    參數:
        equity_series: 累計盈虧序列
        initial_balance: 初始資金（預設為 0）
    
    返回:
        mdd_value: 最大回撤百分比
        drawdown_series: 回撤序列
    """
    if len(equity_series) < 2:
        return 0.0, pd.Series([0.0])
    
    # 計算資產曲線
    cumulative_equity = initial_balance + equity_series.cumsum()
    
    # 計算歷史最高點
    running_max = cumulative_equity.cummax()
    
    # 計算回撤（處理分母為 0 的情況）
    drawdown = np.where(
        running_max != 0,
        (cumulative_equity - running_max) / running_max,
        0
    )
    
    # MDD 為最大負值
    mdd = np.min(drawdown)
    
    return mdd, pd.Series(drawdown, index=equity_series.index)


# ==================== 計算所有 AID 的統計數據 ====================
@st.cache_data(show_spinner=False)
def calculate_all_aid_stats(df, initial_balance, scalper_threshold_seconds):
    """計算所有 AID 的統計數據"""
    aid_col = COLUMN_MAP['aid']
    volume_col = COLUMN_MAP['volume']
    exec_col = COLUMN_MAP['execution_time']
    instrument_col = COLUMN_MAP['instrument']
    closed_pl_col = COLUMN_MAP['closed_pl']
    
    closing_df = filter_closing_trades(df)
    
    results = []
    
    for aid in closing_df[aid_col].unique():
        aid_data = closing_df[closing_df[aid_col] == aid].copy()
        
        # 基本統計
        net_pl = aid_data['Net_PL'].sum()
        trade_count = len(aid_data)
        trade_volume = aid_data[volume_col].sum() if volume_col in aid_data.columns else trade_count
        
        # 勝率
        wins = (aid_data['Net_PL'] > 0).sum()
        win_rate = (wins / trade_count * 100) if trade_count > 0 else 0
        
        # 平均持倉時間（秒）
        avg_hold_seconds = aid_data['Hold_Seconds'].mean() if 'Hold_Seconds' in aid_data.columns else 0
        avg_hold_seconds = avg_hold_seconds if pd.notna(avg_hold_seconds) else 0
        
        # Scalper 交易比例
        scalper_trades = aid_data[aid_data['Hold_Seconds'] < scalper_threshold_seconds]
        scalper_count = len(scalper_trades)
        scalper_ratio = (scalper_count / trade_count * 100) if trade_count > 0 else 0
        scalper_pl = scalper_trades['Net_PL'].sum() if not scalper_trades.empty else 0
        
        # MDD 計算
        aid_sorted = aid_data.sort_values(exec_col)
        if len(aid_sorted) >= 2:
            cumulative_pl = aid_sorted['Net_PL'].cumsum()
            equity = initial_balance + cumulative_pl
            running_max = equity.cummax()
            drawdown = np.where(
                running_max != 0,
                (equity - running_max) / running_max,
                0
            )
            mdd_pct = abs(np.min(drawdown) * 100)
        else:
            mdd_pct = 0.0
        
        # Profit Factor 計算
        gains = aid_data[aid_data[closed_pl_col] > 0][closed_pl_col].sum()
        losses = abs(aid_data[aid_data[closed_pl_col] < 0][closed_pl_col].sum())
        if losses == 0 and gains > 0:
            profit_factor = 5.0
        elif gains == 0:
            profit_factor = 0.0
        else:
            profit_factor = gains / losses
        
        # 主要交易品種
        if instrument_col in aid_data.columns and not aid_data[instrument_col].empty:
            main_symbol = aid_data[instrument_col].mode().iloc[0] if len(aid_data[instrument_col].mode()) > 0 else 'N/A'
        else:
            main_symbol = 'N/A'
        
        results.append({
            'AID': aid,
            'Net_PL': round(net_pl, 2),
            'Trade_Count': trade_count,
            'Trade_Volume': round(trade_volume, 2),
            'Win_Rate': round(win_rate, 2),
            'Avg_Hold_Seconds': round(avg_hold_seconds, 2),
            'MDD_Pct': round(mdd_pct, 2),
            'Profit_Factor': round(profit_factor, 2),
            'Scalper_Count': scalper_count,
            'Scalper_Ratio': round(scalper_ratio, 2),
            'Scalper_PL': round(scalper_pl, 2),
            'Main_Symbol': main_symbol
        })
    
    return pd.DataFrame(results)


# ==================== 圖表 1：累計盈虧走勢圖 ====================
@st.cache_data(show_spinner=False)
def create_cumulative_pnl_chart(df, initial_balance, scalper_threshold_seconds):
    """
    創建累計淨盈虧走勢圖：整體 vs. Scalper（無 MDD 陰影）
    """
    exec_col = COLUMN_MAP['execution_time']
    scalper_minutes = scalper_threshold_seconds / 60
    
    closing_df = filter_closing_trades(df)
    df_sorted = closing_df.sort_values(exec_col).copy()
    df_sorted['Date'] = df_sorted[exec_col].dt.date
    
    # 計算每日盈虧（整體）
    daily_pnl = df_sorted.groupby('Date')['Net_PL'].sum().reset_index()
    daily_pnl.columns = ['Date', 'Daily_PL']
    daily_pnl = daily_pnl.sort_values('Date')
    daily_pnl['Cumulative_PL'] = daily_pnl['Daily_PL'].cumsum()
    
    # 篩選 Scalper 交易
    scalper_df = df_sorted[df_sorted['Hold_Seconds'] < scalper_threshold_seconds].copy()
    
    if not scalper_df.empty:
        scalper_daily_pnl = scalper_df.groupby('Date')['Net_PL'].sum().reset_index()
        scalper_daily_pnl.columns = ['Date', 'Scalper_Daily_PL']
    else:
        scalper_daily_pnl = pd.DataFrame({'Date': daily_pnl['Date'], 'Scalper_Daily_PL': 0})
    
    # 合併數據
    merged_df = daily_pnl.merge(scalper_daily_pnl, on='Date', how='left')
    merged_df['Scalper_Daily_PL'] = merged_df['Scalper_Daily_PL'].fillna(0)
    merged_df['Scalper_Cumulative_PL'] = merged_df['Scalper_Daily_PL'].cumsum()
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    
    # 創建圖表（無 MDD 陰影）
    fig = go.Figure()
    
    # 整體累計盈虧
    fig.add_trace(go.Scatter(
        x=merged_df['Date'],
        y=merged_df['Cumulative_PL'],
        mode='lines+markers',
        name='整體累計盈虧',
        line=dict(color='#2E86AB', width=2.5),
        marker=dict(size=6),
        hovertemplate='<b>日期:</b> %{x|%Y-%m-%d}<br><b>整體累計:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    # Scalper 累計盈虧
    fig.add_trace(go.Scatter(
        x=merged_df['Date'],
        y=merged_df['Scalper_Cumulative_PL'],
        mode='lines+markers',
        name=f'Scalper 累計盈虧 (<{scalper_minutes:.0f}分鐘)',
        line=dict(color='#F39C12', width=2.5, dash='dot'),
        marker=dict(size=6, symbol='diamond'),
        hovertemplate='<b>日期:</b> %{x|%Y-%m-%d}<br><b>Scalper 累計:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    # Y=0 基準線
    fig.add_hline(
        y=0, 
        line_dash="dash", 
        line_color="gray", 
        line_width=1.5,
        annotation_text="損益平衡線",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=dict(
            text=f'📈 累計淨盈虧走勢：整體 vs. Scalper',
            font=dict(size=16)
        ),
        xaxis_title='日期',
        yaxis_title='累計淨盈虧 ($)',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified',
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    # 統計摘要
    total_pnl = merged_df['Cumulative_PL'].iloc[-1] if len(merged_df) > 0 else 0
    scalper_pnl = merged_df['Scalper_Cumulative_PL'].iloc[-1] if len(merged_df) > 0 else 0
    
    return fig, {'total_pnl': total_pnl, 'scalper_pnl': scalper_pnl}


# ==================== 圖表 2：小提琴圖 ====================
@st.cache_data(show_spinner=False)
def create_violin_plot(df):
    """
    創建小提琴圖 (Violin Plot)，Y 軸自動縮放至 1%-99% 區間
    """
    aid_col = COLUMN_MAP['aid']
    
    closing_df = filter_closing_trades(df)
    aid_pl = closing_df.groupby(aid_col)['Net_PL'].sum().reset_index()
    aid_pl.columns = ['AID', '累計淨盈虧']
    
    # 計算 1% 和 99% 百分位數
    Q1_percentile = aid_pl['累計淨盈虧'].quantile(0.01)
    Q99_percentile = aid_pl['累計淨盈虧'].quantile(0.99)
    
    # 過濾極端值
    plot_data = aid_pl[
        (aid_pl['累計淨盈虧'] >= Q1_percentile) & 
        (aid_pl['累計淨盈虧'] <= Q99_percentile)
    ].copy()
    
    filtered_count = len(aid_pl) - len(plot_data)
    
    # 計算統計值
    mean_val = plot_data['累計淨盈虧'].mean()
    median_val = plot_data['累計淨盈虧'].median()
    
    # 創建 Violin Plot
    fig = go.Figure()
    
    fig.add_trace(go.Violin(
        y=plot_data['累計淨盈虧'],
        name='盈虧分布',
        box_visible=True,
        meanline_visible=True,
        line_color='#2C3E50',
        fillcolor='rgba(52, 152, 219, 0.5)',
        opacity=0.8,
        points='all',
        pointpos=-0.8,
        jitter=0.3,
        marker=dict(
            color='#3498DB',
            size=5,
            opacity=0.5
        ),
        box=dict(
            visible=True,
            fillcolor='rgba(255, 255, 255, 0.8)',
            line=dict(color='#2C3E50', width=2)
        ),
        meanline=dict(
            visible=True,
            color='#E74C3C',
            width=2
        ),
        hoverinfo='y'
    ))
    
    # Y軸範圍：1%-99% 區間
    y_range = [Q1_percentile, Q99_percentile]
    y_padding = (y_range[1] - y_range[0]) * 0.1
    
    fig.update_layout(
        title=dict(
            text=f'🎻 盈虧分佈小提琴圖 (已過濾 {filtered_count} 位極端值，聚焦 1%-99% 區間)',
            font=dict(size=16)
        ),
        yaxis=dict(
            title='累計淨盈虧 ($)',
            range=[y_range[0] - y_padding, y_range[1] + y_padding],
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.3)',
            zerolinewidth=2
        ),
        xaxis=dict(showticklabels=False),
        height=500,
        showlegend=False,
        plot_bgcolor='rgba(248,249,250,1)',
        annotations=[
            dict(
                x=0.02, y=0.98,
                xref='paper', yref='paper',
                text=f'<b>統計摘要</b><br>'
                     f'客戶數: {len(plot_data):,}<br>'
                     f'平均值: ${mean_val:,.2f}<br>'
                     f'中位數: ${median_val:,.2f}',
                showarrow=False,
                font=dict(size=11),
                align='left',
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#3498DB',
                borderwidth=1,
                borderpad=6
            )
        ]
    )
    
    # 零線
    if y_range[0] <= 0 <= y_range[1]:
        fig.add_hline(y=0, line_dash="solid", line_color="rgba(0,0,0,0.5)", line_width=2)
    
    return fig


# ==================== 圖表 3：獲利因子分布 ====================
@st.cache_data(show_spinner=False)
def create_profit_factor_chart(aid_stats_df):
    """創建獲利因子分布圖（直方圖）"""
    pf_data = aid_stats_df[['AID', 'Profit_Factor']].copy()
    
    # 過濾異常值，聚焦 0-5 區間
    pf_display = pf_data[pf_data['Profit_Factor'] <= 5].copy()
    
    fig = px.histogram(
        pf_display,
        x='Profit_Factor',
        nbins=20,
        title='📊 獲利因子分布 (Profit Factor Distribution)',
        labels={'Profit_Factor': 'Profit Factor', 'count': '交易者數量'},
        color_discrete_sequence=['#3498DB']
    )
    
    # PF=1.0 分界線
    fig.add_vline(
        x=1.0, 
        line_dash="dash", 
        line_color="red", 
        line_width=2,
        annotation_text="PF=1.0 盈虧分界",
        annotation_position="top"
    )
    
    fig.update_layout(
        xaxis_title='Profit Factor (總獲利 / |總虧損|)',
        yaxis_title='交易者數量',
        height=400,
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    # 計算 PF > 1.0 的比例
    profitable_ratio = (pf_data['Profit_Factor'] > 1.0).sum() / len(pf_data) * 100 if len(pf_data) > 0 else 0
    
    return fig, profitable_ratio


# ==================== 圖表 4：風險回報矩陣 ====================
@st.cache_data(show_spinner=False)
def create_risk_return_scatter(aid_stats_df):
    """創建風險回報矩陣散佈圖 (X: MDD% 0-100%, Y: 總盈虧)"""
    scatter_df = aid_stats_df.copy()
    
    # 計算點的大小（基於交易量）
    min_size, max_size = 10, 50
    if scatter_df['Trade_Volume'].max() > scatter_df['Trade_Volume'].min():
        scatter_df['Size'] = min_size + (scatter_df['Trade_Volume'] - scatter_df['Trade_Volume'].min()) / \
                             (scatter_df['Trade_Volume'].max() - scatter_df['Trade_Volume'].min()) * (max_size - min_size)
    else:
        scatter_df['Size'] = 20
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=scatter_df['MDD_Pct'],
        y=scatter_df['Net_PL'],
        mode='markers',
        marker=dict(
            size=scatter_df['Size'],
            color=scatter_df['Net_PL'],
            colorscale=['#E74C3C', '#F39C12', '#27AE60'],
            showscale=True,
            colorbar=dict(title='Net P/L ($)')
        ),
        customdata=np.column_stack((
            scatter_df['AID'],
            scatter_df['Trade_Count'],
            scatter_df['Win_Rate']
        )),
        hovertemplate=(
            '<b>AID:</b> %{customdata[0]}<br>'
            '<b>淨盈虧:</b> $%{y:,.2f}<br>'
            '<b>MDD:</b> %{x:.2f}%<br>'
            '<b>交易筆數:</b> %{customdata[1]}<br>'
            '<b>勝率:</b> %{customdata[2]:.1f}%<extra></extra>'
        ),
        name='交易者'
    ))
    
    fig.update_layout(
        title='🎯 風險回報矩陣 (Risk-Return Matrix)',
        xaxis=dict(
            title='最大回撤 MDD (%)',
            range=[0, 100]
        ),
        yaxis_title='總盈虧 (Net P/L $)',
        height=550,
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    # 四象限分隔線
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.add_vline(x=50, line_dash="dash", line_color="gray", line_width=1)
    
    # 象限標註
    fig.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper",
                       text="🌟 明星交易員", showarrow=False, font=dict(size=12, color="green"))
    fig.add_annotation(x=0.95, y=0.95, xref="paper", yref="paper",
                       text="⚡ 激進型", showarrow=False, font=dict(size=12, color="orange"))
    fig.add_annotation(x=0.05, y=0.05, xref="paper", yref="paper",
                       text="🐢 守舊型", showarrow=False, font=dict(size=12, color="gray"))
    fig.add_annotation(x=0.95, y=0.05, xref="paper", yref="paper",
                       text="⚠️ 高風險", showarrow=False, font=dict(size=12, color="red"))
    
    return fig


# ==================== 圖表 5：持倉時間 vs 勝率 ====================
@st.cache_data(show_spinner=False)
def create_hold_time_vs_winrate(aid_stats_df, scalper_threshold_seconds):
    """創建持倉時間 vs 勝率散點圖"""
    scalper_minutes = scalper_threshold_seconds / 60
    
    plot_df = aid_stats_df[aid_stats_df['Avg_Hold_Seconds'] > 0].copy()
    
    if plot_df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=plot_df['Avg_Hold_Seconds'],
        y=plot_df['Win_Rate'],
        mode='markers',
        marker=dict(
            size=10,
            color=plot_df['Net_PL'],
            colorscale=['#E74C3C', '#F39C12', '#27AE60'],
            showscale=True,
            colorbar=dict(title='Net P/L ($)')
        ),
        customdata=np.column_stack((
            plot_df['AID'],
            plot_df['Trade_Count'],
            plot_df['Net_PL']
        )),
        hovertemplate=(
            '<b>AID:</b> %{customdata[0]}<br>'
            '<b>平均持倉秒數:</b> %{x:,.0f}<br>'
            '<b>勝率:</b> %{y:.1f}%<br>'
            '<b>交易筆數:</b> %{customdata[1]}<br>'
            '<b>淨盈虧:</b> $%{customdata[2]:,.2f}<extra></extra>'
        ),
        name='交易者'
    ))
    
    # Scalper 閾值垂直虛線
    fig.add_vline(
        x=scalper_threshold_seconds,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Scalper 定義 ({scalper_minutes:.0f}分鐘)",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=f'⏱️ 持倉時間 vs 勝率關聯分析',
        xaxis_title='平均持倉秒數',
        yaxis=dict(
            title='勝率 (%)',
            range=[0, 100]
        ),
        height=500,
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    return fig


# ==================== 圖表 6：每日盈虧柱狀圖 ====================
@st.cache_data(show_spinner=False)
def create_daily_pnl_chart(df):
    """創建每日盈虧柱狀圖"""
    exec_col = COLUMN_MAP['execution_time']
    
    closing_df = filter_closing_trades(df)
    df_daily = closing_df.copy()
    df_daily['Date'] = df_daily[exec_col].dt.date
    daily_pnl = df_daily.groupby('Date')['Net_PL'].sum().reset_index()
    daily_pnl.columns = ['日期', '每日盈虧']
    daily_pnl = daily_pnl.sort_values('日期')
    
    # 設定顏色
    colors = ['#27AE60' if x > 0 else '#E74C3C' for x in daily_pnl['每日盈虧']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=daily_pnl['日期'],
        y=daily_pnl['每日盈虧'],
        marker_color=colors,
        name='每日盈虧',
        hovertemplate='<b>日期:</b> %{x}<br><b>淨盈虧:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_color="black", line_width=1)
    
    fig.update_layout(
        title='📅 每日盈虧柱狀圖',
        xaxis_title='日期',
        yaxis_title='淨盈虧 ($)',
        height=400,
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    return fig


# ==================== Top 10 Scalpers 統計表 ====================
@st.cache_data(show_spinner=False)
def get_top_scalpers(aid_stats_df, n=10):
    """取得 Top 10 Scalpers（按 Scalper 交易筆數排序）"""
    scalpers = aid_stats_df[aid_stats_df['Scalper_Count'] > 0].copy()
    
    if scalpers.empty:
        return pd.DataFrame()
    
    top_scalpers = scalpers.nlargest(n, 'Scalper_Count')[
        ['AID', 'Scalper_Count', 'Scalper_PL', 'Win_Rate', 'Avg_Hold_Seconds', 'Main_Symbol']
    ].copy()
    
    top_scalpers.columns = ['AID', '交易筆數', '總盈虧', '勝率(%)', '平均持倉秒數', '主要品種']
    top_scalpers['總盈虧'] = top_scalpers['總盈虧'].apply(lambda x: f"${x:,.2f}")
    top_scalpers['平均持倉秒數'] = top_scalpers['平均持倉秒數'].round(1)
    top_scalpers['勝率(%)'] = top_scalpers['勝率(%)'].round(2)
    
    return top_scalpers


# ==================== 個別客戶分析函數 ====================
@st.cache_data(show_spinner=False)
def get_client_details(df, aid, initial_balance, scalper_threshold_seconds):
    """取得單一客戶的詳細數據"""
    aid_col = COLUMN_MAP['aid']
    exec_col = COLUMN_MAP['execution_time']
    instrument_col = COLUMN_MAP['instrument']
    
    closing_df = filter_closing_trades(df)
    client_df = closing_df[closing_df[aid_col] == str(aid)].copy()
    
    if client_df.empty:
        return None
    
    # 基本統計
    net_pl = client_df['Net_PL'].sum()
    trade_count = len(client_df)
    wins = (client_df['Net_PL'] > 0).sum()
    win_rate = (wins / trade_count * 100) if trade_count > 0 else 0
    avg_hold_seconds = client_df['Hold_Seconds'].mean()
    avg_hold_seconds = avg_hold_seconds if pd.notna(avg_hold_seconds) else 0
    
    # 累計盈虧序列
    client_sorted = client_df.sort_values(exec_col).copy()
    client_sorted['Cumulative_PL'] = client_sorted['Net_PL'].cumsum()
    
    # Scalper 累計盈虧
    scalper_mask = client_sorted['Hold_Seconds'] < scalper_threshold_seconds
    client_sorted['Scalper_PL'] = np.where(scalper_mask, client_sorted['Net_PL'], 0)
    client_sorted['Scalper_Cumulative_PL'] = client_sorted['Scalper_PL'].cumsum()
    
    # Symbol 分佈
    if instrument_col in client_df.columns:
        symbol_dist = client_df.groupby(instrument_col)['Net_PL'].count().reset_index()
        symbol_dist.columns = ['Symbol', 'Count']
    else:
        symbol_dist = pd.DataFrame()
    
    # 持倉時間分佈
    hold_times = client_df['Hold_Seconds'].dropna()
    
    return {
        'net_pl': net_pl,
        'trade_count': trade_count,
        'win_rate': win_rate,
        'avg_hold_seconds': avg_hold_seconds,
        'cumulative_df': client_sorted[[exec_col, 'Cumulative_PL', 'Scalper_Cumulative_PL']],
        'symbol_dist': symbol_dist,
        'hold_times': hold_times
    }


def create_client_cumulative_chart(cumulative_df, scalper_minutes):
    """創建客戶累計盈虧走勢圖"""
    exec_col = COLUMN_MAP['execution_time']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cumulative_df[exec_col],
        y=cumulative_df['Cumulative_PL'],
        mode='lines',
        name='累計總盈虧',
        line=dict(color='#2E86AB', width=2),
        hovertemplate='<b>時間:</b> %{x}<br><b>累計盈虧:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=cumulative_df[exec_col],
        y=cumulative_df['Scalper_Cumulative_PL'],
        mode='lines',
        name=f'累計 Scalper 盈虧 (<{scalper_minutes}分鐘)',
        line=dict(color='#F39C12', width=2, dash='dot'),
        hovertemplate='<b>時間:</b> %{x}<br><b>Scalper 盈虧:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    
    fig.update_layout(
        title='📈 個人累計盈虧走勢（整體 vs Scalper）',
        xaxis_title='時間',
        yaxis_title='累計盈虧 ($)',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    return fig


def create_symbol_pie_chart(symbol_dist):
    """創建產品分佈圓餅圖"""
    if symbol_dist.empty:
        return None
    
    fig = px.pie(
        symbol_dist,
        values='Count',
        names='Symbol',
        title='🥧 交易品種分佈',
        hole=0.3
    )
    
    fig.update_layout(height=400)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig


def create_hold_time_histogram(hold_times, scalper_threshold_seconds):
    """創建持倉時間直方圖"""
    scalper_minutes = scalper_threshold_seconds / 60
    
    # 轉換為分鐘
    hold_minutes = hold_times / 60
    
    fig = px.histogram(
        x=hold_minutes,
        nbins=30,
        title='⏱️ 持倉時間分佈',
        labels={'x': '持倉時間 (分鐘)', 'count': '交易筆數'},
        color_discrete_sequence=['#3498DB']
    )
    
    # Scalper 閾值紅線
    fig.add_vline(
        x=scalper_minutes,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Scalper 定義 ({scalper_minutes}分鐘)",
        annotation_position="top"
    )
    
    fig.update_layout(
        xaxis_title='持倉時間 (分鐘)',
        yaxis_title='交易筆數',
        height=400,
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    return fig


# ==================== 導出 Excel ====================
def export_to_excel(df, aid_stats_df, initial_balance, scalper_threshold_seconds):
    """
    導出完整分析數據到 Excel（多分頁）
    
    分頁結構:
        Sheet 1 (Summary): 數據摘要
        Sheet 2 (Risk_Return): 風險回報清單
        Sheet 3 (Scalper_List): Scalper 清單
    """
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    
    output = BytesIO()
    
    closing_df = filter_closing_trades(df)
    aid_col = COLUMN_MAP['aid']
    scalper_minutes = scalper_threshold_seconds / 60
    
    # ========== Sheet 1: Summary ==========
    total_trades = len(df)
    closing_trades = len(closing_df)
    unique_clients = df[aid_col].nunique()
    total_net_pl = closing_df['Net_PL'].sum()
    profitable_clients = (aid_stats_df['Net_PL'] > 0).sum()
    losing_clients = (aid_stats_df['Net_PL'] <= 0).sum()
    
    scalper_count = (aid_stats_df['Scalper_Ratio'] > 50).sum()
    
    summary_data = [
        ['指標', '數值', '說明'],
        ['總交易筆數', total_trades, '所有交易記錄'],
        ['平倉交易筆數', closing_trades, 'CLOSING 類型交易'],
        ['總客戶數', unique_clients, '不重複的 AID 數量'],
        ['總淨盈虧', round(total_net_pl, 2), 'Net_PL 總和'],
        ['', '', ''],
        ['盈利客戶數', profitable_clients, '累計 Net_PL > 0'],
        ['虧損客戶數', losing_clients, '累計 Net_PL <= 0'],
        ['', '', ''],
        [f'Scalper 數量 (<{scalper_minutes:.0f}分鐘)', scalper_count, 'Scalper 比例 > 50%'],
        ['初始資金設定', initial_balance, '用於 MDD 計算'],
        ['Scalper 閾值(秒)', scalper_threshold_seconds, '持倉時間閾值'],
        ['報告生成時間', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '']
    ]
    
    summary_df = pd.DataFrame(summary_data[1:], columns=summary_data[0])
    
    # ========== Sheet 2: Risk_Return ==========
    risk_return_df = aid_stats_df[[
        'AID', 'Net_PL', 'MDD_Pct', 'Trade_Count', 'Trade_Volume',
        'Win_Rate', 'Avg_Hold_Seconds', 'Profit_Factor', 'Scalper_Ratio'
    ]].copy()
    risk_return_df.columns = [
        'AID', 'Net_PL', 'MDD(%)', 'Trade_Count', 'Trade_Volume',
        'Win_Rate(%)', 'Avg_Hold_Seconds', 'Profit_Factor', 'Scalper_Ratio(%)'
    ]
    risk_return_df = risk_return_df.sort_values('Net_PL', ascending=False)
    
    # ========== Sheet 3: Scalper_List ==========
    scalper_list = aid_stats_df[aid_stats_df['Scalper_Count'] > 0][[
        'AID', 'Scalper_Count', 'Scalper_PL', 'Win_Rate', 'Avg_Hold_Seconds', 'Main_Symbol'
    ]].copy()
    scalper_list.columns = ['AID', 'Scalper交易數', 'Scalper盈虧', '勝率(%)', '平均持倉秒數', '主要品種']
    scalper_list = scalper_list.sort_values('Scalper交易數', ascending=False)
    
    # ========== 寫入 Excel ==========
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        risk_return_df.to_excel(writer, sheet_name='Risk_Return', index=False)
        scalper_list.to_excel(writer, sheet_name='Scalper_List', index=False)
        
        # 格式化
        workbook = writer.book
        header_font = Font(bold=True, color='FFFFFF')
        header_fill = PatternFill(start_color='2E86AB', end_color='2E86AB', fill_type='solid')
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            
            for column in ws.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                ws.column_dimensions[column_letter].width = adjusted_width
            
            for cell in ws[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal='center')
                cell.border = thin_border
    
    output.seek(0)
    return output


# ==================== 主程式 ====================
def main():
    st.title("📊 交易數據分析系統")
    st.markdown("**支持大規模交易數據的處理與分析**")
    
    # ==================== 側邊欄 ====================
    with st.sidebar:
        st.header("📁 數據上傳")
        uploaded_files = st.file_uploader(
            "上傳交易數據檔案 (.xlsx 或 .csv)",
            type=['xlsx', 'csv'],
            accept_multiple_files=True
        )
        
        st.header("⚙️ 全域參數設定")
        
        initial_balance = st.number_input(
            "初始資金",
            value=10000,
            min_value=0,
            step=1000,
            help="設定每位交易者的初始資金，用於計算 MDD"
        )
        
        scalper_minutes = st.number_input(
            "Scalper 持倉定義 (分鐘)",
            value=5,
            min_value=1,
            max_value=60,
            step=1,
            help="持倉時間小於此值的交易將被歸類為 Scalp 交易"
        )
        
        scalper_threshold_seconds = scalper_minutes * 60
        
        if uploaded_files:
            st.success(f"已上傳 {len(uploaded_files)} 個檔案")
            st.markdown("---")
            st.info(f"💰 初始資金: **${initial_balance:,}**")
            st.info(f"⏱️ Scalper 定義: **<{scalper_minutes} 分鐘**")
    
    # ==================== 主內容區 ====================
    if not uploaded_files:
        st.info("👈 請在左側上傳交易數據檔案開始分析")
        
        st.markdown("""
        ### 📋 功能說明
        
        **Tab 1 - 整體數據概覽**
        1. 累計盈虧走勢圖（整體 vs Scalper）
        2. 盈虧分佈小提琴圖
        3. 獲利因子分布圖
        4. 風險回報矩陣
        5. 持倉時間 vs 勝率關聯分析
        6. 每日盈虧柱狀圖
        7. Top 10 Scalpers 統計表
        
        **Tab 2 - 個別客戶探查**
        - 選擇 AID 查看個人詳細數據
        - 個人累計走勢圖
        - 交易品種圓餅圖
        - 持倉時間直方圖
        """)
        return
    
    # 載入數據
    with st.spinner("正在載入和處理數據..."):
        df = load_and_preprocess(uploaded_files)
    
    if df is None or df.empty:
        st.error("無法載入數據，請檢查檔案格式")
        return
    
    # 計算所有 AID 統計
    with st.spinner("正在計算統計數據..."):
        aid_stats_df = calculate_all_aid_stats(df, initial_balance, scalper_threshold_seconds)
    
    # ==================== 數據摘要 ====================
    st.markdown("---")
    closing_df = filter_closing_trades(df)
    aid_col = COLUMN_MAP['aid']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("總交易筆數", f"{len(df):,}")
    with col2:
        st.metric("平倉交易筆數", f"{len(closing_df):,}")
    with col3:
        st.metric("交易者數量", f"{df[aid_col].nunique():,}")
    with col4:
        total_pnl = closing_df['Net_PL'].sum()
        st.metric("總淨盈虧", f"${total_pnl:,.2f}")
    
    # ==================== 側邊欄下載按鈕 ====================
    with st.sidebar:
        st.markdown("---")
        st.header("📥 下載報表")
        
        excel_data = export_to_excel(df, aid_stats_df, initial_balance, scalper_threshold_seconds)
        
        st.download_button(
            label="📊 下載 Excel 報表",
            data=excel_data,
            file_name=f"trading_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="包含 Summary, Risk_Return, Scalper_List 三個分頁",
            type="primary"
        )
    
    # ==================== Tabs ====================
    tab1, tab2 = st.tabs(["📊 整體數據概覽", "👤 個別客戶探查"])
    
    # ==================== Tab 1: 整體數據概覽 ====================
    with tab1:
        st.header("📊 整體數據概覽")
        
        # 1. 累計盈虧走勢圖
        st.markdown("### 1️⃣ 累計淨盈虧走勢（整體 vs Scalper）")
        cumulative_fig, pnl_stats = create_cumulative_pnl_chart(df, initial_balance, scalper_threshold_seconds)
        st.plotly_chart(cumulative_fig, use_container_width=True)
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("整體淨盈虧", f"${pnl_stats['total_pnl']:,.2f}")
        with col_stat2:
            st.metric(f"Scalper 淨盈虧 (<{scalper_minutes}分鐘)", f"${pnl_stats['scalper_pnl']:,.2f}")
        
        st.markdown("---")
        
        # 2. 小提琴圖
        st.markdown("### 2️⃣ 盈虧分佈小提琴圖 (Violin Plot)")
        violin_fig = create_violin_plot(df)
        st.plotly_chart(violin_fig, use_container_width=True)
        
        st.markdown("---")
        
        # 3. 獲利因子分布
        st.markdown("### 3️⃣ 獲利因子分布 (Profit Factor)")
        pf_fig, profitable_ratio = create_profit_factor_chart(aid_stats_df)
        st.plotly_chart(pf_fig, use_container_width=True)
        st.success(f"📈 **PF > 1.0 的交易者佔比: {profitable_ratio:.1f}%** (盈利者)")
        
        st.markdown("---")
        
        # 4. 風險回報矩陣
        st.markdown("### 4️⃣ 風險回報矩陣 (Risk-Return Matrix)")
        scatter_fig = create_risk_return_scatter(aid_stats_df)
        st.plotly_chart(scatter_fig, use_container_width=True)
        
        st.markdown("""
        **象限說明：**
        - 🌟 **左上 (Low MDD, High P/L)**: 明星交易員
        - ⚡ **右上 (High MDD, High P/L)**: 激進型交易員
        - 🐢 **左下 (Low MDD, Low P/L)**: 守舊型交易員
        - ⚠️ **右下 (High MDD, Low P/L)**: 高風險交易員
        """)
        
        st.markdown("---")
        
        # 5. 持倉時間 vs 勝率
        st.markdown("### 5️⃣ 持倉時間 vs 勝率關聯分析")
        hold_winrate_fig = create_hold_time_vs_winrate(aid_stats_df, scalper_threshold_seconds)
        if hold_winrate_fig:
            st.plotly_chart(hold_winrate_fig, use_container_width=True)
        else:
            st.warning("無持倉時間數據可供分析")
        
        st.markdown("---")
        
        # 6. 每日盈虧柱狀圖
        st.markdown("### 6️⃣ 每日盈虧柱狀圖")
        daily_fig = create_daily_pnl_chart(df)
        st.plotly_chart(daily_fig, use_container_width=True)
        
        st.markdown("---")
        
        # 7. Top 10 Scalpers 統計表
        st.markdown(f"### 7️⃣ Top 10 Scalpers 統計表 (定義: <{scalper_minutes} 分鐘)")
        top_scalpers = get_top_scalpers(aid_stats_df)
        if not top_scalpers.empty:
            st.dataframe(top_scalpers, use_container_width=True, hide_index=True)
        else:
            st.info("無符合條件的 Scalper 數據")
    
    # ==================== Tab 2: 個別客戶探查 ====================
    with tab2:
        st.header("👤 個別客戶探查")
        
        # 取得所有 AID 列表
        all_aids = sorted(aid_stats_df['AID'].unique().tolist())
        
        # 搜尋器
        selected_aid = st.selectbox(
            "🔍 選擇或搜尋 AID",
            options=all_aids,
            index=0 if all_aids else None,
            help="輸入 AID 進行搜尋"
        )
        
        if selected_aid:
            # 取得客戶詳細數據
            client_data = get_client_details(df, selected_aid, initial_balance, scalper_threshold_seconds)
            
            if client_data:
                st.markdown("---")
                
                # 個人指標
                st.markdown(f"### 📋 AID: {selected_aid} 的統計指標")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("總盈虧", f"${client_data['net_pl']:,.2f}")
                with col2:
                    st.metric("勝率", f"{client_data['win_rate']:.1f}%")
                with col3:
                    st.metric("交易筆數", f"{client_data['trade_count']:,}")
                with col4:
                    avg_hold_min = client_data['avg_hold_seconds'] / 60
                    st.metric("平均持倉時間", f"{avg_hold_min:.1f} 分鐘")
                
                st.markdown("---")
                
                # 個人走勢圖
                st.markdown("### 📈 累計盈虧走勢")
                client_chart = create_client_cumulative_chart(
                    client_data['cumulative_df'], 
                    scalper_minutes
                )
                st.plotly_chart(client_chart, use_container_width=True)
                
                st.markdown("---")
                
                # 產品分佈 & 持倉分佈
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("### 🥧 產品分佈")
                    pie_chart = create_symbol_pie_chart(client_data['symbol_dist'])
                    if pie_chart:
                        st.plotly_chart(pie_chart, use_container_width=True)
                    else:
                        st.info("無產品分佈數據")
                
                with col_right:
                    st.markdown("### ⏱️ 持倉時間分佈")
                    if len(client_data['hold_times']) > 0:
                        hist_chart = create_hold_time_histogram(
                            client_data['hold_times'],
                            scalper_threshold_seconds
                        )
                        st.plotly_chart(hist_chart, use_container_width=True)
                    else:
                        st.info("無持倉時間數據")
            else:
                st.warning(f"找不到 AID: {selected_aid} 的數據")
        else:
            st.info("請選擇一個 AID 查看詳細分析")


if __name__ == "__main__":
    main()
