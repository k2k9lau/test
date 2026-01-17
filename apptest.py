"""
交易數據分析系統 (Trading Analysis System)
支持大規模交易數據（十萬筆以上）的處理與分析

第一階段重構：整合核心數據引擎
- 高效快取機制
- 向量化運算（禁止 apply/loop）
- AID 強制字串化（解決複製失效）
- 大數據採樣優化
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from io import BytesIO
from typing import Tuple, Optional, Dict, Any

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


# ==================== 核心數據引擎（第一階段重構）====================

@st.cache_data(show_spinner=False, ttl=3600)
def load_data(uploaded_files) -> Optional[pd.DataFrame]:
    """
    高效載入並預處理交易數據
    
    特性：
    - 使用 @st.cache_data 快取，避免重複載入
    - 支持多檔案合併
    - 強制 AID 為字串類型（解決複製問題）
    - 向量化時間與盈虧計算
    """
    if not uploaded_files:
        return None
    
    dfs = []
    
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, dtype={COLUMN_MAP['aid']: str})
            else:
                df = pd.read_excel(uploaded_file, dtype={COLUMN_MAP['aid']: str})
            dfs.append(df)
        except Exception as e:
            st.error(f"讀取檔案 {uploaded_file.name} 時發生錯誤: {e}")
            continue
    
    if not dfs:
        return None
    
    # 向量化合併所有數據
    df = pd.concat(dfs, ignore_index=True)
    
    # 數據清洗流程
    df = _clean_data(df)
    
    return df


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    數據清洗（內部函數）
    
    執行順序：
    1. 移除 Total 行
    2. 去重
    3. 強制 AID 為字串（關鍵！解決複製失效問題）
    4. 轉換時間欄位
    5. 填充空值
    6. 計算 Net_PL 和 Hold_Seconds
    """
    exec_col = COLUMN_MAP['execution_time']
    aid_col = COLUMN_MAP['aid']
    
    # 1. 移除 Total 行
    if exec_col in df.columns:
        df = df[df[exec_col] != 'Total'].copy()
    
    # 2. 去重
    df = df.drop_duplicates()
    
    # 3. 【關鍵】強制 AID 為字串類型，移除浮點數 .0 後綴
    if aid_col in df.columns:
        df[aid_col] = (
            df[aid_col]
            .astype(str)
            .str.replace(r'\.0$', '', regex=True)
            .str.strip()
        )
    
    # 4. 向量化轉換時間欄位
    for col_key in ['execution_time', 'open_time']:
        col_name = COLUMN_MAP[col_key]
        if col_name in df.columns:
            df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
    
    # 5. 向量化填充空值（盈虧與費用）
    numeric_cols = ['closed_pl', 'commission', 'swap']
    for col_key in numeric_cols:
        col_name = COLUMN_MAP[col_key]
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
    
    # 6. 向量化計算 Net_PL
    df['Net_PL'] = (
        df[COLUMN_MAP['closed_pl']] + 
        df[COLUMN_MAP['commission']] + 
        df[COLUMN_MAP['swap']]
    )
    
    # 7. 向量化計算 Hold_Seconds
    exec_time = df[COLUMN_MAP['execution_time']]
    open_time = df[COLUMN_MAP['open_time']]
    
    valid_mask = pd.notna(exec_time) & pd.notna(open_time)
    df['Hold_Seconds'] = np.where(
        valid_mask,
        (exec_time - open_time).dt.total_seconds(),
        np.nan
    )
    
    return df


def filter_closing_trades(df: pd.DataFrame) -> pd.DataFrame:
    """篩選已平倉交易（CLOSING）"""
    action_col = COLUMN_MAP['action']
    if action_col in df.columns:
        return df[df[action_col] == 'CLOSING'].copy()
    return df.copy()


@st.cache_data(show_spinner=False)
def get_client_metrics(
    df: pd.DataFrame, 
    initial_balance: float = 10000,
    scalper_threshold_seconds: int = 300
) -> pd.DataFrame:
    """
    向量化計算客戶指標（禁止 apply/loop）
    
    計算指標：
    - Total_PL, Scalp_PL, Scalp_Pct, Win_Rate
    - Sharpe_Ratio, MDD_Pct, PL_Q1, PL_Median, PL_Q3
    """
    aid_col = COLUMN_MAP['aid']
    
    # 確保 AID 為字串
    df = df.copy()
    df[aid_col] = df[aid_col].astype(str)
    
    # 篩選平倉交易
    closing_df = filter_closing_trades(df)
    
    if closing_df.empty:
        return pd.DataFrame()
    
    # ========== 基礎聚合（向量化 groupby）==========
    basic_stats = closing_df.groupby(aid_col, as_index=False).agg(
        Total_PL=('Net_PL', 'sum'),
        Trade_Count=('Net_PL', 'count'),
        Avg_PL=('Net_PL', 'mean'),
        Std_PL=('Net_PL', 'std')
    )
    
    # 勝率計算
    closing_df = closing_df.copy()
    closing_df['_is_win'] = (closing_df['Net_PL'] > 0).astype(int)
    win_stats = closing_df.groupby(aid_col, as_index=False).agg(
        Win_Count=('_is_win', 'sum')
    )
    
    metrics = basic_stats.merge(win_stats, on=aid_col, how='left')
    metrics['Win_Rate'] = (metrics['Win_Count'] / metrics['Trade_Count'] * 100).round(2)
    
    # Sharpe Ratio
    metrics['Sharpe_Ratio'] = np.where(
        metrics['Std_PL'] > 0,
        (metrics['Avg_PL'] / metrics['Std_PL']).round(4),
        0
    )
    
    # 分位數
    quantile_stats = closing_df.groupby(aid_col)['Net_PL'].quantile([0.25, 0.5, 0.75]).unstack()
    quantile_stats.columns = ['PL_Q1', 'PL_Median', 'PL_Q3']
    quantile_stats = quantile_stats.reset_index()
    metrics = metrics.merge(quantile_stats, on=aid_col, how='left')
    
    # ========== Scalp 相關指標 ==========
    scalp_df = closing_df[closing_df['Hold_Seconds'] < scalper_threshold_seconds]
    if not scalp_df.empty:
        scalp_agg = scalp_df.groupby(aid_col, as_index=False).agg(
            Scalp_Count=('Net_PL', 'count'),
            Scalp_PL=('Net_PL', 'sum')
        )
        metrics = metrics.merge(scalp_agg, on=aid_col, how='left')
    
    metrics['Scalp_Count'] = metrics.get('Scalp_Count', 0).fillna(0).astype(int)
    metrics['Scalp_PL'] = metrics.get('Scalp_PL', 0).fillna(0)
    metrics['Scalp_Pct'] = (metrics['Scalp_Count'] / metrics['Trade_Count'] * 100).round(2)
    
    # ========== MDD% 計算 ==========
    mdd_series = _calculate_mdd_vectorized(closing_df, aid_col, initial_balance)
    mdd_df = mdd_series.reset_index()
    mdd_df.columns = [aid_col, 'MDD_Pct']
    metrics = metrics.merge(mdd_df, on=aid_col, how='left')
    metrics['MDD_Pct'] = metrics['MDD_Pct'].fillna(0)
    
    # 強制 AID 為字串
    metrics[aid_col] = metrics[aid_col].astype(str)
    
    output_cols = [
        aid_col, 'Total_PL', 'Scalp_PL', 'Scalp_Pct', 
        'Win_Rate', 'Sharpe_Ratio', 'MDD_Pct',
        'PL_Q1', 'PL_Median', 'PL_Q3', 'Trade_Count'
    ]
    
    for col in output_cols:
        if col not in metrics.columns:
            metrics[col] = 0
    
    return metrics[output_cols].round(2)


def _calculate_mdd_vectorized(
    df: pd.DataFrame, 
    aid_col: str, 
    initial_balance: float
) -> pd.Series:
    """向量化計算每個客戶的 MDD%"""
    exec_col = COLUMN_MAP['execution_time']
    
    df_sorted = df.sort_values([aid_col, exec_col]).copy()
    df_sorted['_cumsum'] = df_sorted.groupby(aid_col)['Net_PL'].cumsum()
    df_sorted['_equity'] = initial_balance + df_sorted['_cumsum']
    df_sorted['_running_max'] = df_sorted.groupby(aid_col)['_equity'].cummax()
    
    df_sorted['_drawdown'] = np.where(
        df_sorted['_running_max'] != 0,
        (df_sorted['_equity'] - df_sorted['_running_max']) / df_sorted['_running_max'],
        0
    )
    
    mdd_series = df_sorted.groupby(aid_col)['_drawdown'].min().abs() * 100
    return mdd_series.round(2)


@st.cache_data(show_spinner=False)
def get_client_summary_for_violin(
    df: pd.DataFrame,
    max_clients: int = 5000,
    sample_rate: float = 0.1
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    為 Violin Plot 準備客戶層級摘要數據（含採樣）
    """
    aid_col = COLUMN_MAP['aid']
    
    df = df.copy()
    df[aid_col] = df[aid_col].astype(str)
    
    # 向量化計算每位客戶的累計盈虧
    client_pl = df.groupby(aid_col, as_index=False)['Net_PL'].sum()
    client_pl.columns = ['AID', '累計淨盈虧']
    
    # 強制 AID 為字串
    client_pl['AID'] = client_pl['AID'].astype(str)
    
    n_clients = len(client_pl)
    
    sampling_info = {
        'original_count': n_clients,
        'sampled': False,
        'sampled_count': n_clients,
        'sample_rate': 1.0
    }
    
    if n_clients <= max_clients:
        return client_pl, sampling_info
    
    # 採樣
    sampled_clients = client_pl.sample(frac=sample_rate, random_state=42)
    
    sampling_info.update({
        'sampled': True,
        'sampled_count': len(sampled_clients),
        'sample_rate': sample_rate
    })
    
    return sampled_clients, sampling_info


# ==================== MDD 計算函數（向量化版本）====================

def calculate_mdd(equity_series, initial_balance=0):
    """計算最大回撤 (Maximum Drawdown)"""
    if len(equity_series) < 2:
        return 0.0, pd.Series([0.0])
    
    cumulative_equity = initial_balance + equity_series.cumsum()
    running_max = cumulative_equity.cummax()
    
    drawdown = np.where(
        running_max != 0,
        (cumulative_equity - running_max) / running_max,
        0
    )
    
    mdd = np.min(drawdown)
    return mdd, pd.Series(drawdown, index=equity_series.index)


# ==================== 當日分析函數 ====================

def get_daily_analysis(df, scalper_threshold_seconds=300):
    """取得當日分析數據（向量化優化）"""
    exec_col = COLUMN_MAP['execution_time']
    aid_col = COLUMN_MAP['aid']
    instrument_col = COLUMN_MAP['instrument']
    
    closing_df = filter_closing_trades(df)
    
    if closing_df.empty:
        return None, None
    
    # 找出最新日期作為「當日」
    latest_date = closing_df[exec_col].max().date()
    daily_df = closing_df[closing_df[exec_col].dt.date == latest_date].copy()
    
    if daily_df.empty:
        return None, None
    
    # Top 10 Profit 客戶（向量化）
    top_profit = daily_df.groupby(aid_col, as_index=False).agg(
        當日總盈虧=('Net_PL', 'sum')
    )
    top_profit = top_profit.nlargest(10, '當日總盈虧')
    top_profit.columns = ['AID', '當日總盈虧']
    top_profit['AID'] = top_profit['AID'].astype(str)
    
    # Top 10 Scalpers（向量化）
    scalp_df = daily_df[daily_df['Hold_Seconds'] < scalper_threshold_seconds].copy()
    
    if not scalp_df.empty:
        # 向量化聚合
        scalper_stats = scalp_df.groupby(aid_col, as_index=False).agg(
            交易筆數=('Net_PL', 'count'),
            當日總盈虧=('Net_PL', 'sum'),
            平均單筆盈虧=('Net_PL', 'mean'),
            平均持倉秒數=('Hold_Seconds', 'mean')
        )
        
        # 計算勝率（向量化）
        scalp_df['_is_win'] = (scalp_df['Net_PL'] > 0).astype(int)
        win_rate_df = scalp_df.groupby(aid_col, as_index=False).agg(
            _wins=('_is_win', 'sum'),
            _total=('_is_win', 'count')
        )
        win_rate_df['勝率(%)'] = (win_rate_df['_wins'] / win_rate_df['_total'] * 100).round(2)
        
        scalper_stats = scalper_stats.merge(win_rate_df[[aid_col, '勝率(%)']], on=aid_col, how='left')
        
        # 主要交易品種（使用 transform + mode 向量化替代）
        if instrument_col in scalp_df.columns:
            mode_df = scalp_df.groupby(aid_col)[instrument_col].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A'
            ).reset_index()
            mode_df.columns = [aid_col, '主要交易品種']
            scalper_stats = scalper_stats.merge(mode_df, on=aid_col, how='left')
        else:
            scalper_stats['主要交易品種'] = 'N/A'
        
        # 取交易筆數最多的前 10 名
        top_scalpers = scalper_stats.nlargest(10, '交易筆數')
        top_scalpers = top_scalpers[[aid_col, '交易筆數', '平均持倉秒數', '當日總盈虧', '勝率(%)', '主要交易品種']]
        top_scalpers.columns = ['AID', '交易筆數', '平均持倉秒數', '當日總盈虧', '勝率(%)', '主要交易品種']
        top_scalpers['AID'] = top_scalpers['AID'].astype(str)
        top_scalpers['平均持倉秒數'] = top_scalpers['平均持倉秒數'].round(1)
    else:
        top_scalpers = pd.DataFrame()
    
    return top_profit, top_scalpers, latest_date


# ==================== 30天分析函數 ====================

def get_30day_analysis(df):
    """取得30天分析數據"""
    exec_col = COLUMN_MAP['execution_time']
    
    closing_df = filter_closing_trades(df)
    
    if closing_df.empty:
        return None
    
    latest_date = closing_df[exec_col].max()
    start_date = latest_date - timedelta(days=30)
    
    df_30d = closing_df[closing_df[exec_col] >= start_date].copy()
    
    return df_30d, start_date, latest_date


# ==================== 圖表函數（向量化優化）====================

def create_cumulative_pnl_chart(df, initial_balance=0, scalper_threshold_seconds=300):
    """創建累計淨盈虧走勢圖：整體 vs. Scalper"""
    exec_col = COLUMN_MAP['execution_time']
    scalper_minutes = scalper_threshold_seconds / 60
    
    df_sorted = df.sort_values(exec_col).copy()
    df_sorted['Date'] = df_sorted[exec_col].dt.date
    
    # 向量化計算每日盈虧
    daily_pnl = df_sorted.groupby('Date', as_index=False)['Net_PL'].sum()
    daily_pnl.columns = ['Date', 'Daily_PL']
    daily_pnl = daily_pnl.sort_values('Date')
    daily_pnl['Cumulative_PL'] = initial_balance + daily_pnl['Daily_PL'].cumsum()
    
    # Scalper 每日盈虧
    scalper_df = df_sorted[df_sorted['Hold_Seconds'] < scalper_threshold_seconds]
    
    if not scalper_df.empty:
        scalper_daily_pnl = scalper_df.groupby('Date', as_index=False)['Net_PL'].sum()
        scalper_daily_pnl.columns = ['Date', 'Scalper_Daily_PL']
    else:
        scalper_daily_pnl = pd.DataFrame({'Date': daily_pnl['Date'], 'Scalper_Daily_PL': 0})
    
    merged_df = daily_pnl.merge(scalper_daily_pnl, on='Date', how='left')
    merged_df['Scalper_Daily_PL'] = merged_df['Scalper_Daily_PL'].fillna(0)
    merged_df['Scalper_Cumulative_PL'] = initial_balance + merged_df['Scalper_Daily_PL'].cumsum()
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=merged_df['Date'],
        y=merged_df['Cumulative_PL'],
        mode='lines+markers',
        name='整體累計盈虧',
        line=dict(color='#2E86AB', width=2.5),
        marker=dict(size=6),
        hovertemplate='<b>日期:</b> %{x|%Y-%m-%d}<br><b>整體累計:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=merged_df['Date'],
        y=merged_df['Scalper_Cumulative_PL'],
        mode='lines+markers',
        name=f'Scalper 累計盈虧 (<{scalper_minutes:.0f}分鐘)',
        line=dict(color='#F39C12', width=2.5, dash='dot'),
        marker=dict(size=6, symbol='diamond'),
        hovertemplate='<b>日期:</b> %{x|%Y-%m-%d}<br><b>Scalper 累計:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    fig.add_hline(
        y=initial_balance, 
        line_dash="dash", 
        line_color="gray", 
        line_width=1.5,
        annotation_text=f"初始資金: ${initial_balance:,}",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=dict(
            text=f'📈 累計淨盈虧走勢：整體 vs. Scalper (初始資金: ${initial_balance:,})',
            font=dict(size=16)
        ),
        xaxis_title='日期',
        yaxis_title='累計淨盈虧 ($)',
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    total_pnl = merged_df['Cumulative_PL'].iloc[-1] - initial_balance
    scalper_pnl = merged_df['Scalper_Cumulative_PL'].iloc[-1] - initial_balance
    scalper_ratio = (scalper_pnl / total_pnl * 100) if total_pnl != 0 else 0
    
    stats = {
        'total_pnl': total_pnl,
        'scalper_pnl': scalper_pnl,
        'scalper_ratio': scalper_ratio,
        'non_scalper_pnl': total_pnl - scalper_pnl
    }
    
    return fig, stats


def create_violin_plot(df, filter_extreme=True, max_clients=5000, sample_rate=0.1):
    """
    創建小提琴圖 (Violin Plot) 並內嵌 Box
    
    重構：使用採樣機制優化大數據渲染
    """
    aid_col = COLUMN_MAP['aid']
    
    # 使用向量化的客戶摘要函數（含採樣）
    aid_pl, sampling_info = get_client_summary_for_violin(df, max_clients, sample_rate)
    
    # 計算 1% 和 99% 百分位數
    Q1_percentile = aid_pl['累計淨盈虧'].quantile(0.01)
    Q99_percentile = aid_pl['累計淨盈虧'].quantile(0.99)
    
    # 根據選項決定是否過濾極端值
    if filter_extreme:
        plot_data = aid_pl[
            (aid_pl['累計淨盈虧'] >= Q1_percentile) & 
            (aid_pl['累計淨盈虧'] <= Q99_percentile)
        ].copy()
        title_suffix = "(已過濾極端值: 1%-99% 區間)"
        filtered_count = len(aid_pl) - len(plot_data)
    else:
        plot_data = aid_pl.copy()
        title_suffix = "(原始數據)"
        filtered_count = 0
    
    # 添加採樣資訊到標題
    if sampling_info['sampled']:
        title_suffix += f" [已採樣 {sampling_info['sample_rate']*100:.0f}%]"
    
    # 計算 IQR 和異常值邊界
    Q1 = aid_pl['累計淨盈虧'].quantile(0.25)
    Q3 = aid_pl['累計淨盈虧'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = aid_pl[(aid_pl['累計淨盈虧'] < lower_bound) | (aid_pl['累計淨盈虧'] > upper_bound)]
    
    mean_val = plot_data['累計淨盈虧'].mean()
    median_val = plot_data['累計淨盈虧'].median()
    std_val = plot_data['累計淨盈虧'].std()
    
    y_lower = plot_data['累計淨盈虧'].quantile(0.05)
    y_upper = plot_data['累計淨盈虧'].quantile(0.95)
    y_padding = (y_upper - y_lower) * 0.15
    y_range = [y_lower - y_padding, y_upper + y_padding]
    
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
        marker=dict(color='#3498DB', size=5, opacity=0.5, line=dict(width=0.5, color='#2C3E50')),
        box=dict(visible=True, fillcolor='rgba(255, 255, 255, 0.8)', line=dict(color='#2C3E50', width=2)),
        meanline=dict(visible=True, color='#E74C3C', width=2),
        hoverinfo='y',
        customdata=plot_data['AID'].values,
        hovertemplate='<b>累計淨盈虧:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[0] * len(plot_data),
        y=plot_data['累計淨盈虧'],
        mode='markers',
        marker=dict(color='rgba(0,0,0,0)', size=10),
        customdata=plot_data['AID'].values,
        hovertemplate='<b>AID:</b> %{customdata}<br><b>累計淨盈虧:</b> $%{y:,.2f}<extra></extra>',
        showlegend=False
    ))
    
    fig.update_layout(
        title=dict(text=f'🎻 客戶盈虧分布圖 (Violin Plot) {title_suffix}', font=dict(size=18)),
        height=700,
        yaxis=dict(title='累計淨盈虧 ($)', range=y_range, zeroline=True, zerolinecolor='rgba(0,0,0,0.3)', zerolinewidth=2, gridcolor='rgba(0,0,0,0.1)'),
        xaxis=dict(showticklabels=False, showgrid=False),
        showlegend=False,
        plot_bgcolor='rgba(248,249,250,1)',
        annotations=[
            dict(
                x=0.02, y=0.98, xref='paper', yref='paper',
                text=f'<b>📊 統計摘要</b><br>━━━━━━━━━━━<br>客戶數: {len(plot_data):,}<br>平均值: ${mean_val:,.2f}<br>中位數: ${median_val:,.2f}<br>標準差: ${std_val:,.2f}<br>━━━━━━━━━━━<br>Q25: ${plot_data["累計淨盈虧"].quantile(0.25):,.2f}<br>Q75: ${plot_data["累計淨盈虧"].quantile(0.75):,.2f}',
                showarrow=False, font=dict(size=11, family='monospace'), align='left',
                bgcolor='rgba(255,255,255,0.95)', bordercolor='#3498DB', borderwidth=2, borderpad=8
            ),
            dict(
                x=0.98, y=0.98, xref='paper', yref='paper',
                text='<b>📖 圖例說明</b><br>━━━━━━━━━━━<br>🔴 紅線 = 平均值<br>⬜ 白框 = IQR (Q25-Q75)<br>🔵 藍點 = 個別客戶<br>🎻 寬度 = 密度分布',
                showarrow=False, font=dict(size=10, family='monospace'), align='left',
                bgcolor='rgba(255,255,255,0.95)', bordercolor='#95a5a6', borderwidth=1, borderpad=8
            )
        ]
    )
    
    if y_range[0] <= 0 <= y_range[1]:
        fig.add_hline(y=0, line_dash="solid", line_color="rgba(0,0,0,0.5)", line_width=2,
                      annotation_text="損益平衡線", annotation_position="right", annotation_font=dict(size=10, color='gray'))
    
    pos_outliers = outliers[outliers['累計淨盈虧'] > upper_bound].nlargest(5, '累計淨盈虧')
    neg_outliers = outliers[outliers['累計淨盈虧'] < lower_bound].nsmallest(5, '累計淨盈虧')
    
    mean_pl = aid_pl['累計淨盈虧'].mean()
    if not pos_outliers.empty:
        pos_outliers = pos_outliers.copy()
        pos_outliers['偏離平均值'] = pos_outliers['累計淨盈虧'] - mean_pl
    if not neg_outliers.empty:
        neg_outliers = neg_outliers.copy()
        neg_outliers['偏離平均值'] = neg_outliers['累計淨盈虧'] - mean_pl
    
    filter_info = {
        'Q1_percentile': Q1_percentile,
        'Q99_percentile': Q99_percentile,
        'filtered_count': filtered_count,
        'total_count': sampling_info['original_count'],
        'y_range': y_range,
        'sampling_info': sampling_info
    }
    
    return fig, pos_outliers, neg_outliers, filter_info


def create_profit_factor_chart(df):
    """
    創建獲利因子分布圖（向量化重構）
    
    使用純向量化運算替代 apply
    """
    aid_col = COLUMN_MAP['aid']
    closed_pl_col = COLUMN_MAP['closed_pl']
    
    df = df.copy()
    df[aid_col] = df[aid_col].astype(str)
    
    # 向量化計算：分別聚合盈利和虧損
    df['_gain'] = np.where(df[closed_pl_col] > 0, df[closed_pl_col], 0)
    df['_loss'] = np.where(df[closed_pl_col] < 0, np.abs(df[closed_pl_col]), 0)
    
    pf_agg = df.groupby(aid_col, as_index=False).agg(
        Total_Gain=('_gain', 'sum'),
        Total_Loss=('_loss', 'sum')
    )
    
    # 向量化計算 Profit Factor
    pf_agg['Profit_Factor'] = np.where(
        pf_agg['Total_Loss'] == 0,
        np.where(pf_agg['Total_Gain'] > 0, 5.0, 0.0),
        pf_agg['Total_Gain'] / pf_agg['Total_Loss']
    )
    
    pf_data = pf_agg[[aid_col, 'Profit_Factor']].copy()
    pf_data.columns = ['AID', 'Profit_Factor']
    pf_data['AID'] = pf_data['AID'].astype(str)
    
    # 定義 PF 區間
    bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, float('inf')]
    labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0-5.0', '5.0+']
    pf_data['PF_Range'] = pd.cut(pf_data['Profit_Factor'], bins=bins, labels=labels, right=False)
    
    pf_dist = pf_data['PF_Range'].value_counts().sort_index().reset_index()
    pf_dist.columns = ['PF區間', '交易者數量']
    
    fig = go.Figure()
    colors = ['#E74C3C', '#E74C3C', '#27AE60', '#27AE60', '#27AE60', '#27AE60', '#27AE60', '#27AE60']
    
    fig.add_trace(go.Bar(
        x=pf_dist['PF區間'],
        y=pf_dist['交易者數量'],
        marker_color=colors[:len(pf_dist)],
        name='交易者數量'
    ))
    
    fig.add_vline(x=1.5, line_dash="dash", line_color="red", line_width=2,
                  annotation_text="PF=1.0 盈虧分界線", annotation_position="top")
    
    fig.update_layout(
        title='獲利因子分布 (Profit Factor Distribution)',
        xaxis_title='Profit Factor 區間',
        yaxis_title='交易者數量',
        height=500
    )
    
    profitable_ratio = (pf_data['Profit_Factor'] > 1.0).sum() / len(pf_data) * 100
    
    return fig, profitable_ratio, pf_data


def create_risk_return_scatter(df, initial_balance=0):
    """
    創建風險回報矩陣散佈圖（向量化重構）
    
    使用 get_client_metrics 替代逐筆迴圈
    """
    aid_col = COLUMN_MAP['aid']
    volume_col = COLUMN_MAP['volume']
    
    df = df.copy()
    df[aid_col] = df[aid_col].astype(str)
    
    # 使用向量化的客戶指標計算
    metrics = get_client_metrics(df, initial_balance)
    
    if metrics.empty:
        return go.Figure(), pd.DataFrame()
    
    # 計算交易量
    if volume_col in df.columns:
        volume_agg = df.groupby(aid_col, as_index=False)[volume_col].sum()
        volume_agg.columns = [aid_col, 'Trade_Volume']
    else:
        volume_agg = df.groupby(aid_col, as_index=False).size()
        volume_agg.columns = [aid_col, 'Trade_Volume']
    
    scatter_df = metrics.merge(volume_agg, on=aid_col, how='left')
    scatter_df = scatter_df.rename(columns={'Total_PL': 'Net_PL'})
    
    # 確保 AID 為字串
    scatter_df['AID'] = scatter_df[aid_col].astype(str)
    
    # 標準化點大小
    min_size, max_size = 10, 50
    vol_min, vol_max = scatter_df['Trade_Volume'].min(), scatter_df['Trade_Volume'].max()
    if vol_max > vol_min:
        scatter_df['Size'] = min_size + (scatter_df['Trade_Volume'] - vol_min) / (vol_max - vol_min) * (max_size - min_size)
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
            colorbar=dict(title='Net P/L')
        ),
        customdata=np.column_stack((
            scatter_df['AID'],
            scatter_df['Trade_Volume'],
            scatter_df['Trade_Count']
        )),
        hovertemplate=(
            '<b>AID:</b> %{customdata[0]}<br>'
            '<b>淨盈虧:</b> $%{y:,.2f}<br>'
            '<b>MDD:</b> %{x:.2f}%<br>'
            '<b>交易量:</b> %{customdata[1]:,.0f}<br>'
            '<b>交易筆數:</b> %{customdata[2]}<extra></extra>'
        ),
        name='交易者'
    ))
    
    fig.update_layout(
        title=f'風險回報矩陣 (Risk-Return Matrix) - 初始資金: ${initial_balance:,.0f}',
        xaxis_title='最大回撤 MDD (%)',
        yaxis_title='月度淨盈虧 (Net P/L)',
        height=600
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    
    fig.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper", text="🌟 明星交易員", showarrow=False, font=dict(size=12, color="green"))
    fig.add_annotation(x=0.95, y=0.95, xref="paper", yref="paper", text="⚡ 激進型", showarrow=False, font=dict(size=12, color="orange"))
    fig.add_annotation(x=0.05, y=0.05, xref="paper", yref="paper", text="🐢 守舊型", showarrow=False, font=dict(size=12, color="gray"))
    fig.add_annotation(x=0.95, y=0.05, xref="paper", yref="paper", text="⚠️ 高風險", showarrow=False, font=dict(size=12, color="red"))
    
    return fig, scatter_df


def create_hold_time_analysis(df, scalper_threshold_seconds=300):
    """創建持倉時間 vs 勝率關聯分析"""
    aid_col = COLUMN_MAP['aid']
    scalp_minutes = scalper_threshold_seconds / 60
    
    df_analysis = df.copy()
    
    # 向量化分類持倉時間
    conditions = [
        df_analysis['Hold_Seconds'] < scalper_threshold_seconds,
        df_analysis['Hold_Seconds'] < 3600,
        df_analysis['Hold_Seconds'] < 86400,
        df_analysis['Hold_Seconds'] >= 86400
    ]
    choices = [f'Scalp (<{scalp_minutes:.0f}m)', 'Short (<1h)', 'Intraday (<24h)', 'Swing (>1d)']
    
    df_analysis['Hold_Category'] = np.select(conditions, choices, default=None)
    df_analysis = df_analysis[df_analysis['Hold_Category'].notna()].copy()
    
    if df_analysis.empty:
        return None, None
    
    # 向量化計算勝率
    df_analysis['_is_win'] = (df_analysis['Net_PL'] > 0).astype(int)
    
    category_stats = df_analysis.groupby('Hold_Category', as_index=False).agg(
        交易筆數=('Net_PL', 'count'),
        總盈虧=('Net_PL', 'sum'),
        平均盈虧=('Net_PL', 'mean'),
        _wins=('_is_win', 'sum')
    )
    category_stats['勝率(%)'] = (category_stats['_wins'] / category_stats['交易筆數'] * 100).round(2)
    category_stats = category_stats.drop(columns=['_wins'])
    category_stats.columns = ['持倉類型', '交易筆數', '總盈虧', '平均盈虧', '勝率(%)']
    
    order = [f'Scalp (<{scalp_minutes:.0f}m)', 'Short (<1h)', 'Intraday (<24h)', 'Swing (>1d)']
    category_stats['持倉類型'] = pd.Categorical(category_stats['持倉類型'], categories=order, ordered=True)
    category_stats = category_stats.sort_values('持倉類型')
    
    # 創建散佈圖
    df_analysis['Hold_Seconds_Log'] = np.log10(df_analysis['Hold_Seconds'].clip(lower=1))
    df_analysis['Color'] = np.where(df_analysis['Net_PL'] > 0, 'Profit', 'Loss')
    
    fig = px.scatter(
        df_analysis,
        x='Hold_Seconds_Log',
        y='Net_PL',
        color='Color',
        color_discrete_map={'Profit': '#27AE60', 'Loss': '#E74C3C'},
        opacity=0.6,
        title=f'持倉時間 vs 單筆盈虧 (Scalp 定義: <{scalp_minutes:.0f}分鐘)'
    )
    
    fig.add_vline(x=np.log10(scalper_threshold_seconds), line_dash="dash", line_color="red", line_width=2,
                  annotation_text=f"Scalp 閾值 ({scalp_minutes:.0f}分鐘)")
    fig.add_vline(x=np.log10(3600), line_dash="dash", line_color="gray", annotation_text="1小時")
    fig.add_vline(x=np.log10(86400), line_dash="dash", line_color="gray", annotation_text="24小時")
    
    fig.update_layout(xaxis_title='持倉時間 (Log10 秒)', yaxis_title='單筆盈虧 (Net P/L)', height=500)
    
    return fig, category_stats


def create_daily_pnl_chart(df):
    """創建每日盈虧柱狀圖"""
    exec_col = COLUMN_MAP['execution_time']
    
    df_daily = df.copy()
    df_daily['Date'] = df_daily[exec_col].dt.date
    
    daily_pnl = df_daily.groupby('Date', as_index=False)['Net_PL'].sum()
    daily_pnl.columns = ['日期', '每日盈虧']
    
    colors = ['#27AE60' if x > 0 else '#E74C3C' for x in daily_pnl['每日盈虧']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=daily_pnl['日期'],
        y=daily_pnl['每日盈虧'],
        marker_color=colors,
        name='每日盈虧'
    ))
    
    fig.add_hline(y=0, line_color="black", line_width=1)
    
    fig.update_layout(
        title='30天每日盈虧分布',
        xaxis_title='日期',
        yaxis_title='淨盈虧',
        height=400
    )
    
    return fig


# ==================== 導出功能 ====================

def export_to_excel(df, initial_balance=10000, scalper_threshold_seconds=300):
    """導出完整分析數據到 Excel（多分頁）"""
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    
    output = BytesIO()
    
    closing_df = filter_closing_trades(df)
    
    aid_col = COLUMN_MAP['aid']
    volume_col = COLUMN_MAP['volume']
    exec_col = COLUMN_MAP['execution_time']
    instrument_col = COLUMN_MAP['instrument']
    
    # ========== Sheet 1: 數據摘要 ==========
    total_trades = len(df)
    closing_trades = len(closing_df)
    unique_clients = df[aid_col].nunique()
    total_net_pl = closing_df['Net_PL'].sum()
    avg_net_pl = closing_df['Net_PL'].mean() if len(closing_df) > 0 else 0
    
    profitable_trades = (closing_df['Net_PL'] > 0).sum()
    losing_trades = (closing_df['Net_PL'] < 0).sum()
    win_rate = (profitable_trades / len(closing_df) * 100) if len(closing_df) > 0 else 0
    
    client_pl = closing_df.groupby(aid_col)['Net_PL'].sum()
    profitable_clients = (client_pl > 0).sum()
    losing_clients = (client_pl < 0).sum()
    
    scalper_trades = closing_df[closing_df['Hold_Seconds'] < scalper_threshold_seconds]
    scalper_count = scalper_trades[aid_col].nunique() if not scalper_trades.empty else 0
    scalper_pl = scalper_trades['Net_PL'].sum() if not scalper_trades.empty else 0
    
    summary_data = [
        ['指標', '數值', '說明'],
        ['總交易筆數', total_trades, '所有交易記錄'],
        ['平倉交易筆數', closing_trades, 'CLOSING 類型交易'],
        ['總客戶數', unique_clients, '不重複的 AID 數量'],
        ['總淨盈虧', round(total_net_pl, 2), 'Net_PL = Closed P/L + Commission + Swap'],
        ['平均單筆盈虧', round(avg_net_pl, 2), '平倉交易的平均 Net_PL'],
        ['', '', ''],
        ['盈利交易筆數', profitable_trades, 'Net_PL > 0'],
        ['虧損交易筆數', losing_trades, 'Net_PL < 0'],
        ['整體勝率(%)', round(win_rate, 2), '盈利交易佔比'],
        ['', '', ''],
        ['盈利客戶數', profitable_clients, '累計 Net_PL > 0'],
        ['虧損客戶數', losing_clients, '累計 Net_PL < 0'],
        ['客戶盈利比(%)', round(profitable_clients / unique_clients * 100, 2) if unique_clients > 0 else 0, '盈利客戶佔比'],
        ['', '', ''],
        [f'Scalper 數量 (<{scalper_threshold_seconds/60:.0f}分鐘)', scalper_count, '短線交易者'],
        ['Scalper 總盈虧', round(scalper_pl, 2), 'Scalper 交易的累計 Net_PL'],
        ['', '', ''],
        ['初始資金設定', initial_balance, '用於 MDD 計算'],
        ['Scalper 閾值(秒)', scalper_threshold_seconds, '持倉時間閾值'],
        ['報告生成時間', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '']
    ]
    
    summary_df = pd.DataFrame(summary_data[1:], columns=summary_data[0])
    
    # ========== Sheet 2: 風險回報清單（使用向量化）==========
    metrics = get_client_metrics(closing_df, initial_balance, scalper_threshold_seconds)
    
    if not metrics.empty:
        # 添加額外欄位
        if volume_col in closing_df.columns:
            volume_agg = closing_df.groupby(aid_col, as_index=False)[volume_col].sum()
            volume_agg.columns = [aid_col, 'Trade_Volume']
            metrics = metrics.merge(volume_agg, on=aid_col, how='left')
        
        # 計算平均持倉時間
        hold_agg = closing_df.groupby(aid_col, as_index=False)['Hold_Seconds'].mean()
        hold_agg.columns = [aid_col, 'Avg_Hold_Seconds']
        hold_agg['Avg_Hold_Minutes'] = (hold_agg['Avg_Hold_Seconds'] / 60).round(2)
        metrics = metrics.merge(hold_agg[[aid_col, 'Avg_Hold_Minutes']], on=aid_col, how='left')
        
        # 是否為 Scalper
        metrics['Is_Scalper'] = np.where(metrics['Scalp_Pct'] > 50, 'Yes', 'No')
        
        risk_return_df = metrics.rename(columns={
            aid_col: 'AID',
            'Total_PL': 'Net_PL',
            'MDD_Pct': 'MDD(%)',
            'Win_Rate': 'Win_Rate(%)',
            'Scalp_Pct': 'Scalper_Ratio(%)'
        })
        risk_return_df = risk_return_df.sort_values('Net_PL', ascending=False)
    else:
        risk_return_df = pd.DataFrame()
    
    # ========== Sheet 3: Scalper 清單 ==========
    scalper_df = closing_df[closing_df['Hold_Seconds'] < scalper_threshold_seconds].copy()
    
    if not scalper_df.empty:
        scalper_export_cols = [
            aid_col, exec_col, COLUMN_MAP['open_time'],
            instrument_col, COLUMN_MAP['side'], volume_col,
            COLUMN_MAP['closed_pl'], COLUMN_MAP['commission'], COLUMN_MAP['swap'],
            'Net_PL', 'Hold_Seconds'
        ]
        
        existing_cols = [col for col in scalper_export_cols if col in scalper_df.columns]
        scalper_export_df = scalper_df[existing_cols].copy()
        scalper_export_df['Hold_Minutes'] = (scalper_export_df['Hold_Seconds'] / 60).round(2)
        
        if exec_col in scalper_export_df.columns:
            scalper_export_df = scalper_export_df.sort_values(exec_col, ascending=False)
    else:
        scalper_export_df = pd.DataFrame({'訊息': ['無符合條件的 Scalper 交易記錄']})
    
    # ========== 寫入 Excel ==========
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='數據摘要', index=False)
        if not risk_return_df.empty:
            risk_return_df.to_excel(writer, sheet_name='風險回報清單', index=False)
        scalper_export_df.to_excel(writer, sheet_name='Scalper清單', index=False)
        
        # 格式化
        workbook = writer.book
        header_font = Font(bold=True, color='FFFFFF')
        header_fill = PatternFill(start_color='2E86AB', end_color='2E86AB', fill_type='solid')
        thin_border = Border(
            left=Side(style='thin'), right=Side(style='thin'),
            top=Side(style='thin'), bottom=Side(style='thin')
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
    st.markdown("**支持大規模交易數據（十萬筆以上）的處理與分析** | *第一階段重構：核心數據引擎*")
    
    # 側邊欄
    with st.sidebar:
        st.header("📁 數據上傳")
        uploaded_files = st.file_uploader(
            "上傳交易數據檔案 (.xlsx 或 .csv)",
            type=['xlsx', 'csv'],
            accept_multiple_files=True
        )
        
        st.header("⚙️ 參數設定")
        
        initial_balance = st.number_input(
            "初始資金 (用於 MDD 計算)",
            value=10000, min_value=0, step=1000,
            help="設定每位交易者的初始資金"
        )
        
        scalper_minutes = st.number_input(
            "Scalper 持倉時間定義 (分鐘)",
            value=5, min_value=1, max_value=60, step=1,
            help="持倉時間小於此值的交易將被歸類為 Scalp 交易"
        )
        
        scalper_threshold_seconds = scalper_minutes * 60
        
        if uploaded_files:
            st.success(f"已上傳 {len(uploaded_files)} 個檔案")
            st.markdown("---")
            st.info(f"💰 初始資金: **${initial_balance:,}**")
            st.info(f"⏱️ Scalper 定義: **<{scalper_minutes} 分鐘**")
            
            st.markdown("---")
            st.header("📥 導出報表")
            
            @st.cache_data(show_spinner=False)
            def generate_excel_report(_df, init_bal, scalp_thresh):
                return export_to_excel(_df, init_bal, scalp_thresh)
            
            df_for_export = load_data(uploaded_files)
            
            if df_for_export is not None:
                with st.spinner("正在生成報表..."):
                    excel_data = generate_excel_report(df_for_export, initial_balance, scalper_threshold_seconds)
                
                st.download_button(
                    label="📊 下載完整分析數據 (.xlsx)",
                    data=excel_data,
                    file_name=f"trading_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="包含三個分頁：數據摘要、風險回報清單、Scalper 清單"
                )
    
    if not uploaded_files:
        st.info("👈 請在左側上傳交易數據檔案開始分析")
        
        st.markdown("""
        ### 📋 功能說明
        
        **第一部分：當日分析 (Daily Analysis)**
        - Top 10 Profit 客戶
        - Top 10 Scalpers（短線交易者）
        
        **第二部分：30天分析 (30-Day Analysis)**
        1. 累計盈虧走勢圖
        2. 客戶盈虧分布圖 (Violin Plot)
        3. 獲利因子分布圖 (PF Distribution)
        4. 風險回報矩陣 (Risk-Return Scatter)
        5. 持倉 vs 勝率關聯分析
        6. 每日盈虧柱狀圖
        
        **🚀 第一階段重構特性：**
        - ✅ 高效快取機制 (`@st.cache_data`)
        - ✅ 向量化運算（禁止 apply/loop）
        - ✅ AID 強制字串化（解決複製失效）
        - ✅ 大數據採樣優化（Violin Plot）
        """)
        return
    
    # 載入數據
    with st.spinner("正在載入和處理數據..."):
        df = load_data(uploaded_files)
    
    if df is None or df.empty:
        st.error("無法載入數據，請檢查檔案格式")
        return
    
    # 顯示數據摘要
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    closing_df = filter_closing_trades(df)
    
    with col1:
        st.metric("總交易筆數", f"{len(df):,}")
    with col2:
        st.metric("平倉交易筆數", f"{len(closing_df):,}")
    with col3:
        st.metric("交易者數量", f"{df[COLUMN_MAP['aid']].nunique():,}")
    with col4:
        total_pnl = closing_df['Net_PL'].sum()
        st.metric("總淨盈虧", f"${total_pnl:,.2f}")
    
    # ==================== 第一部分：當日分析 ====================
    st.markdown("---")
    st.header("📅 第一部分：當日分析 (Daily Analysis)")
    
    daily_result = get_daily_analysis(df, scalper_threshold_seconds)
    
    if daily_result and daily_result[0] is not None:
        top_profit, top_scalpers, latest_date = daily_result
        
        st.subheader(f"📆 分析日期: {latest_date}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏆 Top 10 Profit 客戶")
            if not top_profit.empty:
                st.dataframe(top_profit, use_container_width=True, hide_index=True)
            else:
                st.info("當日無盈利數據")
        
        with col2:
            st.markdown(f"### ⚡ Top 10 Scalpers (定義: 持倉 <{scalper_minutes} 分鐘)")
            if not top_scalpers.empty:
                st.dataframe(top_scalpers, use_container_width=True, hide_index=True)
            else:
                st.info(f"當日無持倉 <{scalper_minutes} 分鐘的短線交易數據")
    else:
        st.warning("無法取得當日分析數據")
    
    # ==================== 第二部分：30天分析 ====================
    st.markdown("---")
    st.header("📊 第二部分：30天分析 (30-Day Analysis)")
    
    result_30d = get_30day_analysis(df)
    
    if result_30d:
        df_30d, start_date, end_date = result_30d
        
        st.subheader(f"📆 分析期間: {start_date.date()} ~ {end_date.date()}")
        
        # 30天 Top 10 列表
        col1, col2 = st.columns(2)
        
        aid_col = COLUMN_MAP['aid']
        instrument_col = COLUMN_MAP['instrument']
        
        with col1:
            st.markdown("### 🏆 30天 Top 10 Profit 客戶")
            top_30d_profit = df_30d.groupby(aid_col)['Net_PL'].sum().nlargest(10).reset_index()
            top_30d_profit.columns = ['AID', '30天總盈虧']
            top_30d_profit['AID'] = top_30d_profit['AID'].astype(str)
            st.dataframe(top_30d_profit, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown(f"### ⚡ 30天 Top 10 Scalpers (定義: 持倉 <{scalper_minutes} 分鐘)")
            scalp_30d = df_30d[df_30d['Hold_Seconds'] < scalper_threshold_seconds]
            if not scalp_30d.empty:
                scalper_30d = scalp_30d.groupby(aid_col, as_index=False).agg(
                    交易筆數=('Net_PL', 'count'),
                    總盈虧=('Net_PL', 'sum'),
                    平均持倉秒數=('Hold_Seconds', 'mean')
                )
                scalper_30d.columns = ['AID', '交易筆數', '總盈虧', '平均持倉秒數']
                scalper_30d = scalper_30d.nlargest(10, '交易筆數')
                scalper_30d['AID'] = scalper_30d['AID'].astype(str)
                scalper_30d['平均持倉秒數'] = scalper_30d['平均持倉秒數'].round(1)
                st.dataframe(scalper_30d, use_container_width=True, hide_index=True)
            else:
                st.info(f"無持倉 <{scalper_minutes} 分鐘的短線交易數據")
        
        st.markdown("---")
        
        # 1. 累計淨盈虧走勢圖
        st.markdown("### 📈 1. 30天累計淨盈虧走勢")
        
        cumulative_fig, pnl_stats = create_cumulative_pnl_chart(df_30d, initial_balance, scalper_threshold_seconds)
        st.plotly_chart(cumulative_fig, use_container_width=True)
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("整體淨盈虧", f"${pnl_stats['total_pnl']:,.2f}")
        with col_stat2:
            st.metric(f"Scalper 淨盈虧 (<{scalper_minutes}分鐘)", f"${pnl_stats['scalper_pnl']:,.2f}",
                     delta=f"{pnl_stats['scalper_ratio']:.1f}% 佔比")
        with col_stat3:
            st.metric("非 Scalper 淨盈虧", f"${pnl_stats['non_scalper_pnl']:,.2f}")
        
        st.markdown("---")
        
        # 2. 小提琴圖 (Violin Plot)
        st.markdown("### 🎻 2. 客戶盈虧分布圖 (Violin Plot)")
        
        col_filter1, col_filter2 = st.columns([1, 2])
        with col_filter1:
            filter_extreme = st.checkbox(
                "隱藏極端離群值 (1%-99%)", 
                value=True,
                help="勾選後將過濾掉最極端的 1% 高值和 1% 低值"
            )
        
        violin_fig, pos_outliers, neg_outliers, filter_info = create_violin_plot(df_30d, filter_extreme)
        st.plotly_chart(violin_fig, use_container_width=True)
        
        # 顯示過濾和採樣資訊
        info_messages = []
        if filter_extreme and filter_info['filtered_count'] > 0:
            info_messages.append(f"已過濾 {filter_info['filtered_count']} 位極端客戶")
        if filter_info.get('sampling_info', {}).get('sampled'):
            info_messages.append(f"已從 {filter_info['sampling_info']['original_count']:,} 位客戶採樣 {filter_info['sampling_info']['sampled_count']:,} 位")
        
        if info_messages:
            st.info(f"📊 {' | '.join(info_messages)}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔴 Top 5 Positive Outliers (暴利)")
            if not pos_outliers.empty:
                pos_outliers['AID'] = pos_outliers['AID'].astype(str)
                st.dataframe(pos_outliers, use_container_width=True, hide_index=True)
            else:
                st.info("無正向異常值")
        
        with col2:
            st.markdown("#### 🔵 Top 5 Negative Outliers (暴損)")
            if not neg_outliers.empty:
                neg_outliers['AID'] = neg_outliers['AID'].astype(str)
                st.dataframe(neg_outliers, use_container_width=True, hide_index=True)
            else:
                st.info("無負向異常值")
        
        st.markdown("---")
        
        # 3. 獲利因子分布圖
        st.markdown("### 📊 3. 獲利因子分布 (Profit Factor)")
        
        pf_fig, profitable_ratio, pf_data = create_profit_factor_chart(df_30d)
        st.plotly_chart(pf_fig, use_container_width=True)
        st.success(f"📈 **30天內 PF > 1.0 的交易者佔比: {profitable_ratio:.1f}%** (賺錢的人)")
        
        st.markdown("---")
        
        # 4. 風險回報矩陣
        st.markdown("### 🎯 4. 風險回報矩陣 (Risk-Return Matrix)")
        
        scatter_fig, scatter_data = create_risk_return_scatter(df_30d, initial_balance)
        st.plotly_chart(scatter_fig, use_container_width=True)
        
        st.markdown("""
        **象限說明：**
        - 🌟 **左上 (Low MDD, High P/L)**: 明星交易員
        - ⚡ **右上 (High MDD, High P/L)**: 激進型交易員
        - 🐢 **左下 (Low MDD, Low P/L)**: 守舊型交易員
        - ⚠️ **右下 (High MDD, Low P/L)**: 高風險交易員
        """)
        
        with st.expander("📋 查看風險回報詳細數據"):
            if not scatter_data.empty:
                display_data = scatter_data[['AID', 'Net_PL', 'MDD_Pct', 'Trade_Volume', 'Trade_Count']].copy()
                display_data['AID'] = display_data['AID'].astype(str)
                display_data.columns = ['AID', '淨盈虧', 'MDD (%)', '交易量', '交易筆數']
                display_data['淨盈虧'] = display_data['淨盈虧'].apply(lambda x: f"${x:,.2f}")
                display_data['MDD (%)'] = display_data['MDD (%)'].apply(lambda x: f"{x:.2f}%")
                st.dataframe(display_data, use_container_width=True, hide_index=True)
                st.caption(f"💡 MDD 計算基於初始資金 ${initial_balance:,}")
        
        st.markdown("---")
        
        # 5. 持倉時間 vs 勝率分析
        st.markdown(f"### ⏱️ 5. 持倉時間 vs 勝率關聯分析 (Scalp 定義: <{scalper_minutes} 分鐘)")
        
        hold_fig, hold_stats = create_hold_time_analysis(df_30d, scalper_threshold_seconds)
        
        if hold_fig is not None:
            st.plotly_chart(hold_fig, use_container_width=True)
            st.markdown("#### 各持倉類型統計")
            st.dataframe(hold_stats, use_container_width=True, hide_index=True)
        else:
            st.warning("無持倉時間數據可供分析")
        
        st.markdown("---")
        
        # 6. 每日盈虧柱狀圖
        st.markdown("### 📅 6. 每日盈虧柱狀圖")
        
        daily_fig = create_daily_pnl_chart(df_30d)
        st.plotly_chart(daily_fig, use_container_width=True)
    
    else:
        st.warning("無法取得 30 天分析數據")
    
    # ==================== 導出功能（底部備用入口）====================
    st.markdown("---")
    st.header("📥 數據導出")
    
    st.info("💡 您也可以在左側邊欄直接點擊「下載完整分析數據」按鈕導出報表。")
    
    col_export1, col_export2 = st.columns([2, 1])
    
    with col_export1:
        st.markdown("""
        **Excel 報表內容說明：**
        - **Sheet 1 (數據摘要)**: 總客戶數、總盈虧、平均盈虧、勝率等基本指標
        - **Sheet 2 (風險回報清單)**: 所有 AID 的 Net_PL, MDD%, Trade_Volume, Win_Rate 等
        - **Sheet 3 (Scalper 清單)**: 符合 Scalper 定義的交易明細
        """)
    
    with col_export2:
        with st.spinner("準備報表..."):
            excel_data = export_to_excel(df, initial_balance, scalper_threshold_seconds)
        
        st.download_button(
            label="📊 下載完整分析數據 (.xlsx)",
            data=excel_data,
            file_name=f"trading_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )


if __name__ == "__main__":
    main()
