"""
交易數據分析系統 (Trading Analysis System) v2.0
完整重構版本：
- Tab 1: 整體數據概覽（含交易風格圓餅圖）
- Tab 2: 個別客戶報告卡（深度行為分析）
- Tab 3: 當日數據概覽（英雄榜 + 進階指標）
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO

# ==================== 頁面配置 ====================
st.set_page_config(page_title="交易數據分析系統", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

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

# 交易風格配色
STYLE_COLORS = {
    '極短線 (Scalp)': '#E74C3C',      # 鮮紅色
    '短線 (Intraday)': '#F39C12',     # 橘色
    '中線 (Day Trade)': '#3498DB',    # 藍色
    '長線 (Swing)': '#27AE60'         # 深綠色
}


# ==================== 數據載入與預處理 ====================
@st.cache_data(show_spinner=False)
def load_and_preprocess(uploaded_files):
    """載入並預處理交易數據"""
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
    
    df = pd.concat(dfs, ignore_index=True)
    exec_col = COLUMN_MAP['execution_time']
    if exec_col in df.columns:
        df = df[df[exec_col] != 'Total'].copy()
    df = df.drop_duplicates()
    
    for col in ['execution_time', 'open_time']:
        if COLUMN_MAP[col] in df.columns:
            df[COLUMN_MAP[col]] = pd.to_datetime(df[COLUMN_MAP[col]], errors='coerce')
    
    for col in ['closed_pl', 'commission', 'swap']:
        if COLUMN_MAP[col] in df.columns:
            df[COLUMN_MAP[col]] = df[COLUMN_MAP[col]].fillna(0)
    
    df['Net_PL'] = df[COLUMN_MAP['closed_pl']] + df[COLUMN_MAP['commission']] + df[COLUMN_MAP['swap']]
    
    exec_time = df[COLUMN_MAP['execution_time']]
    open_time = df[COLUMN_MAP['open_time']]
    df['Hold_Seconds'] = np.where(pd.notna(exec_time) & pd.notna(open_time), (exec_time - open_time).dt.total_seconds(), np.nan)
    df['Hold_Minutes'] = df['Hold_Seconds'] / 60
    
    if COLUMN_MAP['aid'] in df.columns:
        df[COLUMN_MAP['aid']] = df[COLUMN_MAP['aid']].astype(str).str.replace(r'\.0$', '', regex=True).str.replace(',', '', regex=False).str.strip()
    
    return df


def filter_closing_trades(df):
    """篩選已平倉交易"""
    action_col = COLUMN_MAP['action']
    if action_col in df.columns:
        return df[df[action_col] == 'CLOSING'].copy()
    return df


def classify_trading_style(hold_minutes):
    """根據持倉時間分類交易風格"""
    if pd.isna(hold_minutes):
        return '短線 (Intraday)'
    elif hold_minutes < 5:
        return '極短線 (Scalp)'
    elif hold_minutes < 60:
        return '短線 (Intraday)'
    elif hold_minutes < 1440:
        return '中線 (Day Trade)'
    else:
        return '長線 (Swing)'


# ==================== 即時計算 AID 統計 ====================
def calculate_all_aid_stats_realtime(df, initial_balance, scalper_threshold_seconds):
    """即時計算所有 AID 的統計數據"""
    aid_col = COLUMN_MAP['aid']
    volume_col = COLUMN_MAP['volume']
    exec_col = COLUMN_MAP['execution_time']
    instrument_col = COLUMN_MAP['instrument']
    closed_pl_col = COLUMN_MAP['closed_pl']
    
    closing_df = filter_closing_trades(df)
    results = []
    
    for aid in closing_df[aid_col].unique():
        aid_data = closing_df[closing_df[aid_col] == aid].copy()
        
        net_pl = aid_data['Net_PL'].sum()
        trade_count = len(aid_data)
        trade_volume = aid_data[volume_col].sum() if volume_col in aid_data.columns else trade_count
        
        wins = (aid_data['Net_PL'] > 0).sum()
        win_rate = (wins / trade_count * 100) if trade_count > 0 else 0
        
        avg_hold_seconds = aid_data['Hold_Seconds'].mean() if 'Hold_Seconds' in aid_data.columns else 0
        avg_hold_seconds = avg_hold_seconds if pd.notna(avg_hold_seconds) else 0
        
        scalper_trades = aid_data[aid_data['Hold_Seconds'] < scalper_threshold_seconds]
        scalper_count = len(scalper_trades)
        scalper_ratio = (scalper_count / trade_count * 100) if trade_count > 0 else 0
        scalper_pl = scalper_trades['Net_PL'].sum() if not scalper_trades.empty else 0
        
        aid_sorted = aid_data.sort_values(exec_col)
        if len(aid_sorted) >= 2:
            cumulative_pl = aid_sorted['Net_PL'].cumsum()
            equity = initial_balance + cumulative_pl
            running_max = equity.cummax()
            drawdown = np.where(running_max != 0, (equity - running_max) / running_max, 0)
            mdd_pct = abs(np.min(drawdown) * 100)
        else:
            mdd_pct = 0.0
        
        gains = aid_data[aid_data[closed_pl_col] > 0][closed_pl_col].sum()
        losses = abs(aid_data[aid_data[closed_pl_col] < 0][closed_pl_col].sum())
        if losses == 0 and gains > 0:
            profit_factor = 5.0
        elif gains == 0:
            profit_factor = 0.0
        else:
            profit_factor = gains / losses
        
        if instrument_col in aid_data.columns and not aid_data[instrument_col].empty:
            main_symbol = aid_data[instrument_col].mode().iloc[0] if len(aid_data[instrument_col].mode()) > 0 else 'N/A'
        else:
            main_symbol = 'N/A'
        
        results.append({
            'AID': aid, 'Net_PL': round(net_pl, 2), 'Trade_Count': trade_count,
            'Trade_Volume': round(trade_volume, 2), 'Win_Rate': round(win_rate, 2),
            'Avg_Hold_Seconds': round(avg_hold_seconds, 2), 'MDD_Pct': round(mdd_pct, 2),
            'Profit_Factor': round(profit_factor, 2), 'Scalper_Count': scalper_count,
            'Scalper_Ratio': round(scalper_ratio, 2), 'Scalper_PL': round(scalper_pl, 2),
            'Main_Symbol': main_symbol
        })
    
    return pd.DataFrame(results)


# ==================== 深度行為分析計算 ====================
def calculate_deep_behavioral_stats(client_df, scalper_threshold_seconds):
    """計算單一客戶的深度行為分析數據"""
    side_col = COLUMN_MAP['side']
    
    # 基本統計
    total_trades = len(client_df)
    total_pl = client_df['Net_PL'].sum()
    total_minutes = client_df['Hold_Minutes'].sum() if 'Hold_Minutes' in client_df.columns else 0
    total_minutes = total_minutes if pd.notna(total_minutes) else 0
    
    # 連續盈虧分析
    pnl_signs = (client_df['Net_PL'] > 0).astype(int)
    streaks = []
    current_streak = 1
    current_type = pnl_signs.iloc[0] if len(pnl_signs) > 0 else 0
    
    for i in range(1, len(pnl_signs)):
        if pnl_signs.iloc[i] == current_type:
            current_streak += 1
        else:
            streaks.append((current_type, current_streak))
            current_streak = 1
            current_type = pnl_signs.iloc[i]
    if len(pnl_signs) > 0:
        streaks.append((current_type, current_streak))
    
    win_streaks = [s[1] for s in streaks if s[0] == 1]
    loss_streaks = [s[1] for s in streaks if s[0] == 0]
    
    max_win_streak = max(win_streaks) if win_streaks else 0
    max_loss_streak = max(loss_streaks) if loss_streaks else 0
    
    # 連續盈虧金額
    client_sorted = client_df.sort_values(COLUMN_MAP['execution_time']).copy()
    client_sorted['streak_group'] = (client_sorted['Net_PL'] > 0).ne((client_sorted['Net_PL'] > 0).shift()).cumsum()
    streak_sums = client_sorted.groupby('streak_group')['Net_PL'].sum()
    max_streak_profit = streak_sums.max() if not streak_sums.empty else 0
    max_streak_loss = streak_sums.min() if not streak_sums.empty else 0
    
    # 多空拆解
    buy_trades = client_df[client_df[side_col] == 'BUY'] if side_col in client_df.columns else pd.DataFrame()
    sell_trades = client_df[client_df[side_col] == 'SELL'] if side_col in client_df.columns else pd.DataFrame()
    
    buy_count = len(buy_trades)
    sell_count = len(sell_trades)
    buy_ratio = (buy_count / total_trades * 100) if total_trades > 0 else 0
    sell_ratio = (sell_count / total_trades * 100) if total_trades > 0 else 0
    
    buy_pl = buy_trades['Net_PL'].sum() if not buy_trades.empty else 0
    sell_pl = sell_trades['Net_PL'].sum() if not sell_trades.empty else 0
    
    buy_wins = (buy_trades['Net_PL'] > 0).sum() if not buy_trades.empty else 0
    sell_wins = (sell_trades['Net_PL'] > 0).sum() if not sell_trades.empty else 0
    buy_winrate = (buy_wins / buy_count * 100) if buy_count > 0 else 0
    sell_winrate = (sell_wins / sell_count * 100) if sell_count > 0 else 0
    
    # 剝頭皮分析
    scalp_trades = client_df[client_df['Hold_Seconds'] < scalper_threshold_seconds]
    scalp_count = len(scalp_trades)
    scalp_ratio = (scalp_count / total_trades * 100) if total_trades > 0 else 0
    scalp_pl = scalp_trades['Net_PL'].sum() if not scalp_trades.empty else 0
    scalp_contribution = (scalp_pl / total_pl * 100) if total_pl != 0 else 0
    scalp_wins = (scalp_trades['Net_PL'] > 0).sum() if not scalp_trades.empty else 0
    scalp_winrate = (scalp_wins / scalp_count * 100) if scalp_count > 0 else 0
    
    # 時間效率
    avg_minutes = total_minutes / total_trades if total_trades > 0 else 0
    profit_per_minute = total_pl / total_minutes if total_minutes > 0 else 0
    
    # 轉換為時:分:秒
    avg_seconds = avg_minutes * 60
    hours = int(avg_seconds // 3600)
    minutes = int((avg_seconds % 3600) // 60)
    seconds = int(avg_seconds % 60)
    avg_hold_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    avg_hold_days = avg_minutes / 1440
    
    return {
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        'max_streak_profit': max_streak_profit,
        'max_streak_loss': max_streak_loss,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'buy_ratio': buy_ratio,
        'sell_ratio': sell_ratio,
        'buy_pl': buy_pl,
        'sell_pl': sell_pl,
        'buy_winrate': buy_winrate,
        'sell_winrate': sell_winrate,
        'scalp_count': scalp_count,
        'scalp_ratio': scalp_ratio,
        'scalp_pl': scalp_pl,
        'scalp_contribution': scalp_contribution,
        'scalp_winrate': scalp_winrate,
        'avg_hold_formatted': avg_hold_formatted,
        'avg_hold_days': avg_hold_days,
        'profit_per_minute': profit_per_minute,
        'total_minutes': total_minutes
    }


# ==================== 當日進階指標計算 ====================
def calculate_daily_advanced_metrics(day_df, initial_balance, scalper_threshold_seconds):
    """計算當日英雄榜進階指標"""
    aid_col = COLUMN_MAP['aid']
    exec_col = COLUMN_MAP['execution_time']
    
    closing_df = filter_closing_trades(day_df)
    results = []
    
    for aid in closing_df[aid_col].unique():
        aid_data = closing_df[closing_df[aid_col] == aid].copy()
        
        net_pl = aid_data['Net_PL'].sum()
        if net_pl <= 0:
            continue
        
        trade_count = len(aid_data)
        wins = (aid_data['Net_PL'] > 0).sum()
        losses = trade_count - wins
        win_rate = (wins / trade_count * 100) if trade_count > 0 else 0
        
        # Scalp 數據
        scalp_trades = aid_data[aid_data['Hold_Seconds'] < scalper_threshold_seconds]
        scalp_count = len(scalp_trades)
        scalp_pl = scalp_trades['Net_PL'].sum() if not scalp_trades.empty else 0
        scalp_pct = (scalp_count / trade_count * 100) if trade_count > 0 else 0
        
        # Sharpe Ratio
        if trade_count >= 3:
            mean_pl = aid_data['Net_PL'].mean()
            std_pl = aid_data['Net_PL'].std()
            sharpe = mean_pl / std_pl if std_pl > 0 else 0
        else:
            sharpe = np.nan
        
        # Profit Expectancy
        win_trades = aid_data[aid_data['Net_PL'] > 0]['Net_PL']
        loss_trades = aid_data[aid_data['Net_PL'] < 0]['Net_PL']
        avg_win = win_trades.mean() if len(win_trades) > 0 else 0
        avg_loss = abs(loss_trades.mean()) if len(loss_trades) > 0 else 0
        win_prob = wins / trade_count if trade_count > 0 else 0
        loss_prob = losses / trade_count if trade_count > 0 else 0
        p_exp = (win_prob * avg_win) - (loss_prob * avg_loss)
        
        # Profit Factor
        gains = win_trades.sum() if len(win_trades) > 0 else 0
        total_losses = abs(loss_trades.sum()) if len(loss_trades) > 0 else 0
        pf = gains / total_losses if total_losses > 0 else (5.0 if gains > 0 else 0)
        
        # Recovery Factor (當日 MDD)
        aid_sorted = aid_data.sort_values(exec_col)
        if len(aid_sorted) >= 2:
            cumulative_pl = aid_sorted['Net_PL'].cumsum()
            equity = initial_balance + cumulative_pl
            running_max = equity.cummax()
            drawdown = equity - running_max
            max_dd = abs(drawdown.min())
            rec_f = net_pl / max_dd if max_dd > 0 else (net_pl if net_pl > 0 else 0)
        else:
            rec_f = net_pl if net_pl > 0 else 0
        
        results.append({
            'AID': aid,
            '當日盈虧': round(net_pl, 2),
            'Scalp盈虧': round(scalp_pl, 2),
            'Scalp%': round(scalp_pct, 2),
            'Sharpe': round(sharpe, 2) if not np.isnan(sharpe) else 'N/A',
            'P. Exp': round(p_exp, 2),
            'PF': round(pf, 2),
            'Rec.F': round(rec_f, 2),
            '交易筆數': trade_count,
            '勝率%': round(win_rate, 2)
        })
    
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values('當日盈虧', ascending=False).head(20)
    return result_df


# ==================== 圖表函數 ====================
def create_cumulative_pnl_chart(df, initial_balance, scalper_threshold_seconds):
    exec_col = COLUMN_MAP['execution_time']
    scalper_minutes = scalper_threshold_seconds / 60
    
    closing_df = filter_closing_trades(df)
    df_sorted = closing_df.sort_values(exec_col).copy()
    df_sorted['Date'] = df_sorted[exec_col].dt.date
    
    daily_pnl = df_sorted.groupby('Date')['Net_PL'].sum().reset_index()
    daily_pnl.columns = ['Date', 'Daily_PL']
    daily_pnl = daily_pnl.sort_values('Date')
    daily_pnl['Cumulative_PL'] = daily_pnl['Daily_PL'].cumsum()
    
    scalper_df = df_sorted[df_sorted['Hold_Seconds'] < scalper_threshold_seconds].copy()
    if not scalper_df.empty:
        scalper_daily_pnl = scalper_df.groupby('Date')['Net_PL'].sum().reset_index()
        scalper_daily_pnl.columns = ['Date', 'Scalper_Daily_PL']
    else:
        scalper_daily_pnl = pd.DataFrame({'Date': daily_pnl['Date'], 'Scalper_Daily_PL': 0})
    
    merged_df = daily_pnl.merge(scalper_daily_pnl, on='Date', how='left')
    merged_df['Scalper_Daily_PL'] = merged_df['Scalper_Daily_PL'].fillna(0)
    merged_df['Scalper_Cumulative_PL'] = merged_df['Scalper_Daily_PL'].cumsum()
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Cumulative_PL'], mode='lines+markers', name='整體累計盈虧', line=dict(color='#2E86AB', width=2.5), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Scalper_Cumulative_PL'], mode='lines+markers', name=f'Scalper 累計盈虧 (<{scalper_minutes:.0f}分鐘)', line=dict(color='#F39C12', width=2.5, dash='dot'), marker=dict(size=6, symbol='diamond')))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1.5)
    fig.update_layout(title=dict(text='📈 累計淨盈虧走勢：整體 vs. Scalper', font=dict(size=16)), xaxis_title='日期', yaxis_title='累計淨盈虧 ($)', height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode='x unified', plot_bgcolor='rgba(248,249,250,1)')
    
    total_pnl = merged_df['Cumulative_PL'].iloc[-1] if len(merged_df) > 0 else 0
    scalper_pnl = merged_df['Scalper_Cumulative_PL'].iloc[-1] if len(merged_df) > 0 else 0
    return fig, {'total_pnl': total_pnl, 'scalper_pnl': scalper_pnl}


def create_trading_style_pie(df, title="交易風格分佈"):
    """創建交易風格圓餅圖（Donut Chart）"""
    closing_df = filter_closing_trades(df)
    
    if 'Hold_Minutes' not in closing_df.columns or closing_df['Hold_Minutes'].isna().all():
        return None
    
    closing_df = closing_df.copy()
    closing_df['Trading_Style'] = closing_df['Hold_Minutes'].apply(classify_trading_style)
    
    style_counts = closing_df['Trading_Style'].value_counts().reset_index()
    style_counts.columns = ['風格', '筆數']
    
    # 確保順序
    style_order = ['極短線 (Scalp)', '短線 (Intraday)', '中線 (Day Trade)', '長線 (Swing)']
    style_counts['風格'] = pd.Categorical(style_counts['風格'], categories=style_order, ordered=True)
    style_counts = style_counts.sort_values('風格')
    
    colors = [STYLE_COLORS.get(s, '#95a5a6') for s in style_counts['風格']]
    
    fig = px.pie(
        style_counts,
        values='筆數',
        names='風格',
        hole=0.4,
        color='風格',
        color_discrete_map=STYLE_COLORS,
        title=title
    )
    
    fig.update_traces(textposition='inside', textinfo='label+percent', textfont_size=12)
    fig.update_layout(height=400, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
    
    return fig


def create_violin_plot_horizontal(df):
    aid_col = COLUMN_MAP['aid']
    closing_df = filter_closing_trades(df)
    aid_pl = closing_df.groupby(aid_col)['Net_PL'].sum().reset_index()
    aid_pl.columns = ['AID', 'Net_PL']
    
    Q1_pct = aid_pl['Net_PL'].quantile(0.01)
    Q99_pct = aid_pl['Net_PL'].quantile(0.99)
    plot_data = aid_pl.copy()
    filtered_count = len(aid_pl[(aid_pl['Net_PL'] < Q1_pct) | (aid_pl['Net_PL'] > Q99_pct)])
    
    mean_val = plot_data['Net_PL'].mean()
    median_val = plot_data['Net_PL'].median()
    std_val = plot_data['Net_PL'].std()
    q25 = plot_data['Net_PL'].quantile(0.25)
    q75 = plot_data['Net_PL'].quantile(0.75)
    
    fig = go.Figure()
    fig.add_trace(go.Violin(x=plot_data['Net_PL'], y=['盈虧分布'] * len(plot_data), orientation='h', name='盈虧分布', box_visible=True, meanline_visible=True, line_color='#2C3E50', fillcolor='rgba(52, 152, 219, 0.5)', opacity=0.8, points='all', pointpos=-0.5, jitter=0.3, marker=dict(color='#3498DB', size=6, opacity=0.6), box=dict(visible=True, fillcolor='rgba(255, 255, 255, 0.8)', line=dict(color='#2C3E50', width=2)), meanline=dict(visible=True, color='#E74C3C', width=2), customdata=plot_data['AID'].values, hovertemplate='<b>AID:</b> %{customdata}<br><b>Net_PL:</b> $%{x:,.2f}<extra></extra>'))
    
    x_padding = (Q99_pct - Q1_pct) * 0.1
    x_range = [Q1_pct - x_padding, Q99_pct + x_padding]
    if x_range[0] <= 0 <= x_range[1]:
        fig.add_vline(x=0, line_color="black", line_width=3)
    
    fig.update_layout(title=dict(text=f'🎻 盈虧分佈小提琴圖（水平）| 已過濾 {filtered_count} 位極端值', font=dict(size=16)), height=600, xaxis=dict(title='累計淨盈虧 ($)', range=x_range, zeroline=True, zerolinecolor='black', zerolinewidth=3), yaxis=dict(showticklabels=False, showgrid=False), showlegend=False, plot_bgcolor='rgba(248,249,250,1)', annotations=[dict(x=0.02, y=0.98, xref='paper', yref='paper', text=f'<b>📊 統計摘要</b><br>客戶數: {len(plot_data):,}<br>平均值: ${mean_val:,.2f}<br>中位數: ${median_val:,.2f}<br>Q25: ${q25:,.2f}<br>Q75: ${q75:,.2f}', showarrow=False, font=dict(size=11), align='left', bgcolor='rgba(255,255,255,0.95)', bordercolor='#3498DB', borderwidth=2, borderpad=8)])
    return fig


def create_profit_factor_chart_colored(aid_stats_df, min_trades=10):
    pf_data = aid_stats_df[['AID', 'Profit_Factor', 'Net_PL', 'Trade_Count']].copy()
    pf_display = pf_data[pf_data['Profit_Factor'] <= 5].copy()
    
    bins = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]
    pf_display['PF_Bin'] = pd.cut(pf_display['Profit_Factor'], bins=bins, right=False)
    bin_stats = pf_display.groupby('PF_Bin', observed=True).size().reset_index(name='Count')
    bin_stats['PF_Bin_Str'] = bin_stats['PF_Bin'].astype(str)
    bin_stats['Color'] = bin_stats['PF_Bin'].apply(lambda x: '#E74C3C' if x.right <= 1.0 else '#27AE60')
    
    fig = go.Figure()
    for idx, row in bin_stats.iterrows():
        fig.add_trace(go.Bar(x=[row['PF_Bin_Str']], y=[row['Count']], marker=dict(color=row['Color'], opacity=0.75, line=dict(color='#2C3E50', width=1.5)), showlegend=False, hovertemplate=f"<b>PF 區間:</b> {row['PF_Bin_Str']}<br><b>交易者數:</b> {row['Count']}<extra></extra>"))
    
    fig.add_vline(x=3.5, line_dash="dash", line_color="red", line_width=3, annotation_text="PF=1.0", annotation_position="top")
    fig.update_layout(title=dict(text='📊 獲利因子分布 (紅=虧損, 綠=盈利)', font=dict(size=16)), xaxis=dict(title='Profit Factor 區間', tickangle=-45), yaxis_title='交易者數量', height=450, plot_bgcolor='rgba(248,249,250,1)', bargap=0.1)
    
    profitable_ratio = (pf_data['Profit_Factor'] > 1.0).sum() / len(pf_data) * 100 if len(pf_data) > 0 else 0
    elite_traders = pf_data[(pf_data['Profit_Factor'] > 2.0) & (pf_data['Trade_Count'] >= min_trades)].sort_values('Profit_Factor', ascending=False).copy()
    elite_traders['AID'] = elite_traders['AID'].astype(str)
    elite_traders['Net_PL'] = elite_traders['Net_PL'].apply(lambda x: f"${x:,.2f}")
    elite_traders = elite_traders.rename(columns={'Profit_Factor': 'Profit Factor', 'Trade_Count': '交易筆數'})
    
    return fig, profitable_ratio, elite_traders[['AID', 'Profit Factor', 'Net_PL', '交易筆數']]


def create_risk_return_scatter(aid_stats_df, initial_balance):
    scatter_df = aid_stats_df.copy()
    min_size, max_size = 10, 50
    if scatter_df['Trade_Volume'].max() > scatter_df['Trade_Volume'].min():
        scatter_df['Size'] = min_size + (scatter_df['Trade_Volume'] - scatter_df['Trade_Volume'].min()) / (scatter_df['Trade_Volume'].max() - scatter_df['Trade_Volume'].min()) * (max_size - min_size)
    else:
        scatter_df['Size'] = 20
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scatter_df['MDD_Pct'], y=scatter_df['Net_PL'], mode='markers', marker=dict(size=scatter_df['Size'], color=scatter_df['Net_PL'], colorscale=['#E74C3C', '#F39C12', '#27AE60'], showscale=True, colorbar=dict(title='Net P/L ($)')), customdata=np.column_stack((scatter_df['AID'], scatter_df['Trade_Count'], scatter_df['Win_Rate'])), hovertemplate='<b>AID:</b> %{customdata[0]}<br><b>淨盈虧:</b> $%{y:,.2f}<br><b>MDD:</b> %{x:.2f}%<br><b>勝率:</b> %{customdata[2]:.1f}%<extra></extra>'))
    fig.update_layout(title=dict(text=f'🎯 風險回報矩陣 (初始資金: ${initial_balance:,})', font=dict(size=16)), xaxis=dict(title='最大回撤 MDD (%)', range=[0, 100]), yaxis_title='總盈虧 (Net P/L $)', height=550, plot_bgcolor='rgba(248,249,250,1)')
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.add_vline(x=50, line_dash="dash", line_color="gray", line_width=1)
    return fig


def create_hold_time_vs_winrate(aid_stats_df, scalper_threshold_seconds):
    scalper_minutes = scalper_threshold_seconds / 60
    plot_df = aid_stats_df[aid_stats_df['Avg_Hold_Seconds'] > 0].copy()
    if plot_df.empty:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df['Avg_Hold_Seconds'], y=plot_df['Win_Rate'], mode='markers', marker=dict(size=10, color=plot_df['Net_PL'], colorscale=['#E74C3C', '#F39C12', '#27AE60'], showscale=True, colorbar=dict(title='Net P/L ($)')), customdata=np.column_stack((plot_df['AID'], plot_df['Trade_Count'], plot_df['Net_PL'])), hovertemplate='<b>AID:</b> %{customdata[0]}<br><b>平均持倉秒數:</b> %{x:,.0f}<br><b>勝率:</b> %{y:.1f}%<extra></extra>'))
    fig.add_vline(x=scalper_threshold_seconds, line_dash="dash", line_color="red", line_width=2, annotation_text=f"Scalper ({scalper_minutes:.0f}分鐘)")
    fig.update_layout(title='⏱️ 持倉時間 vs 勝率', xaxis_title='平均持倉秒數', yaxis=dict(title='勝率 (%)', range=[0, 100]), height=500, plot_bgcolor='rgba(248,249,250,1)')
    return fig


def create_daily_pnl_chart(df):
    exec_col = COLUMN_MAP['execution_time']
    closing_df = filter_closing_trades(df)
    df_daily = closing_df.copy()
    df_daily['Date'] = df_daily[exec_col].dt.date
    daily_pnl = df_daily.groupby('Date')['Net_PL'].sum().reset_index()
    daily_pnl.columns = ['日期', '每日盈虧']
    daily_pnl = daily_pnl.sort_values('日期')
    colors = ['#27AE60' if x > 0 else '#E74C3C' for x in daily_pnl['每日盈虧']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily_pnl['日期'], y=daily_pnl['每日盈虧'], marker_color=colors, hovertemplate='<b>日期:</b> %{x}<br><b>淨盈虧:</b> $%{y:,.2f}<extra></extra>'))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(title='📅 每日盈虧柱狀圖', xaxis_title='日期', yaxis_title='淨盈虧 ($)', height=400, plot_bgcolor='rgba(248,249,250,1)')
    return fig


def get_top_scalpers(aid_stats_df, n=10):
    scalpers = aid_stats_df[aid_stats_df['Scalper_Count'] > 0].copy()
    if scalpers.empty:
        return pd.DataFrame()
    top_scalpers = scalpers.nlargest(n, 'Scalper_Count')[['AID', 'Scalper_Count', 'Scalper_PL', 'Win_Rate', 'Avg_Hold_Seconds', 'Main_Symbol']].copy()
    top_scalpers.columns = ['AID', '交易筆數', '總盈虧', '勝率(%)', '平均持倉秒數', '主要品種']
    top_scalpers['總盈虧'] = top_scalpers['總盈虧'].apply(lambda x: f"${x:,.2f}")
    top_scalpers['平均持倉秒數'] = top_scalpers['平均持倉秒數'].round(1)
    return top_scalpers


def create_client_cumulative_chart(cumulative_df, scalper_minutes):
    exec_col = COLUMN_MAP['execution_time']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cumulative_df[exec_col], y=cumulative_df['Cumulative_PL'], mode='lines', name='累計總盈虧', line=dict(color='#2E86AB', width=2)))
    fig.add_trace(go.Scatter(x=cumulative_df[exec_col], y=cumulative_df['Scalper_Cumulative_PL'], mode='lines', name=f'Scalper 盈虧 (<{scalper_minutes}分鐘)', line=dict(color='#F39C12', width=2, dash='dot')))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(title='📈 個人累計盈虧走勢', xaxis_title='時間', yaxis_title='累計盈虧 ($)', height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), plot_bgcolor='rgba(248,249,250,1)')
    return fig


def create_symbol_pie_chart(symbol_dist):
    if symbol_dist.empty:
        return None
    fig = px.pie(symbol_dist, values='Count', names='Symbol', title='🥧 交易品種分佈', hole=0.3)
    fig.update_layout(height=400)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def create_daily_product_pie(day_df):
    """創建當日產品盈虧圓餅圖"""
    instrument_col = COLUMN_MAP['instrument']
    closing_df = filter_closing_trades(day_df)
    
    if instrument_col not in closing_df.columns:
        return None
    
    product_pnl = closing_df.groupby(instrument_col)['Net_PL'].sum().reset_index()
    product_pnl.columns = ['產品', '盈虧']
    product_pnl = product_pnl.sort_values('盈虧', ascending=False)
    
    fig = px.pie(product_pnl, values='盈虧', names='產品', title='🥧 當日產品盈虧分佈', hole=0.3)
    fig.update_layout(height=400)
    fig.update_traces(textposition='inside', textinfo='label+percent')
    return fig


# ==================== 個別客戶數據提取 ====================
def get_client_details(df, aid, initial_balance, scalper_threshold_seconds):
    aid_col = COLUMN_MAP['aid']
    exec_col = COLUMN_MAP['execution_time']
    instrument_col = COLUMN_MAP['instrument']
    closed_pl_col = COLUMN_MAP['closed_pl']
    
    closing_df = filter_closing_trades(df)
    client_df = closing_df[closing_df[aid_col] == str(aid)].copy()
    if client_df.empty:
        return None
    
    net_pl = client_df['Net_PL'].sum()
    trade_count = len(client_df)
    wins = (client_df['Net_PL'] > 0).sum()
    win_rate = (wins / trade_count * 100) if trade_count > 0 else 0
    avg_hold_seconds = client_df['Hold_Seconds'].mean()
    avg_hold_seconds = avg_hold_seconds if pd.notna(avg_hold_seconds) else 0
    
    # Profit Factor
    gains = client_df[client_df[closed_pl_col] > 0][closed_pl_col].sum()
    losses = abs(client_df[client_df[closed_pl_col] < 0][closed_pl_col].sum())
    if losses == 0 and gains > 0:
        profit_factor = 5.0
    elif gains == 0:
        profit_factor = 0.0
    else:
        profit_factor = gains / losses
    
    client_sorted = client_df.sort_values(exec_col).copy()
    client_sorted['Cumulative_PL'] = client_sorted['Net_PL'].cumsum()
    scalper_mask = client_sorted['Hold_Seconds'] < scalper_threshold_seconds
    client_sorted['Scalper_PL'] = np.where(scalper_mask, client_sorted['Net_PL'], 0)
    client_sorted['Scalper_Cumulative_PL'] = client_sorted['Scalper_PL'].cumsum()
    
    if instrument_col in client_df.columns:
        symbol_dist = client_df.groupby(instrument_col)['Net_PL'].count().reset_index()
        symbol_dist.columns = ['Symbol', 'Count']
    else:
        symbol_dist = pd.DataFrame()
    
    # 深度行為分析
    behavioral_stats = calculate_deep_behavioral_stats(client_df, scalper_threshold_seconds)
    
    return {
        'net_pl': net_pl, 'trade_count': trade_count, 'win_rate': win_rate,
        'avg_hold_seconds': avg_hold_seconds, 'profit_factor': profit_factor,
        'cumulative_df': client_sorted[[exec_col, 'Cumulative_PL', 'Scalper_Cumulative_PL']],
        'symbol_dist': symbol_dist,
        'client_df': client_df,
        'behavioral': behavioral_stats
    }


# ==================== 導出 Excel ====================
def export_to_excel(df, aid_stats_df, initial_balance, scalper_threshold_seconds):
    from openpyxl.styles import Font, PatternFill, Alignment
    output = BytesIO()
    closing_df = filter_closing_trades(df)
    aid_col = COLUMN_MAP['aid']
    
    summary_data = [['指標', '數值', '說明'], ['總交易筆數', len(df), '所有交易記錄'], ['平倉交易筆數', len(closing_df), 'CLOSING 類型'], ['總客戶數', df[aid_col].nunique(), '不重複 AID'], ['總淨盈虧', round(closing_df['Net_PL'].sum(), 2), 'Net_PL 總和'], ['初始資金設定', initial_balance, '用於 MDD 計算'], ['報告時間', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '']]
    summary_df = pd.DataFrame(summary_data[1:], columns=summary_data[0])
    
    risk_return_df = aid_stats_df[['AID', 'Net_PL', 'MDD_Pct', 'Trade_Count', 'Win_Rate', 'Profit_Factor', 'Scalper_Ratio']].copy()
    risk_return_df.columns = ['AID', 'Net_PL', 'MDD(%)', 'Trade_Count', 'Win_Rate(%)', 'Profit_Factor', 'Scalper_Ratio(%)']
    risk_return_df = risk_return_df.sort_values('Net_PL', ascending=False)
    
    scalper_list = aid_stats_df[aid_stats_df['Scalper_Count'] > 0][['AID', 'Scalper_Count', 'Scalper_PL', 'Win_Rate', 'Main_Symbol']].copy()
    scalper_list.columns = ['AID', 'Scalper交易數', 'Scalper盈虧', '勝率(%)', '主要品種']
    scalper_list = scalper_list.sort_values('Scalper交易數', ascending=False)
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        risk_return_df.to_excel(writer, sheet_name='Risk_Return', index=False)
        scalper_list.to_excel(writer, sheet_name='Scalper_List', index=False)
        
        header_font = Font(bold=True, color='FFFFFF')
        header_fill = PatternFill(start_color='2E86AB', end_color='2E86AB', fill_type='solid')
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for column in ws.columns:
                max_length = max(len(str(cell.value or '')) for cell in column)
                ws.column_dimensions[column[0].column_letter].width = min(max_length + 2, 50)
            for cell in ws[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal='center')
    
    output.seek(0)
    return output


# ==================== 主程式 ====================
def main():
    st.title("📊 交易數據分析系統 v2.0")
    st.markdown("**完整重構版本：整體概覽 | 個人報告卡 | 當日英雄榜**")
    
    with st.sidebar:
        st.header("⚙️ 全域參數設定")
        initial_balance = st.number_input("💰 初始資金", value=10000, min_value=0, step=1000, help="修改此值會即時更新所有 MDD 相關圖表")
        scalper_minutes = st.number_input("⏱️ Scalper 持倉定義 (分鐘)", value=5, min_value=1, max_value=60, step=1)
        scalper_threshold_seconds = scalper_minutes * 60
        
        st.markdown("---")
        st.header("📁 數據上傳")
        uploaded_files = st.file_uploader("上傳交易數據檔案", type=['xlsx', 'csv'], accept_multiple_files=True)
        
        if uploaded_files:
            st.success(f"已上傳 {len(uploaded_files)} 個檔案")
            st.info(f"💰 初始資金: **${initial_balance:,}**")
            st.info(f"⏱️ Scalper 定義: **<{scalper_minutes} 分鐘**")
    
    if not uploaded_files:
        st.info("👈 請在左側上傳交易數據檔案開始分析")
        st.markdown("""
        ### 📋 功能說明
        - **Tab 1 整體數據概覽**: 累計盈虧、小提琴圖、獲利因子、風險矩陣、交易風格分佈
        - **Tab 2 個人報告卡**: 核心指標、多空拆解、剝頭皮診斷、連續紀錄、時間價值、自動標籤
        - **Tab 3 當日數據概覽**: 當日 KPI、產品分析、Top 20 英雄榜（含 Sharpe/P.Exp/PF/Rec.F）
        """)
        return
    
    with st.spinner("正在載入數據..."):
        df = load_and_preprocess(uploaded_files)
    
    if df is None or df.empty:
        st.error("無法載入數據，請檢查檔案格式")
        return
    
    display_df = df.copy()
    
    with st.spinner("正在計算統計數據..."):
        aid_stats_df = calculate_all_aid_stats_realtime(display_df, initial_balance, scalper_threshold_seconds)
    
    st.markdown("---")
    closing_df = filter_closing_trades(display_df)
    aid_col = COLUMN_MAP['aid']
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("總交易筆數", f"{len(display_df):,}")
    col2.metric("平倉交易筆數", f"{len(closing_df):,}")
    col3.metric("交易者數量", f"{display_df[aid_col].nunique():,}")
    col4.metric("總淨盈虧", f"${closing_df['Net_PL'].sum():,.2f}")
    
    with st.sidebar:
        st.markdown("---")
        st.header("📥 下載報表")
        excel_data = export_to_excel(display_df, aid_stats_df, initial_balance, scalper_threshold_seconds)
        st.download_button("📊 下載 Excel 報表", data=excel_data, file_name=f"trading_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    
    # ==================== 三個標籤頁 ====================
    tab1, tab2, tab3 = st.tabs(["📊 整體數據概覽", "👤 個人報告卡", "📅 當日數據概覽"])
    
    # ==================== Tab 1: 整體數據概覽 ====================
    with tab1:
        st.header("📊 整體數據概覽")
        
        st.markdown("### 1️⃣ 累計淨盈虧走勢")
        cumulative_fig, pnl_stats = create_cumulative_pnl_chart(display_df, initial_balance, scalper_threshold_seconds)
        st.plotly_chart(cumulative_fig, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("整體淨盈虧", f"${pnl_stats['total_pnl']:,.2f}")
        c2.metric(f"Scalper 淨盈虧 (<{scalper_minutes}分鐘)", f"${pnl_stats['scalper_pnl']:,.2f}")
        
        st.markdown("---")
        st.markdown("### 2️⃣ 盈虧分佈小提琴圖")
        violin_fig = create_violin_plot_horizontal(display_df)
        st.plotly_chart(violin_fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 3️⃣ 獲利因子分布")
        pf_fig, profitable_ratio, elite_traders = create_profit_factor_chart_colored(aid_stats_df)
        st.plotly_chart(pf_fig, use_container_width=True)
        st.success(f"📈 **PF > 1.0 的交易者佔比: {profitable_ratio:.1f}%**")
        with st.expander("💎 查看 PF > 2 的優質客戶名單"):
            if not elite_traders.empty:
                st.dataframe(elite_traders, use_container_width=True, hide_index=True)
            else:
                st.info("目前沒有符合條件的優質客戶")
        
        st.markdown("---")
        st.markdown("### 4️⃣ 風險回報矩陣")
        st.caption(f"⚠️ MDD 計算基於初始資金: **${initial_balance:,}**")
        scatter_fig = create_risk_return_scatter(aid_stats_df, initial_balance)
        st.plotly_chart(scatter_fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 5️⃣ 交易風格分佈")
        col_style1, col_style2 = st.columns([2, 1])
        with col_style1:
            style_pie = create_trading_style_pie(display_df, "🎨 全公司交易風格佔比")
            if style_pie:
                st.plotly_chart(style_pie, use_container_width=True)
            else:
                st.warning("無持倉時間數據")
        with col_style2:
            st.markdown("""
            **風格分類標準：**
            - 🔴 **極短線 (Scalp)**: < 5 分鐘
            - 🟠 **短線 (Intraday)**: 5-60 分鐘
            - 🔵 **中線 (Day Trade)**: 1-24 小時
            - 🟢 **長線 (Swing)**: > 24 小時
            """)
        
        st.markdown("---")
        st.markdown("### 6️⃣ 持倉時間 vs 勝率")
        hold_fig = create_hold_time_vs_winrate(aid_stats_df, scalper_threshold_seconds)
        if hold_fig:
            st.plotly_chart(hold_fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 7️⃣ 每日盈虧柱狀圖")
        st.plotly_chart(create_daily_pnl_chart(display_df), use_container_width=True)
        
        st.markdown("---")
        st.markdown(f"### 8️⃣ Top 10 Scalpers (<{scalper_minutes}分鐘)")
        top_scalpers = get_top_scalpers(aid_stats_df)
        if not top_scalpers.empty:
            st.dataframe(top_scalpers, use_container_width=True, hide_index=True)
        else:
            st.info("無符合條件的 Scalper")
    
    # ==================== Tab 2: 個人報告卡 ====================
    with tab2:
        st.header("👤 個人報告卡 (Report Card)")
        
        all_aids = sorted(aid_stats_df['AID'].unique().tolist())
        selected_aid = st.selectbox("🔍 選擇或搜尋 AID", options=all_aids, index=0 if all_aids else None)
        
        if selected_aid:
            client_data = get_client_details(display_df, selected_aid, initial_balance, scalper_threshold_seconds)
            
            if client_data:
                behavioral = client_data['behavioral']
                
                # ========== 頂部核心指標 ==========
                st.markdown("---")
                st.markdown("### 🎯 核心指標 (Core Stats)")
                core_cols = st.columns(4)
                with core_cols[0]:
                    st.metric("🆔 AID", selected_aid)
                with core_cols[1]:
                    pl_color = "🟢" if client_data['net_pl'] >= 0 else "🔴"
                    st.metric(f"{pl_color} 總盈虧", f"${client_data['net_pl']:,.2f}")
                with core_cols[2]:
                    st.metric("🎯 勝率", f"{client_data['win_rate']:.2f}%")
                with core_cols[3]:
                    st.metric("📊 獲利因子 (PF)", f"{client_data['profit_factor']:.2f}")
                
                # ========== 第二列：行為特徵對陣 ==========
                st.markdown("---")
                st.markdown("### ⚔️ 行為特徵對陣 (Behavioral Battle)")
                battle_col1, battle_col2 = st.columns(2)
                
                with battle_col1:
                    st.markdown("#### ⚔️ 多空拆解 (BUY vs SELL)")
                    buy_sell_data = {
                        '方向': ['🟢 BUY', '🔴 SELL'],
                        '筆數佔比': [f"{behavioral['buy_ratio']:.1f}%", f"{behavioral['sell_ratio']:.1f}%"],
                        '總盈虧': [f"${behavioral['buy_pl']:,.2f}", f"${behavioral['sell_pl']:,.2f}"],
                        '勝率': [f"{behavioral['buy_winrate']:.1f}%", f"{behavioral['sell_winrate']:.1f}%"]
                    }
                    st.dataframe(pd.DataFrame(buy_sell_data), use_container_width=True, hide_index=True)
                
                with battle_col2:
                    st.markdown("#### ⚡ 剝頭皮診斷 (Scalping)")
                    scalp_data = {
                        '指標': ['Scalp 筆數佔比', 'Scalp 盈虧貢獻度', 'Scalp 勝率'],
                        '數值': [f"{behavioral['scalp_ratio']:.1f}%", f"{behavioral['scalp_contribution']:.1f}%", f"{behavioral['scalp_winrate']:.1f}%"]
                    }
                    st.dataframe(pd.DataFrame(scalp_data), use_container_width=True, hide_index=True)
                
                # ========== 第三列：穩定性與效率 ==========
                st.markdown("---")
                st.markdown("### 📈 穩定性與效率 (Stability & Efficiency)")
                stab_col1, stab_col2 = st.columns(2)
                
                with stab_col1:
                    st.markdown("#### 📈 連續紀錄 (Streak Stats)")
                    streak_data = {
                        '類型': ['🏆 連續獲利', '💔 連續虧損'],
                        '最高次數': [f"{behavioral['max_win_streak']} 次", f"{behavioral['max_loss_streak']} 次"],
                        '最大金額': [f"${behavioral['max_streak_profit']:,.2f}", f"${behavioral['max_streak_loss']:,.2f}"]
                    }
                    st.dataframe(pd.DataFrame(streak_data), use_container_width=True, hide_index=True)
                
                with stab_col2:
                    st.markdown("#### ⏱️ 時間價值 (Time Value)")
                    time_data = {
                        '指標': ['平均持倉時間', '平均持倉天數', '平均分鐘獲利'],
                        '數值': [behavioral['avg_hold_formatted'], f"{behavioral['avg_hold_days']:.2f} 天", f"${behavioral['profit_per_minute']:.4f}/分鐘"]
                    }
                    st.dataframe(pd.DataFrame(time_data), use_container_width=True, hide_index=True)
                
                # ========== 底部風格標籤 ==========
                st.markdown("---")
                st.markdown("### 🏷️ 自動生成標籤 (Automated Tags)")
                tags = []
                if behavioral['scalp_ratio'] > 50:
                    tags.append("🔥 高頻型")
                if behavioral['buy_ratio'] > 70:
                    tags.append("⚖️ 偏多型")
                elif behavioral['buy_ratio'] < 30:
                    tags.append("⚖️ 偏空型")
                if client_data['win_rate'] > 65:
                    tags.append("🎯 高準度")
                if client_data['profit_factor'] > 2:
                    tags.append("💰 高效益")
                if behavioral['max_win_streak'] >= 5:
                    tags.append("🏆 連勝達人")
                
                if tags:
                    st.markdown(" ".join([f"`{tag}`" for tag in tags]))
                else:
                    st.markdown("`📊 一般交易型態`")
                
                # ========== 走勢圖與風格分佈 ==========
                st.markdown("---")
                st.markdown("### 📊 視覺化分析")
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.plotly_chart(create_client_cumulative_chart(client_data['cumulative_df'], scalper_minutes), use_container_width=True)
                
                with chart_col2:
                    # 個人交易風格圓餅圖
                    personal_style_pie = create_trading_style_pie(client_data['client_df'], f"🎨 {selected_aid} 交易風格分佈")
                    if personal_style_pie:
                        st.plotly_chart(personal_style_pie, use_container_width=True)
                    else:
                        pie = create_symbol_pie_chart(client_data['symbol_dist'])
                        if pie:
                            st.plotly_chart(pie, use_container_width=True)
            else:
                st.warning(f"找不到 AID: {selected_aid}")
        else:
            st.info("請選擇一個 AID")
    
    # ==================== Tab 3: 當日數據概覽 ====================
    with tab3:
        st.header("📅 當日數據概覽")
        
        exec_col = COLUMN_MAP['execution_time']
        closing_df = filter_closing_trades(display_df)
        
        # 自動抓取最新日期
        latest_date = closing_df[exec_col].dt.date.max()
        st.info(f"📆 分析基準日期：**{latest_date}**（數據中最新日期）")
        
        # 篩選當日數據
        day_df = closing_df[closing_df[exec_col].dt.date == latest_date].copy()
        
        if day_df.empty:
            st.warning("當日無交易數據")
        else:
            # ========== 當日關鍵指標 ==========
            st.markdown("### 📊 當日關鍵指標 (Daily KPIs)")
            kpi_cols = st.columns(4)
            
            day_total_pl = day_df['Net_PL'].sum()
            day_trade_count = len(day_df)
            day_active_accounts = day_df[aid_col].nunique()
            day_wins = (day_df['Net_PL'] > 0).sum()
            day_win_rate = (day_wins / day_trade_count * 100) if day_trade_count > 0 else 0
            
            with kpi_cols[0]:
                delta_color = "normal" if day_total_pl >= 0 else "inverse"
                st.metric("當日總盈虧", f"${day_total_pl:,.2f}", delta=f"{'盈利' if day_total_pl >= 0 else '虧損'}", delta_color=delta_color)
            with kpi_cols[1]:
                st.metric("當日總交易筆數", f"{day_trade_count:,}")
            with kpi_cols[2]:
                st.metric("當日活躍帳號數", f"{day_active_accounts:,}")
            with kpi_cols[3]:
                st.metric("當日整體勝率", f"{day_win_rate:.1f}%")
            
            st.markdown("---")
            
            # ========== 產品分析 ==========
            st.markdown("### 🥧 產品分析 (Product Analysis)")
            prod_col1, prod_col2 = st.columns(2)
            
            with prod_col1:
                product_pie = create_daily_product_pie(day_df)
                if product_pie:
                    st.plotly_chart(product_pie, use_container_width=True)
                else:
                    st.info("無產品數據")
            
            with prod_col2:
                st.markdown("#### 📊 當日產品交易量 Top 5")
                instrument_col = COLUMN_MAP['instrument']
                if instrument_col in day_df.columns:
                    volume_col = COLUMN_MAP['volume']
                    if volume_col in day_df.columns:
                        product_volume = day_df.groupby(instrument_col)[volume_col].sum().reset_index()
                    else:
                        product_volume = day_df.groupby(instrument_col).size().reset_index(name=volume_col)
                    product_volume.columns = ['產品', '成交量']
                    product_volume = product_volume.sort_values('成交量', ascending=False).head(5)
                    st.dataframe(product_volume, use_container_width=True, hide_index=True)
                else:
                    st.info("無產品數據")
            
            st.markdown("---")
            
            # ========== 當日 Top 20 盈利英雄榜 ==========
            st.markdown("### 🏆 當日 Top 20 盈利英雄榜")
            st.caption(f"篩選條件：當日 Net_PL > 0 | Scalper 定義: <{scalper_minutes} 分鐘")
            
            hero_df = calculate_daily_advanced_metrics(day_df, initial_balance, scalper_threshold_seconds)
            
            if not hero_df.empty:
                # 格式化顯示
                display_hero = hero_df.copy()
                
                # 添加 Scalp% emoji
                display_hero['Scalp%'] = display_hero['Scalp%'].apply(lambda x: f"🔥 {x:.1f}%" if x > 80 else f"{x:.1f}%")
                
                # P.Exp 顏色標記
                display_hero['P. Exp'] = display_hero['P. Exp'].apply(lambda x: f"🟢 {x:.2f}" if x > 0 else f"🔴 {x:.2f}")
                
                # 金額格式化
                display_hero['當日盈虧'] = display_hero['當日盈虧'].apply(lambda x: f"${x:,.2f}")
                display_hero['Scalp盈虧'] = display_hero['Scalp盈虧'].apply(lambda x: f"${x:,.2f}")
                
                st.dataframe(display_hero, use_container_width=True, hide_index=True)
                
                # 下載按鈕
                csv_data = hero_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 下載英雄榜 CSV",
                    data=csv_data,
                    file_name=f"hero_board_{latest_date}.csv",
                    mime="text/csv"
                )
            else:
                st.info("當日無盈利客戶")


if __name__ == "__main__":
    main()
