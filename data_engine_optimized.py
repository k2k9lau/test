import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime

# ==================== 常數定義 ====================
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


# ==================== 數據載入與預處理 (優化版) ====================
@st.cache_data(show_spinner=False, ttl=3600)
def load_and_preprocess(uploaded_files):
    """載入並預處理交易數據 - 優化記憶體使用"""
    dfs = []
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.endswith('.csv'):
                # 使用更高效的 CSV 讀取參數
                df = pd.read_csv(uploaded_file, parse_dates=False)
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
    
    # 清理數據
    if exec_col in df.columns:
        df = df[df[exec_col] != 'Total'].copy()
    df = df.drop_duplicates()

    # 日期轉換 - 向量化處理
    for col in ['execution_time', 'open_time']:
        if COLUMN_MAP[col] in df.columns:
            df[COLUMN_MAP[col]] = pd.to_datetime(df[COLUMN_MAP[col]], errors='coerce')

    # 數值欄位處理 - 一次性填充
    numeric_cols = ['closed_pl', 'commission', 'swap']
    for col in numeric_cols:
        if COLUMN_MAP[col] in df.columns:
            df[COLUMN_MAP[col]] = pd.to_numeric(df[COLUMN_MAP[col]], errors='coerce').fillna(0)

    # 向量化計算 Net_PL
    df['Net_PL'] = df[COLUMN_MAP['closed_pl']] + df[COLUMN_MAP['commission']] + df[COLUMN_MAP['swap']]

    # 向量化計算持倉時間
    exec_time = df[COLUMN_MAP['execution_time']]
    open_time = df[COLUMN_MAP['open_time']]
    df['Hold_Seconds'] = np.where(
        pd.notna(exec_time) & pd.notna(open_time),
        (exec_time - open_time).dt.total_seconds(),
        np.nan
    )
    df['Hold_Minutes'] = df['Hold_Seconds'] / 60

    # AID 清洗
    if COLUMN_MAP['aid'] in df.columns:
        df[COLUMN_MAP['aid']] = (
            df[COLUMN_MAP['aid']]
            .astype(str)
            .str.replace(r'\.0$', '', regex=True)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        # 優化: 轉換為 category 節省記憶體
        df[COLUMN_MAP['aid']] = df[COLUMN_MAP['aid']].astype('category')

    # 記憶體優化: 轉換其他分類欄位
    categorical_cols = ['action', 'instrument', 'side', 'business_type']
    for col_key in categorical_cols:
        if COLUMN_MAP[col_key] in df.columns:
            df[COLUMN_MAP[col_key]] = df[COLUMN_MAP[col_key]].astype('category')

    # 記憶體優化: 降低數值欄位精度 (float64 -> float32)
    float_cols = ['Net_PL', 'Hold_Seconds', 'Hold_Minutes'] + [COLUMN_MAP[c] for c in numeric_cols if COLUMN_MAP[c] in df.columns]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype('float32')
    
    if COLUMN_MAP['volume'] in df.columns:
        df[COLUMN_MAP['volume']] = pd.to_numeric(df[COLUMN_MAP['volume']], errors='coerce').fillna(0).astype('float32')

    return df


def filter_closing_trades(df):
    """過濾平倉交易"""
    action_col = COLUMN_MAP['action']
    if action_col in df.columns:
        return df[df[action_col] == 'CLOSING'].copy()
    return df


def classify_trading_style(hold_minutes):
    """分類交易風格"""
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


# ==================== 向量化輔助函數 ====================
def calculate_mdd_vectorized(group_df, initial_balance, exec_col):
    """向量化計算 MDD% - 使用 cummax 優化"""
    if len(group_df) < 2:
        return 0.0
    
    # 按時間排序並計算累積盈虧
    sorted_df = group_df.sort_values(exec_col)
    cumulative_pl = sorted_df['Net_PL'].cumsum()
    equity = initial_balance + cumulative_pl
    
    # 使用 cummax 計算運行最大值 (極高效)
    running_max = equity.cummax()
    
    # 向量化計算回撤百分比
    drawdown = np.where(running_max != 0, (equity - running_max) / running_max * 100, 0)
    mdd_pct = abs(np.min(drawdown))
    
    return mdd_pct


def calculate_sharpe_vectorized(pnl_series, min_trades=3):
    """向量化計算 Sharpe Ratio"""
    if len(pnl_series) < min_trades:
        return 0.0
    
    mean_pl = pnl_series.mean()
    std_pl = pnl_series.std()
    
    return mean_pl / std_pl if std_pl > 0 else 0.0


# ==================== 核心計算邏輯 (全向量化版本) ====================
@st.cache_data(show_spinner=False, ttl=1800)
def calculate_all_aid_stats_realtime(df, initial_balance, scalper_threshold_seconds):
    """
    計算所有 AID 的即時統計 - 完全向量化版本
    使用 groupby + agg 替代循環,大幅提升效能
    """
    aid_col = COLUMN_MAP['aid']
    volume_col = COLUMN_MAP['volume']
    exec_col = COLUMN_MAP['execution_time']
    instrument_col = COLUMN_MAP['instrument']
    closed_pl_col = COLUMN_MAP['closed_pl']

    closing_df = filter_closing_trades(df)
    
    if closing_df.empty:
        return pd.DataFrame()

    # 預先計算 Scalper mask (向量化)
    closing_df = closing_df.copy()
    closing_df['is_scalper'] = closing_df['Hold_Seconds'] < scalper_threshold_seconds
    closing_df['is_win'] = closing_df['Net_PL'] > 0
    closing_df['is_gain'] = closing_df[closed_pl_col] > 0
    closing_df['abs_loss'] = np.where(closing_df[closed_pl_col] < 0, 
                                       abs(closing_df[closed_pl_col]), 0)

    # 基礎統計 - 一次性 groupby 聚合
    basic_stats = closing_df.groupby(aid_col, observed=True).agg({
        'Net_PL': ['sum', 'mean', 'std', 'count'],
        volume_col: 'sum' if volume_col in closing_df.columns else 'count',
        'Hold_Seconds': 'mean',
        'is_win': 'sum',
        'is_scalper': ['sum', lambda x: (x * closing_df.loc[x.index, 'Net_PL']).sum()],
        closed_pl_col: [
            lambda x: x[x > 0].sum(),  # gains
            lambda x: abs(x[x < 0].sum())  # losses
        ]
    }).reset_index()

    # 展平多級列名
    basic_stats.columns = [
        'AID', 'Net_PL', 'Mean_PL', 'Std_PL', 'Trade_Count', 
        'Trade_Volume', 'Avg_Hold_Seconds', 'Wins',
        'Scalper_Count', 'Scalper_PL', 'Gains', 'Losses'
    ]

    # 計算百分位數 (Q1, Median, Q3) - 向量化
    quantiles = closing_df.groupby(aid_col, observed=True)['Net_PL'].quantile([0.25, 0.5, 0.75]).unstack()
    quantiles.columns = ['Q1', 'Median', 'Q3']
    quantiles = quantiles.reset_index()

    # 合併基礎統計與百分位數
    stats = basic_stats.merge(quantiles, on='AID')

    # 向量化計算衍生指標
    stats['Win_Rate'] = np.where(stats['Trade_Count'] > 0, 
                                  (stats['Wins'] / stats['Trade_Count'] * 100), 0)
    
    stats['Scalper_Ratio'] = np.where(stats['Trade_Count'] > 0,
                                       (stats['Scalper_Count'] / stats['Trade_Count'] * 100), 0)
    
    stats['Profit_Factor'] = np.where(stats['Losses'] > 0,
                                       stats['Gains'] / stats['Losses'],
                                       np.where(stats['Gains'] > 0, 5.0, 0.0))
    
    stats['Sharpe'] = np.where((stats['Trade_Count'] >= 3) & (stats['Std_PL'] > 0),
                                stats['Mean_PL'] / stats['Std_PL'], 0.0)

    # MDD 計算 - 使用向量化函數
    mdd_results = []
    for aid in stats['AID']:
        aid_data = closing_df[closing_df[aid_col] == aid]
        mdd_pct = calculate_mdd_vectorized(aid_data, initial_balance, exec_col)
        mdd_results.append(mdd_pct)
    
    stats['MDD_Pct'] = mdd_results

    # Main Symbol - 使用向量化 mode
    if instrument_col in closing_df.columns:
        main_symbols = closing_df.groupby(aid_col, observed=True)[instrument_col].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A'
        ).reset_index()
        main_symbols.columns = ['AID', 'Main_Symbol']
        stats = stats.merge(main_symbols, on='AID')
    else:
        stats['Main_Symbol'] = 'N/A'

    # 填充缺失值並四捨五入
    stats['Avg_Hold_Seconds'] = stats['Avg_Hold_Seconds'].fillna(0)
    
    # 格式化輸出欄位
    numeric_cols = ['Net_PL', 'Trade_Volume', 'Win_Rate', 'Avg_Hold_Seconds', 
                    'MDD_Pct', 'Profit_Factor', 'Scalper_Ratio', 'Scalper_PL', 
                    'Sharpe', 'Q1', 'Median', 'Q3']
    
    for col in numeric_cols:
        if col in stats.columns:
            stats[col] = stats[col].round(2)
    
    # 轉換整數欄位
    stats['Trade_Count'] = stats['Trade_Count'].astype(int)
    stats['Scalper_Count'] = stats['Scalper_Count'].astype(int)

    # 選擇最終輸出欄位
    final_cols = [
        'AID', 'Net_PL', 'Trade_Count', 'Trade_Volume', 'Win_Rate', 
        'Avg_Hold_Seconds', 'MDD_Pct', 'Profit_Factor', 'Scalper_Count', 
        'Scalper_Ratio', 'Scalper_PL', 'Sharpe', 'Q1', 'Median', 'Q3', 'Main_Symbol'
    ]
    
    return stats[final_cols]


@st.cache_data(show_spinner=False, ttl=1800)
def calculate_hero_metrics(data_df, initial_balance, scalper_threshold_seconds,
                          filter_positive=True, min_scalp_pct=None, min_scalp_pl=None,
                          min_pnl=None, min_winrate=None, min_sharpe=None, max_mdd=None):
    """
    統一計算英雄榜指標 - 向量化優化版本
    """
    aid_col = COLUMN_MAP['aid']
    exec_col = COLUMN_MAP['execution_time']

    closing_df = filter_closing_trades(data_df)
    
    if closing_df.empty:
        return pd.DataFrame()

    # 預計算向量化欄位
    closing_df = closing_df.copy()
    closing_df['is_scalper'] = closing_df['Hold_Seconds'] < scalper_threshold_seconds
    closing_df['is_win'] = closing_df['Net_PL'] > 0
    closing_df['is_loss'] = closing_df['Net_PL'] < 0

    # 一次性 groupby 聚合所有基礎指標
    grouped = closing_df.groupby(aid_col, observed=True).agg({
        'Net_PL': ['sum', 'mean', 'std', 'count'],
        'is_win': 'sum',
        'is_scalper': ['sum', lambda x: (x * closing_df.loc[x.index, 'Net_PL']).sum()]
    })
    
    # 展平欄位名
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    grouped = grouped.reset_index()
    grouped.columns = ['AID', 'Net_PL', 'Mean_PL', 'Std_PL', 'Trade_Count', 
                       'Wins', 'Scalp_Count', 'Scalp_PL']

    # 應用初步過濾
    if filter_positive:
        grouped = grouped[grouped['Net_PL'] > 0]
    
    if grouped.empty:
        return pd.DataFrame()

    # 計算衍生指標 (向量化)
    grouped['Win_Rate'] = np.where(grouped['Trade_Count'] > 0,
                                    (grouped['Wins'] / grouped['Trade_Count'] * 100), 0)
    
    grouped['Scalp_Pct'] = np.where(grouped['Trade_Count'] > 0,
                                     (grouped['Scalp_Count'] / grouped['Trade_Count'] * 100), 0)
    
    grouped['Sharpe'] = np.where((grouped['Trade_Count'] >= 3) & (grouped['Std_PL'] > 0),
                                  grouped['Mean_PL'] / grouped['Std_PL'], 0.0)

    # 計算 Q1, Median, Q3, IQR
    quantiles = closing_df.groupby(aid_col, observed=True)['Net_PL'].quantile([0.25, 0.5, 0.75]).unstack()
    quantiles.columns = ['Q1', 'Median', 'Q3']
    quantiles['IQR'] = quantiles['Q3'] - quantiles['Q1']
    quantiles = quantiles.reset_index()
    
    grouped = grouped.merge(quantiles, on='AID', how='left')

    # 計算 Profit Expectancy 與 Profit Factor
    win_loss_stats = []
    for aid in grouped['AID']:
        aid_data = closing_df[closing_df[aid_col] == aid]
        
        win_trades = aid_data[aid_data['Net_PL'] > 0]['Net_PL']
        loss_trades = aid_data[aid_data['Net_PL'] < 0]['Net_PL']
        
        avg_win = win_trades.mean() if len(win_trades) > 0 else 0
        avg_loss = abs(loss_trades.mean()) if len(loss_trades) > 0 else 0
        
        trade_count = len(aid_data)
        win_prob = len(win_trades) / trade_count if trade_count > 0 else 0
        loss_prob = len(loss_trades) / trade_count if trade_count > 0 else 0
        
        p_exp = (win_prob * avg_win) - (loss_prob * avg_loss)
        
        gains = win_trades.sum() if len(win_trades) > 0 else 0
        total_losses = abs(loss_trades.sum()) if len(loss_trades) > 0 else 0
        pf = gains / total_losses if total_losses > 0 else (5.0 if gains > 0 else 0.0)
        
        win_loss_stats.append({'AID': aid, 'P_Exp': p_exp, 'PF': pf})
    
    wl_df = pd.DataFrame(win_loss_stats)
    grouped = grouped.merge(wl_df, on='AID', how='left')

    # 計算 MDD 與 Recovery Factor
    mdd_rec_stats = []
    for aid in grouped['AID']:
        aid_data = closing_df[closing_df[aid_col] == aid]
        aid_sorted = aid_data.sort_values(exec_col)
        
        if len(aid_sorted) >= 2:
            cumulative_pl = aid_sorted['Net_PL'].cumsum()
            equity = initial_balance + cumulative_pl
            running_max = equity.cummax()
            drawdown = np.where(running_max != 0, (equity - running_max) / running_max * 100, 0)
            mdd_pct = abs(np.min(drawdown))
            max_dd_abs = abs((equity - running_max).min())
        else:
            mdd_pct = 0.0
            max_dd_abs = 0.0
        
        net_pl = aid_data['Net_PL'].sum()
        rec_f = net_pl / max_dd_abs if max_dd_abs > 0 else (net_pl if net_pl > 0 else 0.0)
        
        mdd_rec_stats.append({'AID': aid, 'MDD_Pct': mdd_pct, 'Rec_F': rec_f})
    
    mdd_df = pd.DataFrame(mdd_rec_stats)
    grouped = grouped.merge(mdd_df, on='AID', how='left')

    # 應用過濾條件
    if min_scalp_pct is not None:
        grouped = grouped[grouped['Scalp_Pct'] >= float(min_scalp_pct)]
    if min_scalp_pl is not None:
        grouped = grouped[grouped['Scalp_PL'] >= float(min_scalp_pl)]
    if min_pnl is not None:
        grouped = grouped[grouped['Net_PL'] >= float(min_pnl)]
    if min_winrate is not None:
        grouped = grouped[grouped['Win_Rate'] >= float(min_winrate)]
    if min_sharpe is not None:
        grouped = grouped[grouped['Sharpe'] >= float(min_sharpe)]
    if max_mdd is not None:
        grouped = grouped[grouped['MDD_Pct'] <= float(max_mdd)]

    # 格式化輸出
    result = pd.DataFrame({
        'AID': grouped['AID'].astype(str),
        '盈虧': grouped['Net_PL'].round(2),
        'Scalp盈虧': grouped['Scalp_PL'].round(2),
        'Scalp%': grouped['Scalp_Pct'].round(2),
        'Sharpe': grouped['Sharpe'].round(2),
        'MDD%': grouped['MDD_Pct'].round(2),
        'Q1': grouped['Q1'].round(2),
        'Median': grouped['Median'].round(2),
        'Q3': grouped['Q3'].round(2),
        'IQR': grouped['IQR'].round(2),
        'P. Exp': grouped['P_Exp'].round(2),
        'PF': grouped['PF'].round(2),
        'Rec.F': grouped['Rec_F'].round(2),
        '勝率%': grouped['Win_Rate'].round(2),
        '筆數': grouped['Trade_Count'].astype(int)
    })

    # 排序並取前 20
    result = result.sort_values('盈虧', ascending=False).head(20).reset_index(drop=True)
    
    return result


@st.cache_data(show_spinner=False, ttl=1800)
def calculate_product_scalp_breakdown(day_df, scalper_threshold_seconds):
    """計算產品 Scalp 拆解 - 向量化版本"""
    instrument_col = COLUMN_MAP['instrument']
    closing_df = filter_closing_trades(day_df)

    if instrument_col not in closing_df.columns or closing_df.empty:
        return None, None

    # 預計算 scalp mask
    closing_df = closing_df.copy()
    closing_df['is_scalp'] = closing_df['Hold_Seconds'] < scalper_threshold_seconds

    # 向量化 groupby 聚合
    stats = closing_df.groupby(instrument_col, observed=True).agg({
        'Net_PL': 'sum',
        'is_scalp': ['sum', 'count', lambda x: (x * closing_df.loc[x.index, 'Net_PL']).sum()]
    })
    
    stats.columns = ['Total_PL', 'Scalp_Count', 'Total_Count', 'Scalp_PL']
    stats = stats.reset_index()
    stats.columns = ['Product', 'Total_PL', 'Scalp_Count', 'Total_Count', 'Scalp_PL']
    
    stats['NonScalp_PL'] = stats['Total_PL'] - stats['Scalp_PL']
    stats['Scalp_Pct'] = np.where(stats['Total_Count'] > 0,
                                   (stats['Scalp_Count'] / stats['Total_Count'] * 100), 0)

    # 分離盈利與虧損產品
    profit_products = stats[stats['Total_PL'] > 0].nlargest(5, 'Total_PL')
    loss_products = stats[stats['Total_PL'] < 0].nsmallest(5, 'Total_PL')

    return profit_products if not profit_products.empty else None, \
           loss_products if not loss_products.empty else None


def calculate_deep_behavioral_stats(client_df, scalper_threshold_seconds):
    """計算深度行為統計 - 優化版"""
    side_col = COLUMN_MAP['side']

    total_trades = len(client_df)
    if total_trades == 0:
        return {k: 0 for k in ['max_win_streak', 'max_loss_streak', 'max_streak_profit',
                                'max_streak_loss', 'buy_count', 'sell_count', 'buy_ratio',
                                'sell_ratio', 'buy_pl', 'sell_pl', 'buy_winrate', 
                                'sell_winrate', 'scalp_count', 'scalp_ratio', 'scalp_pl',
                                'scalp_contribution', 'scalp_winrate', 'avg_hold_formatted',
                                'avg_hold_days', 'profit_per_minute', 'q1', 'median', 'q3', 'iqr']}

    total_pl = client_df['Net_PL'].sum()
    total_minutes = client_df['Hold_Minutes'].sum() if 'Hold_Minutes' in client_df.columns else 0
    total_minutes = total_minutes if pd.notna(total_minutes) else 0

    # 連續紀錄計算 (向量化)
    pnl_signs = (client_df['Net_PL'] > 0).astype(int).values
    
    # 使用 numpy diff 找出變化點
    changes = np.concatenate(([0], np.where(np.diff(pnl_signs) != 0)[0] + 1, [len(pnl_signs)]))
    streak_lengths = np.diff(changes)
    streak_types = pnl_signs[changes[:-1]]
    
    win_streaks = streak_lengths[streak_types == 1]
    loss_streaks = streak_lengths[streak_types == 0]
    
    max_win_streak = int(win_streaks.max()) if len(win_streaks) > 0 else 0
    max_loss_streak = int(loss_streaks.max()) if len(loss_streaks) > 0 else 0

    # 連續盈虧金額
    client_sorted = client_df.sort_values(COLUMN_MAP['execution_time']).copy()
    client_sorted['streak_group'] = (client_sorted['Net_PL'] > 0).ne(
        (client_sorted['Net_PL'] > 0).shift()
    ).cumsum()
    streak_sums = client_sorted.groupby('streak_group')['Net_PL'].sum()
    max_streak_profit = float(streak_sums.max()) if not streak_sums.empty else 0
    max_streak_loss = float(streak_sums.min()) if not streak_sums.empty else 0

    # 多空統計 (向量化)
    if side_col in client_df.columns:
        buy_mask = client_df[side_col] == 'BUY'
        sell_mask = client_df[side_col] == 'SELL'
        
        buy_count = int(buy_mask.sum())
        sell_count = int(sell_mask.sum())
        
        buy_pl = float(client_df.loc[buy_mask, 'Net_PL'].sum()) if buy_count > 0 else 0
        sell_pl = float(client_df.loc[sell_mask, 'Net_PL'].sum()) if sell_count > 0 else 0
        
        buy_wins = int((client_df.loc[buy_mask, 'Net_PL'] > 0).sum()) if buy_count > 0 else 0
        sell_wins = int((client_df.loc[sell_mask, 'Net_PL'] > 0).sum()) if sell_count > 0 else 0
    else:
        buy_count = sell_count = 0
        buy_pl = sell_pl = 0
        buy_wins = sell_wins = 0

    buy_ratio = (buy_count / total_trades * 100) if total_trades > 0 else 0
    sell_ratio = (sell_count / total_trades * 100) if total_trades > 0 else 0
    buy_winrate = (buy_wins / buy_count * 100) if buy_count > 0 else 0
    sell_winrate = (sell_wins / sell_count * 100) if sell_count > 0 else 0

    # Scalper 統計 (向量化)
    scalp_mask = client_df['Hold_Seconds'] < scalper_threshold_seconds
    scalp_count = int(scalp_mask.sum())
    scalp_pl = float(client_df.loc[scalp_mask, 'Net_PL'].sum()) if scalp_count > 0 else 0
    scalp_wins = int((client_df.loc[scalp_mask, 'Net_PL'] > 0).sum()) if scalp_count > 0 else 0

    scalp_ratio = (scalp_count / total_trades * 100) if total_trades > 0 else 0
    scalp_contribution = (scalp_pl / total_pl * 100) if total_pl != 0 else 0
    scalp_winrate = (scalp_wins / scalp_count * 100) if scalp_count > 0 else 0

    # Box Plot 指標
    q1 = float(client_df['Net_PL'].quantile(0.25))
    median = float(client_df['Net_PL'].median())
    q3 = float(client_df['Net_PL'].quantile(0.75))
    iqr = q3 - q1

    # 時間效率
    avg_minutes = total_minutes / total_trades if total_trades > 0 else 0
    profit_per_minute = total_pl / total_minutes if total_minutes > 0 else 0
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
        'q1': q1,
        'median': median,
        'q3': q3,
        'iqr': iqr
    }


def get_client_details(df, aid, initial_balance, scalper_threshold_seconds):
    """獲取客戶詳細資料 (Tab 2 用)"""
    aid_col = COLUMN_MAP['aid']
    exec_col = COLUMN_MAP['execution_time']
    instrument_col = COLUMN_MAP['instrument']
    closed_pl_col = COLUMN_MAP['closed_pl']

    closing_df = filter_closing_trades(df)
    client_df = closing_df[closing_df[aid_col] == str(aid)].copy()
    
    if client_df.empty:
        return None

    # 基礎統計
    net_pl = float(client_df['Net_PL'].sum())
    trade_count = len(client_df)
    wins = int((client_df['Net_PL'] > 0).sum())
    win_rate = (wins / trade_count * 100) if trade_count > 0 else 0
    
    avg_hold_seconds = client_df['Hold_Seconds'].mean()
    avg_hold_seconds = float(avg_hold_seconds) if pd.notna(avg_hold_seconds) else 0

    # Profit Factor
    gains = float(client_df[client_df[closed_pl_col] > 0][closed_pl_col].sum())
    losses = float(abs(client_df[client_df[closed_pl_col] < 0][closed_pl_col].sum()))
    profit_factor = gains / losses if losses > 0 else (5.0 if gains > 0 else 0)

    # Sharpe
    sharpe = calculate_sharpe_vectorized(client_df['Net_PL'])

    # MDD
    mdd_pct = calculate_mdd_vectorized(client_df, initial_balance, exec_col)

    # 累積 PL 計算
    client_sorted = client_df.sort_values(exec_col).copy()
    client_sorted['Cumulative_PL'] = client_sorted['Net_PL'].cumsum()
    
    scalper_mask = client_sorted['Hold_Seconds'] < scalper_threshold_seconds
    client_sorted['Scalper_PL'] = np.where(scalper_mask, client_sorted['Net_PL'], 0)
    client_sorted['Scalper_Cumulative_PL'] = client_sorted['Scalper_PL'].cumsum()

    # Symbol 分佈
    if instrument_col in client_df.columns:
        symbol_dist = client_df.groupby(instrument_col, observed=True).size().reset_index(name='Count')
        symbol_dist.columns = ['Symbol', 'Count']
    else:
        symbol_dist = pd.DataFrame()

    # 行為統計
    behavioral_stats = calculate_deep_behavioral_stats(client_df, scalper_threshold_seconds)

    return {
        'net_pl': net_pl,
        'trade_count': trade_count,
        'win_rate': win_rate,
        'avg_hold_seconds': avg_hold_seconds,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
        'mdd_pct': mdd_pct,
        'cumulative_df': client_sorted[[exec_col, 'Cumulative_PL', 'Scalper_Cumulative_PL']],
        'symbol_dist': symbol_dist,
        'client_df': client_df,
        'behavioral': behavioral_stats
    }


def get_client_ranking(aid_stats_df, aid, metric='Net_PL'):
    """獲取客戶排名"""
    sorted_df = aid_stats_df.sort_values(metric, ascending=False).reset_index(drop=True)
    try:
        rank = sorted_df[sorted_df['AID'] == str(aid)].index[0] + 1
        return int(rank), len(sorted_df)
    except:
        return None, len(sorted_df)


def export_to_excel(df, aid_stats_df, initial_balance, scalper_threshold_seconds):
    """匯出至 Excel - 優化版"""
    from openpyxl.styles import Font, PatternFill, Alignment
    
    output = BytesIO()
    closing_df = filter_closing_trades(df)
    aid_col = COLUMN_MAP['aid']

    # Summary
    summary_df = pd.DataFrame([
        ['總交易筆數', len(df)],
        ['平倉交易筆數', len(closing_df)],
        ['總客戶數', df[aid_col].nunique()],
        ['總淨盈虧', round(closing_df['Net_PL'].sum(), 2)],
        ['初始資金', initial_balance]
    ], columns=['指標', '數值'])

    # Risk Return (優化: 只選取需要的欄位)
    risk_cols = ['AID', 'Net_PL', 'MDD_Pct', 'Sharpe', 'Trade_Count',
                 'Win_Rate', 'Profit_Factor', 'Scalper_Ratio', 'Q1', 'Median', 'Q3']
    
    risk_return_df = aid_stats_df[risk_cols].sort_values('Net_PL', ascending=False)

    # 寫入 Excel (優化: 減少格式設置次數)
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        risk_return_df.to_excel(writer, sheet_name='Risk_Return', index=False)
        
        # 統一格式設置
        header_font = Font(bold=True, color='FFFFFF')
        header_fill = PatternFill(start_color='2E86AB', end_color='2E86AB', fill_type='solid')
        header_align = Alignment(horizontal='center')
        
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            # 只設置第一行
            for cell in ws[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = header_align

    output.seek(0)
    return output
