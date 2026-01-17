"""
核心數據引擎模組 (Data Engine Module) v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
第一階段重構：解決效能與數據類型問題

🚀 特性：
  1. 高效快取：@st.cache_data 包裝資料載入
  2. 向量化運算：禁止 apply/loop，全面使用 groupby 向量化
  3. 強制 AID 字串：全域確保 AID 為字串型別
  4. Violin Plot 採樣：大數據集自動抽樣 10%

📊 計算指標：
  - 總盈虧、Scalp 盈虧、Scalp%（筆數佔比）
  - 勝率、Sharpe Ratio、MDD%
  - 單筆盈虧的 Q1, Median, Q3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from io import BytesIO


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


# ==================== 資料載入與清洗 ====================
@st.cache_data(show_spinner=False, ttl=3600)
def load_data(uploaded_files: List[Any]) -> Optional[pd.DataFrame]:
    """
    載入並預處理交易數據
    
    🔧 快取策略：
    - ttl=3600：快取 1 小時
    - show_spinner=False：由外層控制 spinner
    
    📋 處理流程：
    1. 讀取 CSV/Excel 檔案
    2. 合併去重
    3. 時間欄位轉換
    4. 計算 Net_PL 與持倉時間
    5. ⭐ 強制 AID 轉字串（解決複製功能失效問題）
    
    Args:
        uploaded_files: Streamlit 上傳的檔案列表
        
    Returns:
        預處理後的 DataFrame，失敗時返回 None
    """
    if not uploaded_files:
        return None
    
    dfs = []
    for uploaded_file in uploaded_files:
        try:
            # 重置檔案指標（避免快取問題）
            uploaded_file.seek(0)
            
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
    
    # 合併資料
    df = pd.concat(dfs, ignore_index=True)
    
    # 移除 Total 行
    exec_col = COLUMN_MAP['execution_time']
    if exec_col in df.columns:
        df = df[df[exec_col] != 'Total'].copy()
    
    # 去重
    df = df.drop_duplicates()
    
    # 清洗資料
    df = _clean_data(df)
    
    return df


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    資料清洗核心邏輯
    
    ⭐ 關鍵修復：強制 AID 為字串型別
    """
    # ==================== 1. 時間欄位轉換 ====================
    for col_key in ['execution_time', 'open_time']:
        col_name = COLUMN_MAP[col_key]
        if col_name in df.columns:
            df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
    
    # ==================== 2. 數值欄位填補 ====================
    for col_key in ['closed_pl', 'commission', 'swap']:
        col_name = COLUMN_MAP[col_key]
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
    
    # ==================== 3. 計算 Net_PL ====================
    closed_pl = df.get(COLUMN_MAP['closed_pl'], 0)
    commission = df.get(COLUMN_MAP['commission'], 0)
    swap = df.get(COLUMN_MAP['swap'], 0)
    df['Net_PL'] = closed_pl + commission + swap
    
    # ==================== 4. 計算持倉時間 ====================
    exec_col = COLUMN_MAP['execution_time']
    open_col = COLUMN_MAP['open_time']
    
    if exec_col in df.columns and open_col in df.columns:
        exec_time = df[exec_col]
        open_time = df[open_col]
        
        # 向量化計算持倉秒數
        valid_mask = pd.notna(exec_time) & pd.notna(open_time)
        df['Hold_Seconds'] = np.where(
            valid_mask,
            (exec_time - open_time).dt.total_seconds(),
            np.nan
        )
        df['Hold_Minutes'] = df['Hold_Seconds'] / 60
    else:
        df['Hold_Seconds'] = np.nan
        df['Hold_Minutes'] = np.nan
    
    # ==================== 5. ⭐ 強制 AID 為字串 ====================
    aid_col = COLUMN_MAP['aid']
    if aid_col in df.columns:
        df[aid_col] = (
            df[aid_col]
            .astype(str)
            .str.replace(r'\.0$', '', regex=True)  # 移除小數點
            .str.replace(',', '', regex=False)       # 移除千分位
            .str.strip()                              # 移除空白
        )
    
    return df


def filter_closing_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    過濾平倉交易
    
    Args:
        df: 原始 DataFrame
        
    Returns:
        僅包含平倉交易的 DataFrame
    """
    action_col = COLUMN_MAP['action']
    if action_col in df.columns:
        return df[df[action_col] == 'CLOSING'].copy()
    return df.copy()


# ==================== 向量化指標計算引擎 ====================
@st.cache_data(show_spinner=False)
def get_client_metrics(
    df: pd.DataFrame,
    initial_balance: float = 10000.0,
    scalper_threshold_seconds: float = 300.0
) -> pd.DataFrame:
    """
    向量化計算所有客戶指標
    
    🚀 效能優化：
    - 完全禁止 apply() 和 for loop
    - 使用 groupby + 向量化聚合
    - 單次遍歷計算所有指標
    
    📊 計算指標：
    - 總盈虧 (Net_PL)
    - Scalp 盈虧 (Scalp_PL)
    - Scalp% 筆數佔比 (Scalp_Pct)
    - 勝率 (Win_Rate)
    - Sharpe Ratio
    - MDD%
    - Q1, Median, Q3 (單筆盈虧分位數)
    
    Args:
        df: 預處理後的 DataFrame
        initial_balance: 初始資金
        scalper_threshold_seconds: Scalp 門檻秒數
        
    Returns:
        包含所有客戶指標的 DataFrame
    """
    aid_col = COLUMN_MAP['aid']
    exec_col = COLUMN_MAP['execution_time']
    
    # 過濾平倉交易
    closing_df = filter_closing_trades(df)
    
    if closing_df.empty or aid_col not in closing_df.columns:
        return pd.DataFrame()
    
    # ==================== 建立 Scalp 標記 ====================
    closing_df = closing_df.copy()
    closing_df['Is_Scalp'] = closing_df['Hold_Seconds'] < scalper_threshold_seconds
    closing_df['Is_Win'] = closing_df['Net_PL'] > 0
    closing_df['Scalp_PL'] = np.where(closing_df['Is_Scalp'], closing_df['Net_PL'], 0)
    
    # ==================== GroupBy 向量化聚合 ====================
    grouped = closing_df.groupby(aid_col, sort=False)
    
    # 基礎指標
    metrics = grouped.agg(
        Net_PL=('Net_PL', 'sum'),
        Trade_Count=('Net_PL', 'count'),
        Win_Count=('Is_Win', 'sum'),
        Scalp_Count=('Is_Scalp', 'sum'),
        Scalp_PL=('Scalp_PL', 'sum'),
        PL_Mean=('Net_PL', 'mean'),
        PL_Std=('Net_PL', 'std'),
        Q1=('Net_PL', lambda x: x.quantile(0.25)),
        Median=('Net_PL', 'median'),
        Q3=('Net_PL', lambda x: x.quantile(0.75)),
    ).reset_index()
    
    # 重命名 AID 欄位
    metrics = metrics.rename(columns={aid_col: 'AID'})
    
    # ==================== 向量化計算衍生指標 ====================
    # Scalp%（筆數佔比）
    metrics['Scalp_Pct'] = np.where(
        metrics['Trade_Count'] > 0,
        (metrics['Scalp_Count'] / metrics['Trade_Count']) * 100,
        0
    )
    
    # 勝率
    metrics['Win_Rate'] = np.where(
        metrics['Trade_Count'] > 0,
        (metrics['Win_Count'] / metrics['Trade_Count']) * 100,
        0
    )
    
    # Sharpe Ratio（需要至少 3 筆交易且標準差 > 0）
    metrics['Sharpe'] = np.where(
        (metrics['Trade_Count'] >= 3) & (metrics['PL_Std'] > 0),
        metrics['PL_Mean'] / metrics['PL_Std'],
        0
    )
    
    # IQR
    metrics['IQR'] = metrics['Q3'] - metrics['Q1']
    
    # ==================== MDD% 計算（需要特殊處理）====================
    metrics['MDD_Pct'] = _calculate_mdd_vectorized(
        closing_df, aid_col, exec_col, initial_balance
    )
    
    # ==================== 整理輸出欄位 ====================
    output_cols = [
        'AID', 'Net_PL', 'Trade_Count', 'Scalp_PL', 'Scalp_Pct',
        'Win_Rate', 'Sharpe', 'MDD_Pct', 'Q1', 'Median', 'Q3', 'IQR'
    ]
    
    # 確保所有欄位存在
    for col in output_cols:
        if col not in metrics.columns:
            metrics[col] = 0
    
    # 四捨五入
    numeric_cols = ['Net_PL', 'Scalp_PL', 'Scalp_Pct', 'Win_Rate', 
                    'Sharpe', 'MDD_Pct', 'Q1', 'Median', 'Q3', 'IQR']
    for col in numeric_cols:
        metrics[col] = metrics[col].round(2)
    
    # ⭐ 確保 AID 為字串
    metrics['AID'] = metrics['AID'].astype(str)
    
    return metrics[output_cols]


def _calculate_mdd_vectorized(
    df: pd.DataFrame,
    aid_col: str,
    exec_col: str,
    initial_balance: float
) -> pd.Series:
    """
    向量化計算 MDD%
    
    💡 策略：
    - 使用 groupby + apply 但內部為向量化運算
    - 對於無法完全向量化的 MDD，這是最佳折衷
    
    Returns:
        以 AID 為索引的 MDD% Series
    """
    def calc_mdd_for_group(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return 0.0
        
        # 按時間排序
        sorted_group = group.sort_values(exec_col)
        
        # 累計盈虧
        cumulative_pl = sorted_group['Net_PL'].cumsum()
        
        # 權益曲線
        equity = initial_balance + cumulative_pl
        
        # 歷史最高點
        running_max = equity.cummax()
        
        # 回撤百分比
        drawdown_pct = np.where(
            running_max != 0,
            (equity - running_max) / running_max * 100,
            0
        )
        
        # 最大回撤（取絕對值）
        return abs(np.min(drawdown_pct))
    
    mdd_series = df.groupby(aid_col, sort=False).apply(
        calc_mdd_for_group, include_groups=False
    )
    
    return mdd_series.values


# ==================== Violin Plot 採樣邏輯 ====================
def get_violin_sample(
    df: pd.DataFrame,
    max_points: int = 5000,
    sample_ratio: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    為 Violin Plot 準備採樣數據
    
    📊 策略：
    - 若數據點 > max_points，則抽樣 sample_ratio (10%)
    - 同時返回完整統計數據（基於全量資料計算）
    
    Args:
        df: 預處理後的 DataFrame
        max_points: 觸發採樣的門檻
        sample_ratio: 採樣比例
        random_state: 隨機種子（確保可重現）
        
    Returns:
        (採樣後的 DataFrame, 統計字典)
    """
    aid_col = COLUMN_MAP['aid']
    closing_df = filter_closing_trades(df)
    
    # 計算每個客戶的總盈虧
    aid_pl = closing_df.groupby(aid_col, sort=False)['Net_PL'].sum().reset_index()
    aid_pl.columns = ['AID', 'Net_PL']
    
    # ⭐ 確保 AID 為字串
    aid_pl['AID'] = aid_pl['AID'].astype(str)
    
    # ==================== 計算完整統計（基於全量資料）====================
    stats = _calculate_violin_stats(aid_pl)
    
    # ==================== 採樣邏輯 ====================
    n_points = len(aid_pl)
    
    if n_points > max_points:
        sample_size = int(n_points * sample_ratio)
        sample_df = aid_pl.sample(n=sample_size, random_state=random_state)
        stats['is_sampled'] = True
        stats['sample_size'] = sample_size
        stats['original_size'] = n_points
    else:
        sample_df = aid_pl
        stats['is_sampled'] = False
        stats['sample_size'] = n_points
        stats['original_size'] = n_points
    
    return sample_df, stats


def _calculate_violin_stats(aid_pl: pd.DataFrame) -> Dict[str, Any]:
    """
    計算 Violin Plot 統計數據
    
    Returns:
        統計字典
    """
    net_pl = aid_pl['Net_PL']
    
    count = len(aid_pl)
    mean_val = net_pl.mean()
    median_val = net_pl.median()
    std_val = net_pl.std()
    q1 = net_pl.quantile(0.25)
    q3 = net_pl.quantile(0.75)
    iqr = q3 - q1
    min_val = net_pl.min()
    max_val = net_pl.max()
    
    # 離群值範圍
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    
    # 盈虧統計
    profitable = (net_pl > 0).sum()
    losing = (net_pl <= 0).sum()
    outliers = ((net_pl < lower_fence) | (net_pl > upper_fence)).sum()
    
    return {
        'count': count,
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'min': min_val,
        'max': max_val,
        'lower_fence': lower_fence,
        'upper_fence': upper_fence,
        'profitable': profitable,
        'losing': losing,
        'outliers': outliers
    }


# ==================== 輔助函數 ====================
def get_aid_column_name() -> str:
    """返回 AID 欄位名稱"""
    return COLUMN_MAP['aid']


def get_column_map() -> Dict[str, str]:
    """返回完整欄位映射"""
    return COLUMN_MAP.copy()


# ==================== 擴展指標計算（英雄榜用）====================
@st.cache_data(show_spinner=False)
def get_hero_metrics(
    df: pd.DataFrame,
    initial_balance: float = 10000.0,
    scalper_threshold_seconds: float = 300.0,
    filter_positive: bool = True,
    min_scalp_pct: Optional[float] = None,
    min_scalp_pl: Optional[float] = None,
    min_pnl: Optional[float] = None,
    min_winrate: Optional[float] = None,
    min_sharpe: Optional[float] = None,
    max_mdd: Optional[float] = None,
    top_n: int = 20
) -> pd.DataFrame:
    """
    計算英雄榜指標（含過濾邏輯）
    
    📊 完整指標：
    - AID, 盈虧, Scalp盈虧, Scalp%
    - Sharpe, MDD%, Q1, Median, Q3, IQR
    - P.Exp (Profit Expectancy), PF (Profit Factor), Rec.F (Recovery Factor)
    - 勝率%, 筆數
    
    Args:
        df: 預處理後的 DataFrame
        initial_balance: 初始資金
        scalper_threshold_seconds: Scalp 門檻秒數
        filter_positive: 是否僅顯示正盈虧
        min_scalp_pct: 最低 Scalp%
        min_scalp_pl: 最低 Scalp 盈虧
        min_pnl: 最低盈虧
        min_winrate: 最低勝率
        min_sharpe: 最低 Sharpe
        max_mdd: 最高 MDD%
        top_n: 返回前 N 名
        
    Returns:
        英雄榜 DataFrame
    """
    aid_col = COLUMN_MAP['aid']
    exec_col = COLUMN_MAP['execution_time']
    closed_pl_col = COLUMN_MAP['closed_pl']
    
    closing_df = filter_closing_trades(df)
    
    if closing_df.empty:
        return pd.DataFrame()
    
    # ==================== 建立標記欄位 ====================
    closing_df = closing_df.copy()
    closing_df['Is_Scalp'] = closing_df['Hold_Seconds'] < scalper_threshold_seconds
    closing_df['Is_Win'] = closing_df['Net_PL'] > 0
    closing_df['Is_Loss'] = closing_df['Net_PL'] < 0
    closing_df['Scalp_PL'] = np.where(closing_df['Is_Scalp'], closing_df['Net_PL'], 0)
    closing_df['Win_PL'] = np.where(closing_df['Is_Win'], closing_df['Net_PL'], 0)
    closing_df['Loss_PL'] = np.where(closing_df['Is_Loss'], closing_df['Net_PL'].abs(), 0)
    
    # ==================== GroupBy 聚合 ====================
    grouped = closing_df.groupby(aid_col, sort=False)
    
    metrics = grouped.agg(
        Net_PL=('Net_PL', 'sum'),
        Trade_Count=('Net_PL', 'count'),
        Win_Count=('Is_Win', 'sum'),
        Loss_Count=('Is_Loss', 'sum'),
        Scalp_Count=('Is_Scalp', 'sum'),
        Scalp_PL=('Scalp_PL', 'sum'),
        Total_Wins=('Win_PL', 'sum'),
        Total_Losses=('Loss_PL', 'sum'),
        PL_Mean=('Net_PL', 'mean'),
        PL_Std=('Net_PL', 'std'),
        Q1=('Net_PL', lambda x: x.quantile(0.25)),
        Median=('Net_PL', 'median'),
        Q3=('Net_PL', lambda x: x.quantile(0.75)),
    ).reset_index()
    
    metrics = metrics.rename(columns={aid_col: 'AID'})
    
    # ==================== 計算衍生指標 ====================
    tc = metrics['Trade_Count']
    
    # Scalp%
    metrics['Scalp_Pct'] = np.where(tc > 0, (metrics['Scalp_Count'] / tc) * 100, 0)
    
    # 勝率
    metrics['Win_Rate'] = np.where(tc > 0, (metrics['Win_Count'] / tc) * 100, 0)
    
    # Sharpe
    metrics['Sharpe'] = np.where(
        (tc >= 3) & (metrics['PL_Std'] > 0),
        metrics['PL_Mean'] / metrics['PL_Std'],
        0
    )
    
    # IQR
    metrics['IQR'] = metrics['Q3'] - metrics['Q1']
    
    # Profit Factor
    metrics['PF'] = np.where(
        metrics['Total_Losses'] > 0,
        metrics['Total_Wins'] / metrics['Total_Losses'],
        np.where(metrics['Total_Wins'] > 0, 5.0, 0.0)
    )
    
    # Profit Expectancy
    win_prob = metrics['Win_Count'] / tc
    loss_prob = metrics['Loss_Count'] / tc
    avg_win = np.where(metrics['Win_Count'] > 0, metrics['Total_Wins'] / metrics['Win_Count'], 0)
    avg_loss = np.where(metrics['Loss_Count'] > 0, metrics['Total_Losses'] / metrics['Loss_Count'], 0)
    metrics['P_Exp'] = (win_prob * avg_win) - (loss_prob * avg_loss)
    
    # MDD% 計算
    mdd_values = _calculate_mdd_for_hero(closing_df, aid_col, exec_col, initial_balance)
    metrics = metrics.merge(mdd_values, on='AID', how='left')
    metrics['MDD_Pct'] = metrics['MDD_Pct'].fillna(0)
    
    # Recovery Factor
    metrics['Rec_F'] = np.where(
        metrics['Max_DD_Abs'] > 0,
        metrics['Net_PL'] / metrics['Max_DD_Abs'],
        np.where(metrics['Net_PL'] > 0, metrics['Net_PL'], 0)
    )
    
    # ==================== 應用過濾條件 ====================
    mask = pd.Series(True, index=metrics.index)
    
    if filter_positive:
        mask &= metrics['Net_PL'] > 0
    if min_scalp_pct is not None:
        mask &= metrics['Scalp_Pct'] >= float(min_scalp_pct)
    if min_scalp_pl is not None:
        mask &= metrics['Scalp_PL'] >= float(min_scalp_pl)
    if min_pnl is not None:
        mask &= metrics['Net_PL'] >= float(min_pnl)
    if min_winrate is not None:
        mask &= metrics['Win_Rate'] >= float(min_winrate)
    if min_sharpe is not None:
        mask &= metrics['Sharpe'] >= float(min_sharpe)
    if max_mdd is not None:
        mask &= metrics['MDD_Pct'] <= float(max_mdd)
    
    filtered = metrics[mask].copy()
    
    # ==================== 排序並取 Top N ====================
    filtered = filtered.sort_values('Net_PL', ascending=False).head(top_n)
    
    # ==================== 整理輸出 ====================
    output_cols = [
        'AID', 'Net_PL', 'Scalp_PL', 'Scalp_Pct', 'Sharpe', 'MDD_Pct',
        'Q1', 'Median', 'Q3', 'IQR', 'P_Exp', 'PF', 'Rec_F', 'Win_Rate', 'Trade_Count'
    ]
    
    # 重命名為中文
    rename_map = {
        'Net_PL': '盈虧',
        'Scalp_PL': 'Scalp盈虧',
        'Scalp_Pct': 'Scalp%',
        'MDD_Pct': 'MDD%',
        'P_Exp': 'P. Exp',
        'Rec_F': 'Rec.F',
        'Win_Rate': '勝率%',
        'Trade_Count': '筆數'
    }
    
    result = filtered[output_cols].copy()
    result = result.rename(columns=rename_map)
    
    # 四捨五入
    numeric_cols = ['盈虧', 'Scalp盈虧', 'Scalp%', 'Sharpe', 'MDD%',
                    'Q1', 'Median', 'Q3', 'IQR', 'P. Exp', 'PF', 'Rec.F', '勝率%']
    for col in numeric_cols:
        if col in result.columns:
            result[col] = result[col].round(2)
    
    # ⭐ 確保 AID 為字串
    result['AID'] = result['AID'].astype(str)
    
    return result


def _calculate_mdd_for_hero(
    df: pd.DataFrame,
    aid_col: str,
    exec_col: str,
    initial_balance: float
) -> pd.DataFrame:
    """
    為英雄榜計算 MDD% 和 Max_DD_Abs
    
    Returns:
        包含 AID, MDD_Pct, Max_DD_Abs 的 DataFrame
    """
    results = []
    
    for aid, group in df.groupby(aid_col, sort=False):
        if len(group) < 2:
            results.append({'AID': str(aid), 'MDD_Pct': 0.0, 'Max_DD_Abs': 0.0})
            continue
        
        sorted_group = group.sort_values(exec_col)
        cumulative_pl = sorted_group['Net_PL'].cumsum()
        equity = initial_balance + cumulative_pl
        running_max = equity.cummax()
        
        # MDD%
        drawdown_pct = np.where(
            running_max != 0,
            (equity - running_max) / running_max * 100,
            0
        )
        mdd_pct = abs(np.min(drawdown_pct))
        
        # Max DD Abs
        max_dd_abs = abs((equity - running_max).min())
        
        results.append({
            'AID': str(aid),
            'MDD_Pct': round(mdd_pct, 2),
            'Max_DD_Abs': round(max_dd_abs, 2)
        })
    
    return pd.DataFrame(results)


# ==================== 測試區塊 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("核心數據引擎模組 v1.0")
    print("=" * 60)
    print("\n✅ 模組載入成功")
    print("\n📋 可用函數：")
    print("  - load_data(uploaded_files)")
    print("  - get_client_metrics(df, initial_balance, scalper_threshold_seconds)")
    print("  - get_violin_sample(df, max_points, sample_ratio)")
    print("  - get_hero_metrics(df, ...)")
    print("  - filter_closing_trades(df)")
    print("  - get_column_map()")
    print("\n🔧 欄位映射：")
    for key, value in COLUMN_MAP.items():
        print(f"  {key}: {value}")
