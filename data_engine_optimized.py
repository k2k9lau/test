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
    """
    載入並預處理交易數據 - 優化記憶體使用
    增強錯誤處理和多檔案穩定性
    """
    if not uploaded_files:
        return None
    
    dfs = []
    error_files = []
    
    for uploaded_file in uploaded_files:
        try:
            # 重置檔案指針
            uploaded_file.seek(0)
            
            if uploaded_file.name.endswith('.csv'):
                # 使用更高效的 CSV 讀取參數
                df = pd.read_csv(
                    uploaded_file, 
                    parse_dates=False,
                    low_memory=False
                )
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                st.warning(f"⚠️ 跳過不支援的檔案格式: {uploaded_file.name}")
                continue
            
            # 驗證必要欄位
            required_cols = [COLUMN_MAP['aid'], COLUMN_MAP['execution_time']]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                error_files.append(f"{uploaded_file.name} (缺少欄位: {', '.join(missing_cols)})")
                continue
            
            dfs.append(df)
            
        except Exception as e:
            error_files.append(f"{uploaded_file.name} ({str(e)})")
            continue
    
    # 顯示錯誤訊息
    if error_files:
        st.error(f"❌ 以下檔案載入失敗:\n" + "\n".join([f"- {f}" for f in error_files]))
    
    if not dfs:
        st.error("❌ 無法載入任何有效檔案")
        return None
    
    # 合併數據框
    try:
        df = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        st.error(f"❌ 合併檔案時發生錯誤: {e}")
        return None
    
    # 數據清理
    exec_col = COLUMN_MAP['execution_time']
    aid_col = COLUMN_MAP['aid']
    
    try:
        # 移除 Total 行
        if exec_col in df.columns:
            df = df[df[exec_col] != 'Total'].copy()
        
        # 去重
        original_count = len(df)
        df = df.drop_duplicates()
        if len(df) < original_count:
            st.info(f"ℹ️ 已移除 {original_count - len(df)} 筆重複數據")
        
        # 日期轉換 - 向量化處理
        for col_key in ['execution_time', 'open_time']:
            col_name = COLUMN_MAP[col_key]
            if col_name in df.columns:
                df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                
                # 檢查無效日期
                invalid_dates = df[col_name].isna().sum()
                if invalid_dates > 0:
                    st.warning(f"⚠️ {col_name}: {invalid_dates} 筆日期無效")
        
        # 數值欄位處理 - 一次性填充
        numeric_cols = ['closed_pl', 'commission', 'swap']
        for col_key in numeric_cols:
            col_name = COLUMN_MAP[col_key]
            if col_name in df.columns:
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
        
        # 向量化計算 Net_PL
        df['Net_PL'] = (
            df[COLUMN_MAP['closed_pl']] + 
            df[COLUMN_MAP['commission']] + 
            df[COLUMN_MAP['swap']]
        )
        
        # 向量化計算持倉時間
        exec_time = df[COLUMN_MAP['execution_time']]
        open_time = df[COLUMN_MAP['open_time']]
        
        df['Hold_Seconds'] = np.where(
            pd.notna(exec_time) & pd.notna(open_time),
            (exec_time - open_time).dt.total_seconds(),
            np.nan
        )
        df['Hold_Minutes'] = df['Hold_Seconds'] / 60
        
        # AID 清洗 - 確保不會變成 NaN
        if aid_col in df.columns:
            df[aid_col] = (
                df[aid_col]
                .astype(str)
                .str.replace(r'\.0$', '', regex=True)
                .str.replace(',', '', regex=False)
                .str.strip()
                .replace('nan', '')  # 移除字串 'nan'
            )
            
            # 移除空值 AID
            before_filter = len(df)
            df = df[df[aid_col] != ''].copy()
            if len(df) < before_filter:
                st.warning(f"⚠️ 已移除 {before_filter - len(df)} 筆無效 AID")
            
            # 優化: 轉換為 category 節省記憶體
            df[aid_col] = df[aid_col].astype('category')
        
        # 記憶體優化: 轉換其他分類欄位
        categorical_cols = ['action', 'instrument', 'side', 'business_type']
        for col_key in categorical_cols:
            col_name = COLUMN_MAP[col_key]
            if col_name in df.columns:
                df[col_name] = df[col_name].astype('category')
        
        # 記憶體優化: 降低數值欄位精度 (float64 -> float32)
        float_cols = ['Net_PL', 'Hold_Seconds', 'Hold_Minutes']
        for col_key in numeric_cols:
            if COLUMN_MAP[col_key] in df.columns:
                float_cols.append(COLUMN_MAP[col_key])
        
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].astype('float32')
        
        if COLUMN_MAP['volume'] in df.columns:
            df[COLUMN_MAP['volume']] = pd.to_numeric(
                df[COLUMN_MAP['volume']], 
                errors='coerce'
            ).fillna(0).astype('float32')
        
        st.success(f"✅ 成功載入 {len(dfs)} 個檔案，共 {len(df):,} 筆數據")
        
        return df
        
    except Exception as e:
        st.error(f"❌ 數據處理時發生錯誤: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def filter_closing_trades(df):
    """過濾平倉交易"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    action_col = COLUMN_MAP['action']
    if action_col in df.columns:
        return df[df[action_col] == 'CLOSING'].copy()
    return df.copy()


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
    
    try:
        # 按時間排序並計算累積盈虧
        sorted_df = group_df.sort_values(exec_col)
        cumulative_pl = sorted_df['Net_PL'].cumsum()
        equity = initial_balance + cumulative_pl
        
        # 使用 cummax 計算運行最大值 (極高效)
        running_max = equity.cummax()
        
        # 向量化計算回撤百分比
        drawdown = np.where(running_max != 0, (equity - running_max) / running_max * 100, 0)
        mdd_pct = abs(np.min(drawdown))
        
        return float(mdd_pct)
    except:
        return 0.0


def calculate_sharpe_vectorized(pnl_series, min_trades=3):
    """向量化計算 Sharpe Ratio"""
    if len(pnl_series) < min_trades:
        return 0.0
    
    try:
        mean_pl = pnl_series.mean()
        std_pl = pnl_series.std()
        
        if pd.isna(std_pl) or std_pl == 0:
            return 0.0
        
        return float(mean_pl / std_pl)
    except:
        return 0.0


# ==================== 核心計算邏輯 (全向量化版本) ====================
@st.cache_data(show_spinner=False, ttl=1800)
def calculate_all_aid_stats_realtime(df, initial_balance, scalper_threshold_seconds):
    """
    計算所有 AID 的即時統計 - 完全向量化版本
    使用 groupby + agg 替代循環,大幅提升效能
    **修復: 所有 groupby 後都加上 reset_index()**
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    aid_col = COLUMN_MAP['aid']
    volume_col = COLUMN_MAP['volume']
    exec_col = COLUMN_MAP['execution_time']
    instrument_col = COLUMN_MAP['instrument']
    closed_pl_col = COLUMN_MAP['closed_pl']
    
    closing_df = filter_closing_trades(df)
    
    if closing_df.empty:
        return pd.DataFrame()
    
    try:
        # 預先計算 Scalper mask (向量化)
        closing_df = closing_df.copy()
        closing_df['is_scalper'] = closing_df['Hold_Seconds'] < scalper_threshold_seconds
        closing_df['is_win'] = closing_df['Net_PL'] > 0
        closing_df['is_gain'] = closing_df[closed_pl_col] > 0
        closing_df['abs_loss'] = np.where(
            closing_df[closed_pl_col] < 0, 
            abs(closing_df[closed_pl_col]), 
            0
        )
        
        # 基礎統計 - 一次性 groupby 聚合
        basic_stats = closing_df.groupby(aid_col, observed=True).agg({
            'Net_PL': ['sum', 'mean', 'std', 'count'],
            volume_col: 'sum' if volume_col in closing_df.columns else 'count',
            'Hold_Seconds': 'mean',
            'is_win': 'sum',
            'is_scalper': [
                'sum', 
                lambda x: (x * closing_df.loc[x.index, 'Net_PL']).sum()
            ],
            closed_pl_col: [
                lambda x: x[x > 0].sum(),  # gains
                lambda x: abs(x[x < 0].sum())  # losses
            ]
        }).reset_index()  # ✅ 修復: 加上 reset_index()
        
        # 展平多級列名
        basic_stats.columns = [
            'AID', 'Net_PL', 'Mean_PL', 'Std_PL', 'Trade_Count', 
            'Trade_Volume', 'Avg_Hold_Seconds', 'Wins',
            'Scalper_Count', 'Scalper_PL', 'Gains', 'Losses'
        ]
        
        # 計算百分位數 (Q1, Median, Q3) - 向量化
        quantiles = (
            closing_df.groupby(aid_col, observed=True)['Net_PL']
            .quantile([0.25, 0.5, 0.75])
            .unstack()
            .reset_index()  # ✅ 修復: 加上 reset_index()
        )
        quantiles.columns = ['AID', 'Q1', 'Median', 'Q3']
        
        # 合併基礎統計與百分位數
        stats = basic_stats.merge(quantiles, on='AID', how='left')
        
        # 向量化計算衍生指標
        stats['Win_Rate'] = np.where(
            stats['Trade_Count'] > 0, 
            (stats['Wins'] / stats['Trade_Count'] * 100), 
            0
        )
        
        stats['Scalper_Ratio'] = np.where(
            stats['Trade_Count'] > 0,
            (stats['Scalper_Count'] / stats['Trade_Count'] * 100), 
            0
        )
        
        stats['Profit_Factor'] = np.where(
            stats['Losses'] > 0,
            stats['Gains'] / stats['Losses'],
            np.where(stats['Gains'] > 0, 5.0, 0.0)
        )
        
        stats['IQR'] = stats['Q3'] - stats['Q1']
        
        # Sharpe Ratio (批量計算)
        sharpe_values = []
        for aid in stats['AID']:
            aid_data = closing_df[closing_df[aid_col] == aid]['Net_PL']
            sharpe = calculate_sharpe_vectorized(aid_data)
            sharpe_values.append(sharpe)
        stats['Sharpe'] = sharpe_values
        
        # MDD% (批量計算)
        mdd_values = []
        for aid in stats['AID']:
            aid_group = closing_df[closing_df[aid_col] == aid]
            mdd = calculate_mdd_vectorized(aid_group, initial_balance, exec_col)
            mdd_values.append(mdd)
        stats['MDD_Pct'] = mdd_values
        
        # 確保所有數值欄位為 float
        numeric_columns = [
            'Net_PL', 'Mean_PL', 'Std_PL', 'Trade_Volume', 'Avg_Hold_Seconds',
            'Scalper_PL', 'Gains', 'Losses', 'Win_Rate', 'Scalper_Ratio',
            'Profit_Factor', 'Q1', 'Median', 'Q3', 'IQR', 'Sharpe', 'MDD_Pct'
        ]
        
        for col in numeric_columns:
            if col in stats.columns:
                stats[col] = pd.to_numeric(stats[col], errors='coerce').fillna(0)
        
        # 確保整數欄位
        stats['Trade_Count'] = stats['Trade_Count'].astype(int)
        stats['Wins'] = stats['Wins'].astype(int)
        stats['Scalper_Count'] = stats['Scalper_Count'].astype(int)
        
        return stats
        
    except Exception as e:
        st.error(f"❌ 計算統計時發生錯誤: {e}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=1800)
def calculate_hero_metrics(
    df, 
    initial_balance, 
    scalper_threshold_seconds,
    filter_positive=True,
    min_pnl=0.0,
    min_winrate=0.0,
    min_sharpe=-10.0,
    max_mdd=100.0,
    min_scalp_pct=0.0,
    min_scalp_pl=0.0
):
    """
    計算英雄榜指標
    **修復: 所有 groupby 後都加上 reset_index()**
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    aid_col = COLUMN_MAP['aid']
    exec_col = COLUMN_MAP['execution_time']
    closed_pl_col = COLUMN_MAP['closed_pl']
    
    closing_df = filter_closing_trades(df)
    
    if closing_df.empty:
        return pd.DataFrame()
    
    try:
        # 預計算 mask
        closing_df = closing_df.copy()
        closing_df['is_scalper'] = closing_df['Hold_Seconds'] < scalper_threshold_seconds
        closing_df['is_win'] = closing_df['Net_PL'] > 0
        
        # 聚合統計 - ✅ 加上 reset_index()
        hero_stats = closing_df.groupby(aid_col, observed=True).agg({
            'Net_PL': ['sum', 'count'],
            'is_win': 'sum',
            'is_scalper': [
                'sum',
                lambda x: (x * closing_df.loc[x.index, 'Net_PL']).sum()
            ],
            closed_pl_col: [
                lambda x: x[x > 0].sum(),
                lambda x: abs(x[x < 0].sum())
            ]
        }).reset_index()  # ✅ 修復
        
        hero_stats.columns = [
            'AID', 'Net_PL', 'Trade_Count', 'Wins', 
            'Scalper_Count', 'Scalper_PL', 'Gains', 'Losses'
        ]
        
        # 向量化計算
        hero_stats['Win_Rate'] = np.where(
            hero_stats['Trade_Count'] > 0,
            (hero_stats['Wins'] / hero_stats['Trade_Count'] * 100),
            0
        )
        
        hero_stats['Scalper_Ratio'] = np.where(
            hero_stats['Trade_Count'] > 0,
            (hero_stats['Scalper_Count'] / hero_stats['Trade_Count'] * 100),
            0
        )
        
        hero_stats['Profit_Factor'] = np.where(
            hero_stats['Losses'] > 0,
            hero_stats['Gains'] / hero_stats['Losses'],
            np.where(hero_stats['Gains'] > 0, 5.0, 0.0)
        )
        
        # Sharpe 和 MDD
        sharpe_list = []
        mdd_list = []
        
        for aid in hero_stats['AID']:
            aid_data = closing_df[closing_df[aid_col] == aid]
            sharpe = calculate_sharpe_vectorized(aid_data['Net_PL'])
            mdd = calculate_mdd_vectorized(aid_data, initial_balance, exec_col)
            sharpe_list.append(sharpe)
            mdd_list.append(mdd)
        
        hero_stats['Sharpe'] = sharpe_list
        hero_stats['MDD_Pct'] = mdd_list
        
        # 篩選
        mask = (hero_stats['Net_PL'] > 0) if filter_positive else (hero_stats['Net_PL'] != 0)
        
        mask &= (hero_stats['Net_PL'] >= min_pnl)
        mask &= (hero_stats['Win_Rate'] >= min_winrate)
        mask &= (hero_stats['Sharpe'] >= min_sharpe)
        mask &= (hero_stats['MDD_Pct'] <= max_mdd)
        mask &= (hero_stats['Scalper_Ratio'] >= min_scalp_pct)
        mask &= (hero_stats['Scalper_PL'] >= min_scalp_pl)
        
        result = hero_stats[mask].copy()
        result = result.sort_values('Net_PL', ascending=False).head(20)
        
        # 數值格式化
        for col in ['Net_PL', 'Scalper_PL', 'Gains', 'Losses', 'Sharpe']:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0)
        
        for col in ['Win_Rate', 'Scalper_Ratio', 'Profit_Factor', 'MDD_Pct']:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0)
        
        # ✅ 修復：重命名欄位以符合顯示函數的期望
        # 將內部使用的欄位名稱轉換為顯示用的欄位名稱
        result = result.rename(columns={
            'Scalper_Ratio': 'Scalp%',
            'Scalper_PL': 'Scalp盈虧',
            'Net_PL': '盈虧',
            'Win_Rate': '勝率%',
            'MDD_Pct': 'MDD%',
            'Profit_Factor': 'PF'
        })
        
        # 計算額外的顯示欄位
        # P.Exp (Profit Expectancy) = (Win_Rate * Avg_Win) - (Loss_Rate * Avg_Loss)
        # 簡化版本：使用 Mean_PL
        if '盈虧' in result.columns and 'Trade_Count' in result.columns:
            result['P. Exp'] = result['盈虧'] / result['Trade_Count']
        
        # Rec.F (Recovery Factor) = Net_PL / Max_Drawdown
        if '盈虧' in result.columns and 'MDD%' in result.columns:
            result['Rec.F'] = np.where(
                result['MDD%'] > 0,
                abs(result['盈虧'] / result['MDD%']),
                0
            )
        
        # 計算 Q1, Median, Q3, IQR (需要從原始數據重新計算)
        quantile_data = []
        for aid in result['AID']:
            aid_data = closing_df[closing_df[aid_col] == aid]['Net_PL']
            quantile_data.append({
                'AID': aid,
                'Q1': float(aid_data.quantile(0.25)),
                'Median': float(aid_data.median()),
                'Q3': float(aid_data.quantile(0.75))
            })
        
        if quantile_data:
            quantile_df = pd.DataFrame(quantile_data)
            quantile_df['IQR'] = quantile_df['Q3'] - quantile_df['Q1']
            result = result.merge(quantile_df, on='AID', how='left')
        
        # 重新排序欄位
        column_order = [
            'AID', '盈虧', 'Scalp盈虧', 'Scalp%', 'Sharpe', 'MDD%',
            'Q1', 'Median', 'Q3', 'IQR', 'P. Exp', 'PF', 'Rec.F', '勝率%', 'Trade_Count'
        ]
        
        # 只保留存在的欄位
        existing_cols = [col for col in column_order if col in result.columns]
        result = result[existing_cols]
        
        # 重命名 Trade_Count 為「筆數」
        if 'Trade_Count' in result.columns:
            result = result.rename(columns={'Trade_Count': '筆數'})
        
        return result.reset_index(drop=True)
        
    except Exception as e:
        st.error(f"❌ 計算英雄榜時發生錯誤: {e}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=1800)
def calculate_product_scalp_breakdown(day_df, scalper_threshold_seconds):
    """
    計算產品 Scalper 分解
    **修復: 所有 groupby 後都加上 reset_index()**
    """
    if day_df is None or day_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    instrument_col = COLUMN_MAP['instrument']
    
    if instrument_col not in day_df.columns:
        return pd.DataFrame(), pd.DataFrame()
    
    try:
        day_df = day_df.copy()
        day_df['is_scalper'] = day_df['Hold_Seconds'] < scalper_threshold_seconds
        
        # 分組聚合 - ✅ 加上 reset_index()
        product_stats = day_df.groupby(instrument_col, observed=True).agg({
            'Net_PL': 'sum',
            'is_scalper': [
                'sum',
                lambda x: (x * day_df.loc[x.index, 'Net_PL']).sum()
            ]
        }).reset_index()  # ✅ 修復
        
        product_stats.columns = ['Product', 'Total_PL', 'Scalper_Count', 'Scalp_PL']
        product_stats['NonScalp_PL'] = product_stats['Total_PL'] - product_stats['Scalp_PL']
        
        # 分離盈利和虧損
        profit_products = product_stats[product_stats['Total_PL'] > 0].copy()
        loss_products = product_stats[product_stats['Total_PL'] < 0].copy()
        
        # 排序
        profit_products = profit_products.sort_values('Total_PL', ascending=False)
        loss_products = loss_products.sort_values('Total_PL')
        
        return profit_products.reset_index(drop=True), loss_products.reset_index(drop=True)
        
    except Exception as e:
        st.error(f"❌ 計算產品分解時發生錯誤: {e}")
        return pd.DataFrame(), pd.DataFrame()


def calculate_deep_behavioral_stats(client_df, scalper_threshold_seconds):
    """
    深度行為統計分析
    **修復: 所有 groupby 後都加上 reset_index()**
    """
    if client_df is None or client_df.empty:
        return {}
    
    side_col = COLUMN_MAP['side']
    total_trades = len(client_df)
    total_pl = float(client_df['Net_PL'].sum())
    total_minutes = float(client_df['Hold_Minutes'].sum())
    
    try:
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
        client_sorted['streak_group'] = (
            client_sorted['Net_PL'] > 0
        ).ne(
            (client_sorted['Net_PL'] > 0).shift()
        ).cumsum()
        
        # ✅ 修復: 加上 reset_index()
        streak_sums = (
            client_sorted.groupby('streak_group', observed=False)['Net_PL']
            .sum()
            .reset_index()
        )
        
        max_streak_profit = float(streak_sums['Net_PL'].max()) if not streak_sums.empty else 0
        max_streak_loss = float(streak_sums['Net_PL'].min()) if not streak_sums.empty else 0
        
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
        
    except Exception as e:
        st.error(f"❌ 計算行為統計時發生錯誤: {e}")
        return {
            'max_win_streak': 0, 'max_loss_streak': 0,
            'max_streak_profit': 0, 'max_streak_loss': 0,
            'buy_count': 0, 'sell_count': 0,
            'buy_ratio': 0, 'sell_ratio': 0,
            'buy_pl': 0, 'sell_pl': 0,
            'buy_winrate': 0, 'sell_winrate': 0,
            'scalp_count': 0, 'scalp_ratio': 0,
            'scalp_pl': 0, 'scalp_contribution': 0,
            'scalp_winrate': 0, 'avg_hold_formatted': '00:00:00',
            'avg_hold_days': 0, 'profit_per_minute': 0,
            'q1': 0, 'median': 0, 'q3': 0, 'iqr': 0
        }


@st.cache_data(show_spinner=False, ttl=1800)
def get_client_details(_df, aid, initial_balance, scalper_threshold_seconds):
    """
    獲取客戶詳細資料 (Tab 2 用)
    **修復: 所有 groupby 後都加上 reset_index()**
    """
    if _df is None or _df.empty:
        return None
    
    aid_col = COLUMN_MAP['aid']
    exec_col = COLUMN_MAP['execution_time']
    instrument_col = COLUMN_MAP['instrument']
    closed_pl_col = COLUMN_MAP['closed_pl']
    
    try:
        closing_df = filter_closing_trades(_df)
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
        
        # Symbol 分佈 - ✅ 加上 reset_index()
        if instrument_col in client_df.columns:
            symbol_dist = (
                client_df.groupby(instrument_col, observed=True)
                .size()
                .reset_index(name='Count')  # ✅ 修復
            )
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
        
    except Exception as e:
        st.error(f"❌ 獲取客戶詳情時發生錯誤 (AID: {aid}): {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def get_client_ranking(aid_stats_df, aid, metric='Net_PL'):
    """獲取客戶排名"""
    if aid_stats_df is None or aid_stats_df.empty:
        return None, 0
    
    try:
        sorted_df = aid_stats_df.sort_values(metric, ascending=False).reset_index(drop=True)
        rank_df = sorted_df[sorted_df['AID'] == str(aid)]
        
        if rank_df.empty:
            return None, len(sorted_df)
        
        rank = rank_df.index[0] + 1
        return int(rank), len(sorted_df)
    except Exception as e:
        st.error(f"❌ 計算排名時發生錯誤: {e}")
        return None, 0


@st.cache_data(show_spinner=False, ttl=1800)
def export_to_excel(_df, _aid_stats_df, initial_balance, scalper_threshold_seconds):
    """匯出至 Excel - 優化版"""
    if _df is None or _df.empty or _aid_stats_df is None or _aid_stats_df.empty:
        st.warning("⚠️ 無數據可匯出")
        return BytesIO()
    
    try:
        from openpyxl.styles import Font, PatternFill, Alignment
        
        output = BytesIO()
        closing_df = filter_closing_trades(_df)
        aid_col = COLUMN_MAP['aid']
        
        # Summary
        summary_df = pd.DataFrame([
            ['總交易筆數', len(_df)],
            ['平倉交易筆數', len(closing_df)],
            ['總客戶數', _df[aid_col].nunique()],
            ['總淨盈虧', round(closing_df['Net_PL'].sum(), 2)],
            ['初始資金', initial_balance]
        ], columns=['指標', '數值'])
        
        # Risk Return (優化: 只選取需要的欄位)
        risk_cols = [
            'AID', 'Net_PL', 'MDD_Pct', 'Sharpe', 'Trade_Count',
            'Win_Rate', 'Profit_Factor', 'Scalper_Ratio', 'Q1', 'Median', 'Q3'
        ]
        
        available_cols = [col for col in risk_cols if col in _aid_stats_df.columns]
        risk_return_df = _aid_stats_df[available_cols].sort_values('Net_PL', ascending=False)
        
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
        
    except Exception as e:
        st.error(f"❌ Excel 匯出時發生錯誤: {e}")
        import traceback
        st.code(traceback.format_exc())
        return BytesIO()


# ==================== 新增：個人產品盈虧分析 (Tab 2) ====================

@st.cache_data(show_spinner=False, ttl=1800)
def calculate_client_product_breakdown(_client_df, scalper_threshold_seconds):
    """
    計算單一客戶的產品級別盈虧分解 (含 Scalp 分類)
    
    參數:
        _client_df: 單一 AID 的交易數據 DataFrame
        scalper_threshold_seconds: Scalper 秒數門檻
    
    返回:
        DataFrame with columns: ['Symbol', 'Scalp_PL', 'NonScalp_PL', 'Total_PL']
    """
    if _client_df is None or _client_df.empty:
        return pd.DataFrame()
    
    try:
        instrument_col = COLUMN_MAP['instrument']
        
        # 確保有 Hold_Seconds 欄位
        if 'Hold_Seconds' not in _client_df.columns:
            st.warning("⚠️ 缺少 Hold_Seconds 欄位，無法分類 Scalp")
            return pd.DataFrame()
        
        df = _client_df.copy()
        
        # 向量化分類 Scalp
        df['Is_Scalp'] = df['Hold_Seconds'] < scalper_threshold_seconds
        
        # 按產品和 Scalp 分組聚合
        product_agg = (
            df.groupby([instrument_col, 'Is_Scalp'], observed=True)['Net_PL']
            .sum()
            .reset_index()  # ✅ 防止 KeyError
        )
        
        # Pivot 表格：行=產品，列=Scalp/NonScalp
        product_pivot = product_agg.pivot_table(
            index=instrument_col,
            columns='Is_Scalp',
            values='Net_PL',
            fill_value=0
        ).reset_index()  # ✅ 防止 KeyError
        
        # 重命名欄位
        product_pivot.columns.name = None
        if True in product_pivot.columns and False in product_pivot.columns:
            product_pivot = product_pivot.rename(columns={
                True: 'Scalp_PL',
                False: 'NonScalp_PL'
            })
        elif True in product_pivot.columns:
            product_pivot = product_pivot.rename(columns={True: 'Scalp_PL'})
            product_pivot['NonScalp_PL'] = 0
        elif False in product_pivot.columns:
            product_pivot = product_pivot.rename(columns={False: 'NonScalp_PL'})
            product_pivot['Scalp_PL'] = 0
        else:
            product_pivot['Scalp_PL'] = 0
            product_pivot['NonScalp_PL'] = 0
        
        # 重命名產品欄位
        product_pivot = product_pivot.rename(columns={instrument_col: 'Symbol'})
        
        # 計算總盈虧
        product_pivot['Total_PL'] = product_pivot['Scalp_PL'] + product_pivot['NonScalp_PL']
        
        # 排序並返回
        product_pivot = product_pivot.sort_values('Total_PL', ascending=False)
        
        return product_pivot[['Symbol', 'Scalp_PL', 'NonScalp_PL', 'Total_PL']]
        
    except Exception as e:
        st.error(f"❌ 計算產品盈虧時發生錯誤: {e}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame()
