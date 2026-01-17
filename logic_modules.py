import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import data_engine_optimized as de  # 引用數據引擎的常數與輔助函數

# ==================== 常數定義 (1:1 移植) ====================
STYLE_COLORS = {
    '極短線 (Scalp)': '#E74C3C',
    '短線 (Intraday)': '#F39C12',
    '中線 (Day Trade)': '#3498DB',
    '長線 (Swing)': '#27AE60'
}


# ==================== UI 與 過濾器組件 ====================

def render_global_filters(key_prefix, default_pnl=0.0, default_winrate=0.0,
                          default_sharpe=-10.0, default_mdd=100.0):
    """
    渲染全局過濾器 (1:1 還原)
    """
    st.markdown("#### 🔧 全局過濾器")
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        min_pnl = st.number_input(
            "最低盈虧 ($)",
            value=float(default_pnl),
            step=100.0,
            key=f"{key_prefix}_pnl",
            help="僅顯示盈虧 ≥ 此值的客戶"
        )
    with f2:
        min_winrate = st.number_input(
            "最低勝率 (%)",
            value=float(default_winrate),
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            key=f"{key_prefix}_wr",
            help="僅顯示勝率 ≥ 此值的客戶"
        )
    with f3:
        min_sharpe = st.number_input(
            "最低 Sharpe",
            value=float(default_sharpe),
            step=0.5,
            key=f"{key_prefix}_sharpe",
            help="僅顯示 Sharpe ≥ 此值的客戶"
        )
    with f4:
        max_mdd = st.number_input(
            "最高 MDD (%)",
            value=float(default_mdd),
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            key=f"{key_prefix}_mdd",
            help="僅顯示 MDD ≤ 此值的客戶"
        )

    return min_pnl, min_winrate, min_sharpe, max_mdd


def render_scalper_filters(key_prefix, default_scalp_pct=80.0, default_scalp_pl=0.0):
    """
    渲染 Scalper 專用過濾器 (1:1 還原)
    """
    s1, s2 = st.columns(2)
    with s1:
        min_scalp_pct = st.slider(
            "Scalp% 門檻",
            min_value=50,
            max_value=100,
            value=int(default_scalp_pct),
            step=5,
            key=f"{key_prefix}_spct",
            help="Scalp 交易筆數佔比"
        )
    with s2:
        min_scalp_pl = st.number_input(
            "Scalp 盈虧金額門檻 ($)",
            value=float(default_scalp_pl),
            step=100.0,
            key=f"{key_prefix}_spl",
            help="僅顯示 Scalp 盈虧 ≥ 此值的客戶"
        )
    return float(min_scalp_pct), min_scalp_pl


def get_table_column_config():
    """
    獲取統一的表格欄位配置 - 確保 AID 為純文字可複製
    """
    return {
        'AID': st.column_config.TextColumn(
            'AID',
            help='📋 點擊單元格可選取複製',
            width='small'
        ),
        '盈虧': st.column_config.TextColumn('盈虧', width='medium'),
        'Scalp盈虧': st.column_config.TextColumn('Scalp盈虧', width='medium'),
        'Scalp%': st.column_config.TextColumn('Scalp%', width='small'),
        'Sharpe': st.column_config.TextColumn('Sharpe', width='small'),
        'MDD%': st.column_config.TextColumn('MDD%', width='small'),
        'Q1': st.column_config.TextColumn('Q1', width='small'),
        'Median': st.column_config.TextColumn('Median', width='small'),
        'Q3': st.column_config.TextColumn('Q3', width='small'),
        'IQR': st.column_config.TextColumn('IQR', width='small'),
        'P. Exp': st.column_config.TextColumn('P.Exp', width='small'),
        'PF': st.column_config.NumberColumn('PF', format='%.2f', width='small'),
        'Rec.F': st.column_config.NumberColumn('Rec.F', format='%.2f', width='small'),
        '勝率%': st.column_config.NumberColumn('勝率%', format='%.1f%%', width='small'),
        '筆數': st.column_config.NumberColumn('筆數', format='%d', width='small')
    }


def format_hero_table_display(hero_df):
    """格式化英雄榜表格顯示 (加入 Emoji 與字串格式)"""
    if hero_df.empty:
        return hero_df

    display_df = hero_df.copy()

    # Scalp% emoji - 安全檢查
    if 'Scalp%' in display_df.columns:
        display_df['Scalp%'] = display_df['Scalp%'].apply(
            lambda x: f"🔥{x:.1f}%" if x > 80 else f"{x:.1f}%"
        )

    # Sharpe 顏色 - 安全檢查
    if 'Sharpe' in display_df.columns:
        display_df['Sharpe'] = display_df['Sharpe'].apply(
            lambda x: f"⭐{x:.2f}" if x > 2 else f"{x:.2f}"
        )

    # MDD% 紅色警示 - 安全檢查
    if 'MDD%' in display_df.columns:
        display_df['MDD%'] = display_df['MDD%'].apply(
            lambda x: f"🔴{x:.1f}%" if x > 20 else f"{x:.1f}%"
        )

    # P.Exp 顏色 - 安全檢查
    if 'P. Exp' in display_df.columns:
        display_df['P. Exp'] = display_df['P. Exp'].apply(
            lambda x: f"🟢{x:.2f}" if x > 0 else f"🔴{x:.2f}"
        )

    # 金額格式 - 安全檢查
    for col in ['盈虧', 'Scalp盈虧', 'Q1', 'Median', 'Q3', 'IQR']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")

    return display_df


def clean_aid_input(raw_input: str) -> str:
    """清理 AID 輸入字串 (Tab 2 搜尋用)"""
    if not raw_input: return ""
    return raw_input.strip().replace(',', '').replace(' ', '')


# ==================== 圖表繪製函數 (1:1 移植) ====================

@st.cache_data(show_spinner=False, ttl=1800)
def create_cumulative_pnl_chart(_df, initial_balance, scalper_threshold_seconds):
    """創建累計盈虧走勢圖"""
    exec_col = de.COLUMN_MAP['execution_time']
    scalper_minutes = scalper_threshold_seconds / 60

    closing_df = de.filter_closing_trades(_df)
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
    fig.add_trace(go.Scatter(
        x=merged_df['Date'],
        y=merged_df['Cumulative_PL'],
        mode='lines+markers',
        name='整體累計',
        line=dict(color='#2E86AB', width=2.5),
        hovertemplate='<b>日期:</b> %{x|%Y-%m-%d}<br><b>累計:</b> $%{y:,.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=merged_df['Date'],
        y=merged_df['Scalper_Cumulative_PL'],
        mode='lines+markers',
        name=f'Scalper (<{scalper_minutes:.0f}分鐘)',
        line=dict(color='#F39C12', width=2.5, dash='dot'),
        hovertemplate='<b>日期:</b> %{x|%Y-%m-%d}<br><b>Scalper:</b> $%{y:,.2f}<extra></extra>'
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1.5)
    fig.update_layout(
        title='📈 累計淨盈虧走勢',
        xaxis_title='日期',
        yaxis_title='累計淨盈虧 ($)',
        height=450,
        hovermode='x unified',
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=60, r=30, t=80, b=60),
        plot_bgcolor='rgba(248,249,250,1)'
    )

    return fig, {
        'total_pnl': merged_df['Cumulative_PL'].iloc[-1] if len(merged_df) > 0 else 0,
        'scalper_pnl': merged_df['Scalper_Cumulative_PL'].iloc[-1] if len(merged_df) > 0 else 0
    }


@st.cache_data(show_spinner=False, ttl=1800)
def create_violin_plot_with_stats(_df):
    """創建小提琴圖並返回統計數據 (含 Outliers 計算)"""
    aid_col = de.COLUMN_MAP['aid']
    closing_df = de.filter_closing_trades(_df)
    aid_pl = closing_df.groupby(aid_col)['Net_PL'].sum().reset_index()
    aid_pl.columns = ['AID', 'Net_PL']

    # 統計數據
    stats = {
        'count': len(aid_pl),
        'mean': aid_pl['Net_PL'].mean(),
        'median': aid_pl['Net_PL'].median(),
        'std': aid_pl['Net_PL'].std(),
        'q1': aid_pl['Net_PL'].quantile(0.25),
        'q3': aid_pl['Net_PL'].quantile(0.75),
        'min': aid_pl['Net_PL'].min(),
        'max': aid_pl['Net_PL'].max(),
        'profitable': (aid_pl['Net_PL'] > 0).sum(),
        'losing': (aid_pl['Net_PL'] <= 0).sum()
    }
    stats['iqr'] = stats['q3'] - stats['q1']
    stats['lower_fence'] = stats['q1'] - 1.5 * stats['iqr']
    stats['upper_fence'] = stats['q3'] + 1.5 * stats['iqr']
    stats['outliers'] = len(aid_pl[
                                (aid_pl['Net_PL'] < stats['lower_fence']) |
                                (aid_pl['Net_PL'] > stats['upper_fence'])
                                ])

    Q1_pct = aid_pl['Net_PL'].quantile(0.01)
    Q99_pct = aid_pl['Net_PL'].quantile(0.99)

    fig = go.Figure()
    fig.add_trace(go.Violin(
        x=aid_pl['Net_PL'],
        y=['盈虧分布'] * len(aid_pl),
        orientation='h',
        box_visible=True,
        meanline_visible=True,
        line_color='#2C3E50',
        fillcolor='rgba(52, 152, 219, 0.5)',
        points='all',
        pointpos=-0.5,
        jitter=0.3,
        marker=dict(color='#3498DB', size=6, opacity=0.6),
        customdata=aid_pl['AID'].values,
        hovertemplate='<b>AID:</b> %{customdata}<br><b>Net_PL:</b> $%{x:,.2f}<extra></extra>'
    ))

    x_padding = (Q99_pct - Q1_pct) * 0.1
    fig.add_vline(x=0, line_color="black", line_width=3)
    fig.update_layout(
        title='🎻 客戶盈虧分佈 (Violin Plot)',
        height=750,
        xaxis=dict(title='累計淨盈虧 ($)', range=[Q1_pct - x_padding, Q99_pct + x_padding]),
        yaxis=dict(showticklabels=False),
        plot_bgcolor='rgba(248,249,250,1)'
    )

    return fig, stats


@st.cache_data(show_spinner=False, ttl=1800)
def create_trading_style_pie(_df, title="交易風格分佈"):
    """創建交易風格圓餅圖"""
    closing_df = de.filter_closing_trades(_df)
    if 'Hold_Minutes' not in closing_df.columns or closing_df['Hold_Minutes'].isna().all():
        return None

    closing_df = closing_df.copy()
    # 使用 de 中的分類函數
    closing_df['Trading_Style'] = closing_df['Hold_Minutes'].apply(de.classify_trading_style)
    style_counts = closing_df['Trading_Style'].value_counts().reset_index()
    style_counts.columns = ['風格', '筆數']

    fig = px.pie(
        style_counts,
        values='筆數',
        names='風格',
        hole=0.4,
        color='風格',
        color_discrete_map=STYLE_COLORS,
        title=title
    )
    fig.update_traces(textposition='inside', textinfo='label+percent')
    fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
    return fig


@st.cache_data(show_spinner=False, ttl=1800)
def create_profit_factor_chart_colored(_aid_stats_df):
    """創建獲利因子分佈圖"""
    pf_data = _aid_stats_df[['AID', 'Profit_Factor', 'Net_PL', 'Trade_Count']].copy()
    pf_display = pf_data[pf_data['Profit_Factor'] <= 5].copy()

    bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    pf_display['PF_Bin'] = pd.cut(pf_display['Profit_Factor'], bins=bins, right=False)
    bin_stats = pf_display.groupby('PF_Bin', observed=True).size().reset_index(name='Count')
    bin_stats['PF_Bin_Str'] = bin_stats['PF_Bin'].astype(str)
    bin_stats['Color'] = bin_stats['PF_Bin'].apply(
        lambda x: '#E74C3C' if x.right <= 1.0 else '#27AE60'
    )

    fig = go.Figure()
    for _, row in bin_stats.iterrows():
        fig.add_trace(go.Bar(
            x=[row['PF_Bin_Str']],
            y=[row['Count']],
            marker=dict(color=row['Color'], opacity=0.75),
            showlegend=False
        ))

    fig.add_vline(x=1.5, line_dash="dash", line_color="red", line_width=2, annotation_text="PF=1.0")
    fig.update_layout(
        title='📊 獲利因子分布',
        xaxis=dict(title='Profit Factor', tickangle=-45),
        yaxis_title='交易者數',
        height=400,
        plot_bgcolor='rgba(248,249,250,1)'
    )

    # 計算 profitable_ratio
    profitable_ratio = (pf_data['Profit_Factor'] > 1.0).sum() / len(pf_data) * 100 if len(pf_data) > 0 else 0
    return fig, profitable_ratio


@st.cache_data(show_spinner=False, ttl=1800)
def create_risk_return_scatter(_aid_stats_df, initial_balance):
    """創建風險回報矩陣散佈圖"""
    scatter_df = _aid_stats_df.copy()
    min_size, max_size = 10, 50
    if scatter_df['Trade_Volume'].max() > scatter_df['Trade_Volume'].min():
        scatter_df['Size'] = min_size + (
                (scatter_df['Trade_Volume'] - scatter_df['Trade_Volume'].min()) /
                (scatter_df['Trade_Volume'].max() - scatter_df['Trade_Volume'].min()) * (max_size - min_size)
        )
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
            colorbar=dict(title='盈虧')
        ),
        customdata=np.column_stack((
            scatter_df['AID'],
            scatter_df['Win_Rate'],
            scatter_df['Sharpe']
        )),
        hovertemplate=(
            '<b>AID:</b> %{customdata[0]}<br>'
            '<b>淨盈虧:</b> $%{y:,.2f}<br>'
            '<b>MDD:</b> %{x:.1f}%<br>'
            '<b>勝率:</b> %{customdata[1]:.1f}%<br>'
            '<b>Sharpe:</b> %{customdata[2]:.2f}<extra></extra>'
        )
    ))
    fig.update_layout(
        title=f'🎯 風險回報矩陣 (初始資金: ${initial_balance:,})',
        xaxis=dict(title='MDD (%)', range=[0, 100]),
        yaxis_title='總盈虧 ($)',
        height=750,
        plot_bgcolor='rgba(248,249,250,1)'
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=50, line_dash="dash", line_color="gray")

    # 象限標註 (1:1 還原)
    fig.add_annotation(x=10, y=0.95, xref="x", yref="paper", text="🌟 低風險高回報", showarrow=False,
                       font=dict(size=12, color="green"))
    fig.add_annotation(x=90, y=0.95, xref="x", yref="paper", text="⚡ 高風險高回報", showarrow=False,
                       font=dict(size=12, color="orange"))
    fig.add_annotation(x=10, y=0.05, xref="x", yref="paper", text="🐢 低風險低回報", showarrow=False,
                       font=dict(size=12, color="gray"))
    fig.add_annotation(x=90, y=0.05, xref="x", yref="paper", text="⚠️ 高風險虧損", showarrow=False,
                       font=dict(size=12, color="red"))

    return fig


@st.cache_data(show_spinner=False, ttl=1800)
def create_daily_pnl_chart(_df):
    """創建每日盈虧柱狀圖"""
    exec_col = de.COLUMN_MAP['execution_time']
    closing_df = de.filter_closing_trades(_df)
    df_daily = closing_df.copy()
    df_daily['Date'] = df_daily[exec_col].dt.date
    daily_pnl = df_daily.groupby('Date')['Net_PL'].sum().reset_index()
    daily_pnl.columns = ['日期', '每日盈虧']
    colors = ['#27AE60' if x > 0 else '#E74C3C' for x in daily_pnl['每日盈虧']]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily_pnl['日期'], y=daily_pnl['每日盈虧'], marker_color=colors))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        title='📅 每日盈虧',
        xaxis_title='日期',
        yaxis_title='淨盈虧 ($)',
        height=350,
        plot_bgcolor='rgba(248,249,250,1)'
    )
    return fig


@st.cache_data(show_spinner=False, ttl=1800)
def create_client_cumulative_chart(_cumulative_df, scalper_minutes):
    """創建個人累計盈虧圖 (Tab 2 用)"""
    exec_col = de.COLUMN_MAP['execution_time']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=_cumulative_df[exec_col],
        y=_cumulative_df['Cumulative_PL'],
        mode='lines',
        name='累計總盈虧',
        line=dict(color='#2E86AB', width=2),
        hovertemplate='<b>時間:</b> %{x|%Y-%m-%d %H:%M}<br><b>累計:</b> $%{y:,.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=_cumulative_df[exec_col],
        y=_cumulative_df['Scalper_Cumulative_PL'],
        mode='lines',
        name=f'Scalper (<{scalper_minutes}分鐘)',
        line=dict(color='#F39C12', width=2, dash='dot'),
        hovertemplate='<b>時間:</b> %{x|%Y-%m-%d %H:%M}<br><b>Scalper:</b> $%{y:,.2f}<extra></extra>'
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title='📈 個人累計盈虧',
        height=350,
        hovermode='x unified',
        legend=dict(orientation="h", y=1.05, x=0),
        margin=dict(l=60, r=30, t=60, b=50),
        plot_bgcolor='rgba(248,249,250,1)'
    )
    return fig


@st.cache_data(show_spinner=False, ttl=1800)
def create_stacked_product_chart(_product_df, is_profit=True):
    """創建堆疊產品柱狀圖 (Tab 3 用)"""
    if _product_df is None or _product_df.empty:
        return None

    df = _product_df.copy()
    if is_profit:
        non_scalp_color, scalp_color = '#1E8449', '#82E0AA'
        title = '📈 當日盈利產品 Top 5'
    else:
        non_scalp_color, scalp_color = '#922B21', '#F1948A'
        title = '📉 當日虧損產品 Top 5'

    df = df.sort_values('Total_PL', ascending=not is_profit)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df['Product'],
        x=df['NonScalp_PL'],
        name='Non-Scalp',
        orientation='h',
        marker_color=non_scalp_color,
        text=df['NonScalp_PL'].apply(lambda x: f"${x:,.0f}"),
        textposition='inside'
    ))
    fig.add_trace(go.Bar(
        y=df['Product'],
        x=df['Scalp_PL'],
        name='Scalp',
        orientation='h',
        marker_color=scalp_color,
        text=df['Scalp_PL'].apply(lambda x: f"${x:,.0f}"),
        textposition='inside'
    ))
    fig.update_layout(
        title=title,
        barmode='relative',
        xaxis_title='盈虧金額 ($)',
        height=300,
        legend=dict(orientation="h", y=1.1),
        plot_bgcolor='rgba(248,249,250,1)'
    )
    fig.add_vline(x=0, line_color="black", line_width=1)

    return fig


# ==================== 新增：個人產品盈虧分析 (Tab 2) ====================

# 定義統一的顏色映射 (與 Tab 3 一致)
COLOR_MAP = {
    'profit': {
        'NonScalp': '#1E8449',  # 深綠色
        'Scalp': '#82E0AA'      # 淺綠色
    },
    'loss': {
        'NonScalp': '#922B21',  # 深紅色
        'Scalp': '#F1948A'      # 淺紅色
    }
}


@st.cache_data(show_spinner=False, ttl=1800)
def plot_top_products_bar(_product_df, is_profit=True, top_n=5):
    """
    創建個人 Top N 產品水平條形圖 (Tab 2 用)
    
    參數:
        _product_df: 產品盈虧 DataFrame，包含 ['Symbol', 'Scalp_PL', 'NonScalp_PL', 'Total_PL']
        is_profit: True=盈利產品, False=虧損產品
        top_n: 顯示前 N 名
    """
    if _product_df is None or _product_df.empty:
        return None
    
    df = _product_df.copy()
    
    # 選擇顏色方案
    if is_profit:
        colors = COLOR_MAP['profit']
        title = f'📈 Top {top_n} 盈利產品'
        # 盈利產品：從大到小排序
        df = df.nlargest(top_n, 'Total_PL')
    else:
        colors = COLOR_MAP['loss']
        title = f'📉 Top {top_n} 虧損產品'
        # 虧損產品：從小到大排序（最虧的在前）
        df = df.nsmallest(top_n, 'Total_PL')
    
    # 反轉順序，讓最大/最小的顯示在最上方
    df = df.iloc[::-1]
    
    fig = go.Figure()
    
    # 添加 Non-Scalp 條形
    fig.add_trace(go.Bar(
        y=df['Symbol'],
        x=df['NonScalp_PL'],
        name='Non-Scalp',
        orientation='h',
        marker_color=colors['NonScalp'],
        text=df['NonScalp_PL'].apply(lambda x: f"${x:,.0f}" if abs(x) >= 1 else ""),
        textposition='inside',
        hovertemplate='<b>產品:</b> %{y}<br><b>Non-Scalp:</b> $%{x:,.2f}<extra></extra>'
    ))
    
    # 添加 Scalp 條形
    fig.add_trace(go.Bar(
        y=df['Symbol'],
        x=df['Scalp_PL'],
        name='Scalp',
        orientation='h',
        marker_color=colors['Scalp'],
        text=df['Scalp_PL'].apply(lambda x: f"${x:,.0f}" if abs(x) >= 1 else ""),
        textposition='inside',
        hovertemplate='<b>產品:</b> %{y}<br><b>Scalp:</b> $%{x:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        barmode='relative',
        xaxis_title='盈虧金額 ($)',
        yaxis_title='產品',
        height=300,
        hovermode='y unified',
        legend=dict(orientation="h", y=1.15, x=0),
        margin=dict(l=100, r=30, t=80, b=50),
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    fig.add_vline(x=0, line_color="black", line_width=1.5)
    
    return fig