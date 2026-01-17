import streamlit as st
import pandas as pd
from datetime import datetime
import data_engine_optimized as de  # 使用優化版數據引擎
import logic_modules as lm  # 導入邏輯模組

# ==================== 1. 頁面基礎配置 ====================
st.set_page_config(
    page_title="交易數據分析系統 v2.6 穩定版",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== 2. Session State 初始化 ====================
def init_session_state():
    """初始化 session state，確保多檔案載入穩定性"""
    if 'df_loaded' not in st.session_state:
        st.session_state.df_loaded = None
    if 'aid_stats_loaded' not in st.session_state:
        st.session_state.aid_stats_loaded = None
    if 'last_upload_count' not in st.session_state:
        st.session_state.last_upload_count = 0
    if 'processing' not in st.session_state:
        st.session_state.processing = False


# ==================== 3. 主程式邏輯 ====================
def main():
    # 初始化 session state
    init_session_state()
    
    st.title("📊 交易數據分析系統 v2.6 穩定優化版")
    st.markdown("**⚡ 向量化運算 | 記憶體優化 | 快取機制 | 多檔案穩定載入**")

    # --- 側邊欄配置 ---
    with st.sidebar:
        st.header("⚙️ 全域參數")
        initial_balance = st.number_input(
            "💰 初始資金",
            value=10000.0,
            min_value=0.0,
            step=1000.0
        )
        scalper_minutes = st.number_input(
            "⏱️ Scalper (分鐘)",
            value=5.0,
            min_value=1.0,
            max_value=60.0,
            step=1.0
        )
        # 計算門檻秒數
        scalper_threshold_seconds = scalper_minutes * 60

        st.markdown("---")
        st.header("📁 數據上傳")
        uploaded_files = st.file_uploader(
            "上傳交易數據",
            type=['xlsx', 'csv'],
            accept_multiple_files=True,
            key='file_uploader'
        )

        # 檢測檔案變化
        current_upload_count = len(uploaded_files) if uploaded_files else 0
        
        if current_upload_count != st.session_state.last_upload_count:
            # 檔案數量變化，清空快取
            st.session_state.df_loaded = None
            st.session_state.aid_stats_loaded = None
            st.session_state.last_upload_count = current_upload_count
            st.session_state.processing = False
        
        if uploaded_files:
            st.success(f"✅ 已選擇 {len(uploaded_files)} 個檔案")
            
            # 顯示檔案列表
            with st.expander("📋 檔案列表"):
                for i, f in enumerate(uploaded_files, 1):
                    st.text(f"{i}. {f.name}")

    # --- 數據載入檢查 ---
    if not uploaded_files:
        st.info("👈 請在左側上傳交易數據檔案")
        st.markdown("""
        ### ⚡ v2.6 優化亮點
        
        **🔧 穩定性改進:**
        - ✅ 多檔案上傳防崩潰機制
        - ✅ Session State 狀態管理
        - ✅ 完整錯誤處理與追蹤
        - ✅ 修復 KeyError: 'AID' 問題
        
        **🚀 效能優化:**
        1. **向量化運算**: NumPy/Pandas groupby 替代循環,速度提升 10-50 倍
        2. **記憶體優化**: Category 類型 + float32,記憶體減少 40-60%
        3. **快取機制**: st.cache_data 智能緩存,避免重複計算
        4. **MDD 優化**: 使用 cummax() 高效矩陣運算
        5. **批量聚合**: 一次性 groupby + agg,大幅減少計算次數
        6. **Reset Index**: 所有 groupby 後加上 reset_index() 防止欄位遺失
        
        ✨ **適合處理 5 萬+ 筆交易數據,穩定性大幅提升!**
        """)
        return

    # --- 數據載入邏輯 (使用 session state) ---
    if st.session_state.df_loaded is None and not st.session_state.processing:
        st.session_state.processing = True
        
        # 顯示載入進度
        with st.spinner("⚡ 載入數據中 (已優化記憶體使用)..."):
            try:
                df = de.load_and_preprocess(uploaded_files)
                
                if df is not None and not df.empty:
                    st.session_state.df_loaded = df
                    st.success(f"✅ 數據載入成功: {len(df):,} 筆交易")
                else:
                    st.error("❌ 無法載入數據或數據為空")
                    st.session_state.processing = False
                    return
                    
            except Exception as e:
                st.error(f"❌ 載入數據時發生錯誤: {e}")
                import traceback
                with st.expander("🔍 錯誤詳情"):
                    st.code(traceback.format_exc())
                st.session_state.processing = False
                return
        
        st.session_state.processing = False
    
    # 使用已載入的數據
    df = st.session_state.df_loaded
    
    if df is None or df.empty:
        st.error("❌ 無有效數據")
        return

    display_df = df.copy()

    # --- 預先計算所有客戶統計 (使用 session state) ---
    if st.session_state.aid_stats_loaded is None:
        with st.spinner("⚡ 計算統計中 (向量化加速)..."):
            try:
                aid_stats_df = de.calculate_all_aid_stats_realtime(
                    display_df,
                    initial_balance,
                    scalper_threshold_seconds
                )
                
                if aid_stats_df is not None and not aid_stats_df.empty:
                    st.session_state.aid_stats_loaded = aid_stats_df
                    st.success(f"✅ 統計計算完成: {len(aid_stats_df)} 個帳戶")
                else:
                    st.warning("⚠️ 無法計算統計或無數據")
                    
            except Exception as e:
                st.error(f"❌ 計算統計時發生錯誤: {e}")
                import traceback
                with st.expander("🔍 錯誤詳情"):
                    st.code(traceback.format_exc())
                return
    
    aid_stats_df = st.session_state.aid_stats_loaded
    
    if aid_stats_df is None or aid_stats_df.empty:
        st.warning("⚠️ 無統計數據")
        return

    st.markdown("---")

    # --- 頂部指標區塊 ---
    try:
        closing_df = de.filter_closing_trades(display_df)
        aid_col = de.COLUMN_MAP['aid']

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("總交易筆數", f"{len(display_df):,}")
        col2.metric("平倉交易", f"{len(closing_df):,}")
        col3.metric("客戶數", f"{display_df[aid_col].nunique():,}")
        col4.metric("總淨盈虧", f"${closing_df['Net_PL'].sum():,.2f}")
    except Exception as e:
        st.error(f"❌ 顯示指標時發生錯誤: {e}")

    # --- 側邊欄 Excel 下載 ---
    with st.sidebar:
        st.markdown("---")
        try:
            excel_data = de.export_to_excel(
                display_df,
                aid_stats_df,
                initial_balance,
                scalper_threshold_seconds
            )
            
            if excel_data.getbuffer().nbytes > 0:
                st.download_button(
                    "📊 下載 Excel 報告",
                    data=excel_data,
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
        except Exception as e:
            st.error(f"❌ Excel 生成失敗: {e}")

    # --- 分頁佈局 ---
    tab1, tab2, tab3 = st.tabs(["📊 整體數據概覽", "👤 個人報告卡", "📅 當日數據概覽"])

    # ==================== Tab 1: 整體數據 ====================
    with tab1:
        st.header("📊 整體數據概覽")

        try:
            # 1. 累計走勢圖
            cumulative_fig, pnl_stats = lm.create_cumulative_pnl_chart(
                display_df,
                initial_balance,
                scalper_threshold_seconds
            )
            st.plotly_chart(cumulative_fig, use_container_width=True)

            m1, m2 = st.columns(2)
            m1.metric("整體淨盈虧", f"${pnl_stats['total_pnl']:,.2f}")
            m2.metric("Scalper 淨盈虧", f"${pnl_stats['scalper_pnl']:,.2f}")

            st.markdown("---")

            # 2. 獲利因子 & 交易風格 (並排)
            st.markdown("### 📊 獲利因子 & 交易風格")
            pf_col, style_col = st.columns(2)
            with pf_col:
                pf_fig, profitable_ratio = lm.create_profit_factor_chart_colored(aid_stats_df)
                st.plotly_chart(pf_fig, use_container_width=True)
                st.success(f"PF > 1.0 佔比: {profitable_ratio:.1f}%")
            with style_col:
                style_pie = lm.create_trading_style_pie(display_df, "🎨 全公司交易風格")
                if style_pie:
                    st.plotly_chart(style_pie, use_container_width=True)

            st.markdown("---")

            # 3. 小提琴圖 + 統計摘要
            st.markdown("### 🎻 客戶盈虧分佈")
            violin_fig, violin_stats = lm.create_violin_plot_with_stats(display_df)

            stat_col, chart_col = st.columns([1, 3])
            with stat_col:
                st.info(f"""
**📊 統計摘要**
━━━━━━━━━━━━
**客戶總數:** {violin_stats['count']:,}
**盈利客戶:** {violin_stats['profitable']:,} ({violin_stats['profitable'] / violin_stats['count'] * 100:.1f}%)
**虧損客戶:** {violin_stats['losing']:,}
━━━━━━━━━━━━
**平均值:** ${violin_stats['mean']:,.2f}
**中位數:** ${violin_stats['median']:,.2f}
**標準差:** ${violin_stats['std']:,.2f}
━━━━━━━━━━━━
**Q1 (25%):** ${violin_stats['q1']:,.2f}
**Q3 (75%):** ${violin_stats['q3']:,.2f}
**IQR:** ${violin_stats['iqr']:,.2f}
━━━━━━━━━━━━
**異常點:** {violin_stats['outliers']} 位
                """)
                st.markdown("""
**📖 圖例說明**
- 🔵 藍點 = 各 AID
- ⬛ 黑線 = 中位數
- 🔴 紅線 = 平均值
- 📦 白框 = IQR 區間
                """)
            with chart_col:
                st.plotly_chart(violin_fig, use_container_width=True)

            st.markdown("---")

            # 4. 風險回報矩陣
            st.markdown("### 🎯 風險回報矩陣")
            st.plotly_chart(
                lm.create_risk_return_scatter(aid_stats_df, initial_balance),
                use_container_width=True
            )

            st.markdown("---")

            # 5. 每日盈虧
            st.plotly_chart(lm.create_daily_pnl_chart(display_df), use_container_width=True)

            st.markdown("---")

            # 6. 歷史盈利客戶表格
            st.markdown("### 🏆 客戶表格")
            st.caption("💡 **點擊表格中的 AID 可選取複製,貼到 Tab 2 搜尋框即可查看詳情**")

            min_pnl_h1, min_wr_h1, min_sharpe_h1, max_mdd_h1 = lm.render_global_filters(
                "hist_hero", 0.0, 0.0, -10.0, 100.0
            )

            history_hero = de.calculate_hero_metrics(
                display_df,
                initial_balance,
                scalper_threshold_seconds,
                filter_positive=True,
                min_pnl=min_pnl_h1,
                min_winrate=min_wr_h1,
                min_sharpe=min_sharpe_h1,
                max_mdd=max_mdd_h1
            )

            if not history_hero.empty:
                st.dataframe(
                    lm.format_hero_table_display(history_hero),
                    use_container_width=True,
                    hide_index=True,
                    column_config=lm.get_table_column_config()
                )
            else:
                st.info("無符合條件的客戶")

            st.markdown("---")

            # 7. 歷史 Scalper 客戶表格
            st.markdown("### 🔥 剝頭皮客戶表格")
            min_scalp_pct_h, min_scalp_pl_h = lm.render_scalper_filters("hist_scalp", 80.0, 0.0)
            min_pnl_hs, min_wr_hs, min_sharpe_hs, max_mdd_hs = lm.render_global_filters(
                "hist_scalp_g", 0.0, 0.0, -10.0, 100.0
            )

            history_scalp = de.calculate_hero_metrics(
                display_df,
                initial_balance,
                scalper_threshold_seconds,
                filter_positive=True,
                min_scalp_pct=min_scalp_pct_h,
                min_scalp_pl=min_scalp_pl_h,
                min_pnl=min_pnl_hs,
                min_winrate=min_wr_hs,
                min_sharpe=min_sharpe_hs,
                max_mdd=max_mdd_hs
            )

            if not history_scalp.empty:
                st.dataframe(
                    lm.format_hero_table_display(history_scalp),
                    use_container_width=True,
                    hide_index=True,
                    column_config=lm.get_table_column_config()
                )
            else:
                st.info("無符合條件的剝頭皮客戶")
                
        except Exception as e:
            st.error(f"❌ Tab 1 顯示時發生錯誤: {e}")
            import traceback
            with st.expander("🔍 錯誤詳情"):
                st.code(traceback.format_exc())

    # ==================== Tab 2: 個人報告卡 ====================
    with tab2:
        st.header("👤 個人報告卡")
        
        try:
            selected_aid = st.text_input(
                "🔍 輸入 AID 查詢",
                placeholder="例如: 12345678",
                help="從 Tab 1 的表格中複製 AID 貼到這裡"
            )

            if selected_aid:
                with st.spinner(f"⚡ 載入 AID {selected_aid} 的數據..."):
                    client_data = de.get_client_details(
                        display_df,
                        selected_aid,
                        initial_balance,
                        scalper_threshold_seconds
                    )

                if client_data:
                    behavioral = client_data['behavioral']
                    rank_overall, total_overall = de.get_client_ranking(
                        aid_stats_df,
                        selected_aid,
                        'Net_PL'
                    )

                    st.markdown("---")
                    st.markdown(f"## 🆔 AID: {selected_aid}")
                    if rank_overall:
                        st.markdown(f"**🏆 整體排名: 第 {rank_overall} 名 / {total_overall} 人**")

                    # 核心指標
                    st.markdown("### 🎯 核心指標")
                    c1, c2, c3, c4, c5, c6 = st.columns(6)
                    pl_icon = "🟢" if client_data['net_pl'] >= 0 else "🔴"
                    c1.metric(f"{pl_icon} 總盈虧", f"${client_data['net_pl']:,.2f}")
                    c2.metric("🎯 勝率", f"{client_data['win_rate']:.1f}%")
                    c3.metric("📊 PF", f"{client_data['profit_factor']:.2f}")
                    c4.metric("📈 Sharpe", f"{client_data['sharpe']:.2f}")
                    mdd_icon = "🔴" if client_data['mdd_pct'] > 20 else ""
                    c5.metric(f"{mdd_icon}MDD%", f"{client_data['mdd_pct']:.1f}%")
                    c6.metric("📝 筆數", f"{client_data['trade_count']}")

                    # Box Plot 指標
                    st.markdown("### 📦 盈虧分佈統計")
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Q1 (25%)", f"${behavioral['q1']:,.2f}")
                    b2.metric("Median", f"${behavioral['median']:,.2f}")
                    b3.metric("Q3 (75%)", f"${behavioral['q3']:,.2f}")
                    b4.metric("IQR", f"${behavioral['iqr']:,.2f}")

                    st.markdown("---")
                    st.markdown("### ⚔️ 行為分析")
                    ba1, ba2 = st.columns(2)

                    with ba1:
                        st.markdown("#### 多空拆解")
                        st.dataframe(pd.DataFrame({
                            '方向': ['🟢 BUY', '🔴 SELL'],
                            '佔比': [f"{behavioral['buy_ratio']:.1f}%", f"{behavioral['sell_ratio']:.1f}%"],
                            '盈虧': [f"${behavioral['buy_pl']:,.2f}", f"${behavioral['sell_pl']:,.2f}"],
                            '勝率': [f"{behavioral['buy_winrate']:.1f}%", f"{behavioral['sell_winrate']:.1f}%"]
                        }), use_container_width=True, hide_index=True)

                    with ba2:
                        st.markdown("#### 剝頭皮診斷")
                        st.dataframe(pd.DataFrame({
                            '指標': ['Scalp%', '盈虧貢獻', 'Scalp勝率'],
                            '數值': [
                                f"{behavioral['scalp_ratio']:.1f}%",
                                f"{behavioral['scalp_contribution']:.1f}%",
                                f"{behavioral['scalp_winrate']:.1f}%"
                            ]
                        }), use_container_width=True, hide_index=True)

                    st.markdown("---")
                    st.markdown("### 📈 連續紀錄 & 時間效率")
                    s1, s2 = st.columns(2)
                    with s1:
                        st.dataframe(pd.DataFrame({
                            '類型': ['🏆 連續獲利', '💔 連續虧損'],
                            '次數': [f"{behavioral['max_win_streak']} 次", f"{behavioral['max_loss_streak']} 次"],
                            '金額': [f"${behavioral['max_streak_profit']:,.2f}", f"${behavioral['max_streak_loss']:,.2f}"]
                        }), use_container_width=True, hide_index=True)
                    with s2:
                        st.dataframe(pd.DataFrame({
                            '指標': ['平均持倉', '天數', '分鐘獲利'],
                            '數值': [
                                behavioral['avg_hold_formatted'],
                                f"{behavioral['avg_hold_days']:.2f}",
                                f"${behavioral['profit_per_minute']:.4f}"
                            ]
                        }), use_container_width=True, hide_index=True)

                    st.markdown("---")
                    ch1, ch2 = st.columns(2)
                    with ch1:
                        st.plotly_chart(
                            lm.create_client_cumulative_chart(
                                client_data['cumulative_df'],
                                scalper_minutes
                            ),
                            use_container_width=True
                        )
                    with ch2:
                        personal_style = lm.create_trading_style_pie(
                            client_data['client_df'],
                            f"{selected_aid} 風格"
                        )
                        if personal_style:
                            st.plotly_chart(personal_style, use_container_width=True)
                    
                    # ==================== 新增：個人產品盈虧分析 ====================
                    st.markdown("---")
                    st.markdown("### 📦 產品盈虧分析")
                    
                    # 計算產品盈虧
                    product_breakdown = de.calculate_client_product_breakdown(
                        client_data['client_df'],
                        scalper_threshold_seconds
                    )
                    
                    if not product_breakdown.empty:
                        # 分離盈利和虧損產品
                        profit_products = product_breakdown[product_breakdown['Total_PL'] > 0]
                        loss_products = product_breakdown[product_breakdown['Total_PL'] < 0]
                        
                        prod1, prod2 = st.columns(2)
                        
                        with prod1:
                            if not profit_products.empty:
                                profit_chart = lm.plot_top_products_bar(profit_products, is_profit=True, top_n=5)
                                if profit_chart:
                                    st.plotly_chart(profit_chart, use_container_width=True)
                            else:
                                st.info("📊 無盈利產品")
                        
                        with prod2:
                            if not loss_products.empty:
                                loss_chart = lm.plot_top_products_bar(loss_products, is_profit=False, top_n=5)
                                if loss_chart:
                                    st.plotly_chart(loss_chart, use_container_width=True)
                            else:
                                st.info("📊 無虧損產品")
                    else:
                        st.info("📊 無產品數據可顯示")
                    
                else:
                    st.warning(f"❌ 找不到 AID: {selected_aid} 的數據")
            else:
                st.info("請輸入或貼上一個 AID 查看報告卡")
                
        except Exception as e:
            st.error(f"❌ Tab 2 顯示時發生錯誤: {e}")
            import traceback
            with st.expander("🔍 錯誤詳情"):
                st.code(traceback.format_exc())

    # ==================== Tab 3: 當日數據 ====================
    with tab3:
        st.header("📅 當日數據概覽")
        
        try:
            exec_col = de.COLUMN_MAP['execution_time']
            closing_df = de.filter_closing_trades(display_df)
            
            if closing_df.empty:
                st.warning("無平倉交易數據")
                return
            
            latest_date = closing_df[exec_col].dt.date.max()
            st.info(f"📆 分析日期: **{latest_date}**")

            day_df = closing_df[closing_df[exec_col].dt.date == latest_date].copy()

            if day_df.empty:
                st.warning("當日無交易數據")
            else:
                day_pl = day_df['Net_PL'].sum()
                day_count = len(day_df)
                day_accounts = day_df[aid_col].nunique()
                day_wins = (day_df['Net_PL'] > 0).sum()
                day_wr = (day_wins / day_count * 100) if day_count > 0 else 0

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("當日總盈虧", f"${day_pl:,.2f}", delta="盈利" if day_pl >= 0 else "虧損")
                k2.metric("當日交易筆數", f"{day_count:,}")
                k3.metric("當日活躍帳號", f"{day_accounts:,}")
                k4.metric("當日勝率", f"{day_wr:.1f}%")

                st.markdown("---")

                st.markdown("### 📊 當日產品分析")
                profit_products, loss_products = de.calculate_product_scalp_breakdown(
                    day_df,
                    scalper_threshold_seconds
                )
                p1, p2 = st.columns(2)
                with p1:
                    profit_chart = lm.create_stacked_product_chart(profit_products, True)
                    if profit_chart:
                        st.plotly_chart(profit_chart, use_container_width=True)
                    else:
                        st.info("無盈利產品")
                with p2:
                    loss_chart = lm.create_stacked_product_chart(loss_products, False)
                    if loss_chart:
                        st.plotly_chart(loss_chart, use_container_width=True)
                    else:
                        st.info("無虧損產品")

                st.markdown("---")

                # 當日盈利客戶表格
                st.markdown("### 🏆 當日客戶表格")
                min_pnl_d1, min_wr_d1, min_sharpe_d1, max_mdd_d1 = lm.render_global_filters(
                    "daily_hero", 0.0, 0.0, -10.0, 100.0
                )

                daily_hero = de.calculate_hero_metrics(
                    day_df,
                    initial_balance,
                    scalper_threshold_seconds,
                    filter_positive=True,
                    min_pnl=min_pnl_d1,
                    min_winrate=min_wr_d1,
                    min_sharpe=min_sharpe_d1,
                    max_mdd=max_mdd_d1
                )

                if not daily_hero.empty:
                    st.dataframe(
                        lm.format_hero_table_display(daily_hero),
                        use_container_width=True,
                        hide_index=True,
                        column_config=lm.get_table_column_config()
                    )
                    csv_data = daily_hero.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        "📥 下載當日客戶 CSV",
                        data=csv_data,
                        file_name=f"daily_hero_{latest_date}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("當日無盈利客戶符合條件")

                st.markdown("---")

                # 當日 Scalper 客戶表格
                st.markdown("### 🔥 當日剝頭皮客戶表格")
                min_scalp_pct_d, min_scalp_pl_d = lm.render_scalper_filters("daily_scalp", 80.0, 0.0)
                min_pnl_ds, min_wr_ds, min_sharpe_ds, max_mdd_ds = lm.render_global_filters(
                    "daily_scalp_g", 0.0, 0.0, -10.0, 100.0
                )

                daily_scalp = de.calculate_hero_metrics(
                    day_df,
                    initial_balance,
                    scalper_threshold_seconds,
                    filter_positive=True,
                    min_scalp_pct=min_scalp_pct_d,
                    min_scalp_pl=min_scalp_pl_d,
                    min_pnl=min_pnl_ds,
                    min_winrate=min_wr_ds,
                    min_sharpe=min_sharpe_ds,
                    max_mdd=max_mdd_ds
                )

                if not daily_scalp.empty:
                    st.dataframe(
                        lm.format_hero_table_display(daily_scalp),
                        use_container_width=True,
                        hide_index=True,
                        column_config=lm.get_table_column_config()
                    )
                    csv_scalp = daily_scalp.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        "📥 下載剝頭皮客戶 CSV",
                        data=csv_scalp,
                        file_name=f"scalper_{latest_date}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("當日無符合條件的剝頭皮客戶")
                    
        except Exception as e:
            st.error(f"❌ Tab 3 顯示時發生錯誤: {e}")
            import traceback
            with st.expander("🔍 錯誤詳情"):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
