import streamlit as st
import pandas as pd
import data_engine as de  # 廚房：數據引擎
import logic_modules as lm  # 經理：邏輯與 UI 模組

# 1. 餐廳基礎佈局設定
st.set_page_config(
    page_title="星級交易分析系統",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 初始化大堂經理的筆記本 (Session State)
lm.init_session_state()

# 3. 側邊欄：數據載入與全域參數設定
with st.sidebar:
    st.header("📥 數據載入")
    uploaded_files = st.file_uploader(
        "上傳交易報告 (CSV/Excel)",
        accept_multiple_files=True,
        help="支援合併多個帳號的交易紀錄"
    )

    st.divider()
    st.header("⚙️ 計算設定")
    # 調用經理提供的設定介面（初始資金與 Scalp 分鐘定義）
    init_bal, scalp_min = lm.render_global_settings()

# 4. 開始營業 (主程式邏輯)
df_raw = de.load_data(uploaded_files)

if df_raw is not None and not df_raw.empty:
    # 建立分層樓層 (Tabs)
    tab1, tab2, tab3 = st.tabs(["📊 整體分析", "🔍 個人診斷", "📅 當日快訊"])

    # --- Tab 1: 整體分析 ---
    with tab1:
        st.title("整體盈利分佈與英雄榜")

        # A. 繪製全寬小提琴圖
        sample_df, stats = de.get_violin_sample(df_raw)
        lm.render_violin_plot(sample_df, stats)

        st.divider()

        # B. 並排顯示風格圖表 (由經理處理排版)
        lm.render_style_charts(df_raw)

        st.divider()

        # C. 歷史英雄榜 (Top 20)
        st.subheader("🏆 歷史英雄榜 (Top 20)")

        # 獲取過濾器參數 (lm 回傳的是字典)
        params_dict = lm.render_combined_filters(key_prefix="hist_hero")

        # 根據參數向廚房要菜，並展開字典
        hero_df = de.get_hero_metrics(
            df_raw,
            initial_balance=init_bal,
            scalper_threshold_seconds=scalp_min * 60,
            **params_dict  # 自動解包：min_pnl, min_winrate, min_sharpe, max_mdd, min_scalp_pl
        )

        if not hero_df.empty:
            lm.render_hero_table(hero_df, key="hist_table")
        else:
            st.warning("⚠️ 沒有符合當前過濾條件的客戶。")

    # --- Tab 2: 個人報告 (搜尋功能) ---
    with tab2:
        st.title("👤 客戶診斷報告")
        raw_aid = st.text_input("🔍 請輸入或貼上 AID 進行查詢", placeholder="從表格複製 AID 後在此處貼上...")
        search_aid = lm.clean_aid_input(raw_aid)

        if search_aid:
            st.info(f"正在分析客戶: {search_aid} ... (功能開發中)")
            # 此處可加入 de.get_individual_stats(df_raw, search_aid)
        else:
            st.caption("📋 提示：您可以直接點擊 Tab 1 表格中的 AID 複製按鈕，然後在此處貼上。")

    # --- Tab 3: 當日快訊 ---
    with tab3:
        st.title("⚡ 今日交易英雄榜")

        # 過濾出今天的數據 (使用 data_engine 定義的欄位名)
        today = pd.Timestamp.now().normalize()
        exec_col = de.COLUMN_MAP['execution_time']
        df_today = df_raw[df_raw[exec_col].dt.normalize() == today]

        if not df_today.empty:
            daily_params = lm.render_combined_filters(key_prefix="daily_scalp")
            daily_hero = de.get_hero_metrics(
                df_today,
                initial_balance=init_bal,
                scalper_threshold_seconds=scalp_min * 60,
                **daily_params
            )
            lm.render_hero_table(daily_hero, key="daily_table")
        else:
            st.info(f"📅 今日 ({today.date()}) 尚無交易紀錄。")

else:
    # 歡迎畫面
    st.info("👋 歡迎！請先在左側邊欄上傳交易數據檔案（CSV/Excel）以開始分析。")

st.sidebar.divider()
st.sidebar.caption("系統狀態: 運作正常 (模組化版本)")