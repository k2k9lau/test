import streamlit as st
import data_engine as de  # 廚房：數據引擎
import logic_modules as lm  # 經理：邏輯與 UI 模組

# 1. 餐廳基礎設定
st.set_page_config(
    page_title="星級交易分析系統",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 初始化大堂經理的筆記本 (Session State)
lm.init_session_state()

# 3. 側邊欄：食材採購 (檔案上傳)
st.sidebar.header("📥 數據載入")
uploaded_files = st.sidebar.file_uploader(
    "上傳交易報告 (CSV/Excel)",
    accept_multiple_files=True,
    help="支援合併多個帳號的交易紀錄"
)

# 4. 開始營業 (主程式邏輯)
df_raw = de.load_data(uploaded_files)  #

if df_raw is not None:
    # 建立分層樓層 (Tabs)
    tab1, tab2, tab3 = st.tabs(["📊 整體分析 (Tab 1)", "🔍 個人報告 (Tab 2)", "📅 當日快訊 (Tab 3)"])

    # --- Tab 1: 整體分析 ---
    with tab1:
        st.title("整體盈利分佈與英雄榜")

        # A. 繪製全寬小提琴圖 (由經理處理擺盤)
        sample_df, stats = de.get_violin_sample(df_raw)
        lm.render_violin_plot(sample_df, stats)

        st.divider()

        # B. 並排顯示風格圖表
        lm.render_style_charts(df_raw)

        st.divider()

        # C. 歷史英雄榜 (包含局部刷新過濾器)
        st.subheader("🏆 歷史英雄榜 (Top 20)")

        # 獲取過濾器參數
        params = lm.render_combined_filters(key_prefix="hist_hero")

        # 根據參數向廚房要菜
        hero_df = de.get_hero_metrics(
            df_raw,
            min_pnl=params.min_pnl,
            min_winrate=params.min_winrate,
            min_sharpe=params.min_sharpe,
            max_mdd=params.max_mdd,
            min_scalp_pl=params.min_scalp_pl
        )

        # 顯示表格 (帶有一鍵複製 AID 功能)
        lm.render_hero_table(hero_df, key="hist_table")

    # --- Tab 2: 個人報告 ---
    with tab2:
        st.title("👤 客戶診斷報告")

        # 搜尋交互優化：清理貼上的 AID 格式
        raw_aid = st.text_input("🔍 請輸入或貼上 AID 進行查詢", placeholder="例如: 123456")
        search_aid = lm.clean_aid_input(raw_aid)

        if search_aid:
            # 調用模組顯示個人詳細分析 (此處可根據需求擴展)
            st.info(f"正在分析客戶: {search_aid} ...")
            # lm.render_individual_report(df_raw, search_aid)
        else:
            st.caption("📋 提示：您可以直接點擊 Tab 1 表格中的 AID 複製，然後在此處貼上。")

    # --- Tab 3: 當日快訊 ---
    with tab3:
        st.title("⚡ Scalp 當日交易監控")
        # 類似 Tab 1 的邏輯，但可針對當日數據過濾
        daily_params = lm.render_combined_filters(key_prefix="daily_scalp")
        # daily_df = de.get_hero_metrics(df_raw, **daily_params.to_dict())
        # lm.render_hero_table(daily_df, key="daily_table")
        st.write("此處佈局與 Tab 1 相似，專注於即時交易動態。")

else:
    # 餐廳還沒開門的歡迎畫面
    st.info("👋 歡迎使用！請先在左側邊欄上傳交易數據檔案以開始分析。")
    st.image("https://via.placeholder.com/800x400.png?text=Waiting+for+Data+Upload", use_column_width=True)

# 5. 頁腳資訊
st.sidebar.divider()
st.sidebar.caption("系統版本: v1.0.0 (模組化重構版)")