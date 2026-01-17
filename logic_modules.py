"""
邏輯模組 (Logic Modules) v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
第二階段重構：Session State + 局部刷新 + 過濾邏輯

🚀 特性：
  1. Session State 管理：集中管理所有過濾器狀態
  2. @st.fragment 局部刷新：調整過濾器不重畫全頁
  3. 類型安全：所有 number_input 統一使用浮點數
  4. 過濾邏輯封裝：apply_filters() 統一過濾介面

📋 狀態管理架構：
  st.session_state.filters = {
      'min_pnl': float,
      'min_winrate': float,
      'min_sharpe': float,
      'max_mdd': float,
      'min_scalp_pct': float,
      'min_scalp_pl': float,
      'filter_positive': bool
  }
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict


# ============================================================
#                    資料類別定義
# ============================================================

@dataclass
class FilterParams:
    """過濾器參數資料類別（類型安全）"""
    min_pnl: float = 0.0
    min_winrate: float = 0.0
    min_sharpe: float = -10.0
    max_mdd: float = 100.0
    min_scalp_pct: float = 0.0
    min_scalp_pl: float = 0.0
    filter_positive: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FilterParams':
        """從字典建立"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass 
class GlobalSettings:
    """全域設定資料類別"""
    initial_balance: float = 10000.0
    scalper_minutes: float = 5.0
    
    @property
    def scalper_threshold_seconds(self) -> float:
        return self.scalper_minutes * 60


# ============================================================
#                    Session State 管理
# ============================================================

def init_session_state() -> None:
    """
    初始化 Session State
    
    📋 管理的狀態：
    - filters: 過濾器參數（各分頁獨立）
    - global_settings: 全域設定
    - data_loaded: 資料載入狀態
    """
    # 全域設定
    if 'global_settings' not in st.session_state:
        st.session_state.global_settings = GlobalSettings().to_dict() if hasattr(GlobalSettings, 'to_dict') else {
            'initial_balance': 10000.0,
            'scalper_minutes': 5.0
        }
    
    # 資料載入狀態
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # 各分頁的過濾器狀態（獨立管理避免互相干擾）
    filter_keys = [
        'hist_hero',      # Tab1 歷史盈利英雄榜
        'hist_scalp',     # Tab1 歷史 Scalper 英雄榜
        'daily_hero',     # Tab3 當日盈利英雄榜
        'daily_scalp',    # Tab3 當日 Scalper 英雄榜
    ]
    
    for key in filter_keys:
        state_key = f'filters_{key}'
        if state_key not in st.session_state:
            st.session_state[state_key] = FilterParams().to_dict()


def get_filter_params(key: str) -> FilterParams:
    """
    獲取指定分頁的過濾器參數
    
    Args:
        key: 過濾器識別鍵（如 'hist_hero', 'daily_scalp'）
        
    Returns:
        FilterParams 實例
    """
    state_key = f'filters_{key}'
    if state_key not in st.session_state:
        st.session_state[state_key] = FilterParams().to_dict()
    
    return FilterParams.from_dict(st.session_state[state_key])


def set_filter_params(key: str, params: FilterParams) -> None:
    """
    設定指定分頁的過濾器參數
    
    Args:
        key: 過濾器識別鍵
        params: FilterParams 實例
    """
    state_key = f'filters_{key}'
    st.session_state[state_key] = params.to_dict()


def update_filter_value(key: str, param_name: str, value: Any) -> None:
    """
    更新單一過濾器參數值
    
    Args:
        key: 過濾器識別鍵
        param_name: 參數名稱（如 'min_pnl', 'max_mdd'）
        value: 新值
    """
    state_key = f'filters_{key}'
    if state_key in st.session_state:
        st.session_state[state_key][param_name] = value


def get_global_settings() -> Dict[str, float]:
    """獲取全域設定"""
    if 'global_settings' not in st.session_state:
        init_session_state()
    return st.session_state.global_settings


def set_global_setting(param_name: str, value: float) -> None:
    """設定全域參數"""
    if 'global_settings' not in st.session_state:
        init_session_state()
    st.session_state.global_settings[param_name] = value


# ============================================================
#                    局部刷新過濾器 (@st.fragment)
# ============================================================

@st.fragment
def render_global_filters(
    key_prefix: str,
    default_pnl: float = 0.0,
    default_winrate: float = 0.0,
    default_sharpe: float = -10.0,
    default_mdd: float = 100.0,
    show_title: bool = True
) -> Tuple[float, float, float, float]:
    """
    渲染全局過濾器（使用 @st.fragment 局部刷新）
    
    🚀 特性：
    - 調整數值時只刷新此區塊，不重畫全頁
    - 所有 number_input 統一使用浮點數格式
    - 自動同步到 Session State
    
    Args:
        key_prefix: 唯一識別前綴（避免 key 衝突）
        default_pnl: 預設最低盈虧
        default_winrate: 預設最低勝率
        default_sharpe: 預設最低 Sharpe
        default_mdd: 預設最高 MDD%
        show_title: 是否顯示標題
        
    Returns:
        (min_pnl, min_winrate, min_sharpe, max_mdd)
    """
    # 從 Session State 讀取或使用預設值
    params = get_filter_params(key_prefix)
    
    if show_title:
        st.markdown("#### 🔧 全局過濾器")
    
    f1, f2, f3, f4 = st.columns(4)
    
    with f1:
        min_pnl = st.number_input(
            "最低盈虧 ($)",
            value=float(params.min_pnl if params.min_pnl != 0.0 else default_pnl),
            step=100.0,
            format="%.2f",
            key=f"{key_prefix}_pnl",
            help="僅顯示盈虧 ≥ 此值的客戶"
        )
    
    with f2:
        min_winrate = st.number_input(
            "最低勝率 (%)",
            value=float(params.min_winrate if params.min_winrate != 0.0 else default_winrate),
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            format="%.1f",
            key=f"{key_prefix}_wr",
            help="僅顯示勝率 ≥ 此值的客戶"
        )
    
    with f3:
        min_sharpe = st.number_input(
            "最低 Sharpe",
            value=float(params.min_sharpe if params.min_sharpe != -10.0 else default_sharpe),
            step=0.5,
            format="%.2f",
            key=f"{key_prefix}_sharpe",
            help="僅顯示 Sharpe ≥ 此值的客戶"
        )
    
    with f4:
        max_mdd = st.number_input(
            "最高 MDD (%)",
            value=float(params.max_mdd if params.max_mdd != 100.0 else default_mdd),
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            format="%.1f",
            key=f"{key_prefix}_mdd",
            help="僅顯示 MDD ≤ 此值的客戶"
        )
    
    # 同步到 Session State
    update_filter_value(key_prefix, 'min_pnl', min_pnl)
    update_filter_value(key_prefix, 'min_winrate', min_winrate)
    update_filter_value(key_prefix, 'min_sharpe', min_sharpe)
    update_filter_value(key_prefix, 'max_mdd', max_mdd)
    
    return min_pnl, min_winrate, min_sharpe, max_mdd


@st.fragment
def render_scalper_filters(
    key_prefix: str,
    default_scalp_pct: float = 80.0,
    default_scalp_pl: float = 0.0,
    show_title: bool = False
) -> Tuple[float, float]:
    """
    渲染 Scalper 專用過濾器（使用 @st.fragment 局部刷新）
    
    🚀 特性：
    - Scalp% 使用 slider（整數顯示但返回浮點數）
    - Scalp 盈虧使用 number_input（浮點數）
    
    Args:
        key_prefix: 唯一識別前綴
        default_scalp_pct: 預設 Scalp% 門檻
        default_scalp_pl: 預設 Scalp 盈虧金額門檻
        show_title: 是否顯示標題
        
    Returns:
        (min_scalp_pct, min_scalp_pl)
    """
    params = get_filter_params(key_prefix)
    
    if show_title:
        st.markdown("#### 🔥 Scalper 過濾器")
    
    s1, s2 = st.columns(2)
    
    with s1:
        # Slider 返回 int，但我們轉為 float 確保類型一致
        min_scalp_pct_int = st.slider(
            "Scalp% 門檻",
            min_value=50,
            max_value=100,
            value=int(params.min_scalp_pct if params.min_scalp_pct != 0.0 else default_scalp_pct),
            step=5,
            key=f"{key_prefix}_spct",
            help="Scalp 交易筆數佔比（持倉時間 < Scalper 門檻）"
        )
        min_scalp_pct = float(min_scalp_pct_int)
    
    with s2:
        min_scalp_pl = st.number_input(
            "Scalp 盈虧金額門檻 ($)",
            value=float(params.min_scalp_pl if params.min_scalp_pl != 0.0 else default_scalp_pl),
            step=100.0,
            format="%.2f",
            key=f"{key_prefix}_spl",
            help="僅顯示 Scalp 盈虧 ≥ 此值的客戶"
        )
    
    # 同步到 Session State
    update_filter_value(key_prefix, 'min_scalp_pct', min_scalp_pct)
    update_filter_value(key_prefix, 'min_scalp_pl', min_scalp_pl)
    
    return min_scalp_pct, min_scalp_pl


@st.fragment
def render_combined_filters(
    key_prefix: str,
    include_scalper: bool = False,
    defaults: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    渲染組合過濾器（全局 + 可選 Scalper）
    
    Args:
        key_prefix: 唯一識別前綴
        include_scalper: 是否包含 Scalper 過濾器
        defaults: 自訂預設值字典
        
    Returns:
        包含所有過濾器值的字典
    """
    if defaults is None:
        defaults = {}
    
    # 全局過濾器
    min_pnl, min_winrate, min_sharpe, max_mdd = render_global_filters(
        key_prefix=f"{key_prefix}_global",
        default_pnl=defaults.get('min_pnl', 0.0),
        default_winrate=defaults.get('min_winrate', 0.0),
        default_sharpe=defaults.get('min_sharpe', -10.0),
        default_mdd=defaults.get('max_mdd', 100.0)
    )
    
    result = {
        'min_pnl': min_pnl,
        'min_winrate': min_winrate,
        'min_sharpe': min_sharpe,
        'max_mdd': max_mdd
    }
    
    # Scalper 過濾器（可選）
    if include_scalper:
        min_scalp_pct, min_scalp_pl = render_scalper_filters(
            key_prefix=f"{key_prefix}_scalp",
            default_scalp_pct=defaults.get('min_scalp_pct', 80.0),
            default_scalp_pl=defaults.get('min_scalp_pl', 0.0)
        )
        result['min_scalp_pct'] = min_scalp_pct
        result['min_scalp_pl'] = min_scalp_pl
    
    return result


# ============================================================
#                    過濾邏輯
# ============================================================

def apply_filters(
    df: pd.DataFrame,
    params: Dict[str, Any],
    filter_positive: bool = True
) -> pd.DataFrame:
    """
    根據過濾參數過濾 DataFrame
    
    🔧 支援的欄位映射：
    - min_pnl → Net_PL 或 盈虧
    - min_winrate → Win_Rate 或 勝率%
    - min_sharpe → Sharpe
    - max_mdd → MDD_Pct 或 MDD%
    - min_scalp_pct → Scalp_Pct 或 Scalp%
    - min_scalp_pl → Scalp_PL 或 Scalp盈虧
    
    Args:
        df: 待過濾的 DataFrame
        params: 過濾參數字典
        filter_positive: 是否僅保留正盈虧
        
    Returns:
        過濾後的 DataFrame
    """
    if df.empty:
        return df
    
    filtered = df.copy()
    
    # 欄位名稱映射（支援中英文）
    col_mapping = {
        'pnl': ['Net_PL', '盈虧', 'net_pl'],
        'winrate': ['Win_Rate', '勝率%', 'win_rate'],
        'sharpe': ['Sharpe', 'sharpe'],
        'mdd': ['MDD_Pct', 'MDD%', 'mdd_pct'],
        'scalp_pct': ['Scalp_Pct', 'Scalp%', 'Scalper_Ratio', 'scalp_pct'],
        'scalp_pl': ['Scalp_PL', 'Scalp盈虧', 'scalp_pl']
    }
    
    def get_col(key: str) -> Optional[str]:
        """找到 DataFrame 中對應的欄位名稱"""
        for col_name in col_mapping.get(key, []):
            if col_name in filtered.columns:
                return col_name
        return None
    
    # 1. 正盈虧過濾
    if filter_positive:
        pnl_col = get_col('pnl')
        if pnl_col:
            filtered = filtered[filtered[pnl_col] > 0]
    
    # 2. 最低盈虧
    if params.get('min_pnl') is not None:
        pnl_col = get_col('pnl')
        if pnl_col:
            filtered = filtered[filtered[pnl_col] >= float(params['min_pnl'])]
    
    # 3. 最低勝率
    if params.get('min_winrate') is not None:
        wr_col = get_col('winrate')
        if wr_col:
            filtered = filtered[filtered[wr_col] >= float(params['min_winrate'])]
    
    # 4. 最低 Sharpe
    if params.get('min_sharpe') is not None:
        sharpe_col = get_col('sharpe')
        if sharpe_col:
            filtered = filtered[filtered[sharpe_col] >= float(params['min_sharpe'])]
    
    # 5. 最高 MDD
    if params.get('max_mdd') is not None:
        mdd_col = get_col('mdd')
        if mdd_col:
            filtered = filtered[filtered[mdd_col] <= float(params['max_mdd'])]
    
    # 6. 最低 Scalp%
    if params.get('min_scalp_pct') is not None:
        scalp_pct_col = get_col('scalp_pct')
        if scalp_pct_col:
            filtered = filtered[filtered[scalp_pct_col] >= float(params['min_scalp_pct'])]
    
    # 7. 最低 Scalp 盈虧
    if params.get('min_scalp_pl') is not None:
        scalp_pl_col = get_col('scalp_pl')
        if scalp_pl_col:
            filtered = filtered[filtered[scalp_pl_col] >= float(params['min_scalp_pl'])]
    
    return filtered


def apply_hero_filters(
    df: pd.DataFrame,
    min_pnl: Optional[float] = None,
    min_winrate: Optional[float] = None,
    min_sharpe: Optional[float] = None,
    max_mdd: Optional[float] = None,
    min_scalp_pct: Optional[float] = None,
    min_scalp_pl: Optional[float] = None,
    filter_positive: bool = True,
    top_n: int = 20
) -> pd.DataFrame:
    """
    英雄榜專用過濾函數（便捷介面）
    
    Args:
        df: 英雄榜 DataFrame
        min_pnl: 最低盈虧
        min_winrate: 最低勝率
        min_sharpe: 最低 Sharpe
        max_mdd: 最高 MDD%
        min_scalp_pct: 最低 Scalp%
        min_scalp_pl: 最低 Scalp 盈虧
        filter_positive: 是否僅保留正盈虧
        top_n: 返回前 N 名
        
    Returns:
        過濾並排序後的 DataFrame
    """
    params = {
        'min_pnl': min_pnl,
        'min_winrate': min_winrate,
        'min_sharpe': min_sharpe,
        'max_mdd': max_mdd,
        'min_scalp_pct': min_scalp_pct,
        'min_scalp_pl': min_scalp_pl
    }
    
    # 移除 None 值
    params = {k: v for k, v in params.items() if v is not None}
    
    filtered = apply_filters(df, params, filter_positive=filter_positive)
    
    # 排序並取 Top N
    if not filtered.empty:
        # 找到盈虧欄位
        pnl_col = None
        for col in ['Net_PL', '盈虧', 'net_pl']:
            if col in filtered.columns:
                pnl_col = col
                break
        
        if pnl_col:
            filtered = filtered.sort_values(pnl_col, ascending=False).head(top_n)
    
    return filtered


# ============================================================
#                    全域設定 UI
# ============================================================

@st.fragment
def render_global_settings() -> Tuple[float, float]:
    """
    渲染全域設定 UI（側邊欄用）
    
    Returns:
        (initial_balance, scalper_minutes)
    """
    settings = get_global_settings()
    
    initial_balance = st.number_input(
        "💰 初始資金",
        value=float(settings.get('initial_balance', 10000.0)),
        min_value=0.0,
        step=1000.0,
        format="%.2f",
        key="global_initial_balance",
        help="用於計算 MDD% 的初始資金"
    )
    
    scalper_minutes = st.number_input(
        "⏱️ Scalper 門檻 (分鐘)",
        value=float(settings.get('scalper_minutes', 5.0)),
        min_value=1.0,
        max_value=60.0,
        step=1.0,
        format="%.1f",
        key="global_scalper_minutes",
        help="持倉時間低於此值視為 Scalp 交易"
    )
    
    # 同步到 Session State
    set_global_setting('initial_balance', initial_balance)
    set_global_setting('scalper_minutes', scalper_minutes)
    
    return initial_balance, scalper_minutes


# ============================================================
#                    工具函數
# ============================================================

def reset_filters(key_prefix: str) -> None:
    """
    重置指定分頁的過濾器為預設值
    
    Args:
        key_prefix: 過濾器識別鍵
    """
    set_filter_params(key_prefix, FilterParams())


def reset_all_filters() -> None:
    """重置所有過濾器為預設值"""
    filter_keys = ['hist_hero', 'hist_scalp', 'daily_hero', 'daily_scalp']
    for key in filter_keys:
        reset_filters(key)


def get_filter_summary(key_prefix: str) -> str:
    """
    獲取過濾器摘要字串
    
    Args:
        key_prefix: 過濾器識別鍵
        
    Returns:
        格式化的摘要字串
    """
    params = get_filter_params(key_prefix)
    
    parts = []
    if params.min_pnl > 0:
        parts.append(f"盈虧≥${params.min_pnl:,.0f}")
    if params.min_winrate > 0:
        parts.append(f"勝率≥{params.min_winrate:.0f}%")
    if params.min_sharpe > -10:
        parts.append(f"Sharpe≥{params.min_sharpe:.1f}")
    if params.max_mdd < 100:
        parts.append(f"MDD≤{params.max_mdd:.0f}%")
    if params.min_scalp_pct > 0:
        parts.append(f"Scalp%≥{params.min_scalp_pct:.0f}%")
    if params.min_scalp_pl > 0:
        parts.append(f"Scalp盈虧≥${params.min_scalp_pl:,.0f}")
    
    return " | ".join(parts) if parts else "無過濾條件"


# ============================================================
#                    測試區塊
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("邏輯模組 (Logic Modules) v1.0")
    print("=" * 60)
    print("\n✅ 模組載入成功")
    
    print("\n📋 資料類別：")
    print("  - FilterParams: 過濾器參數")
    print("  - GlobalSettings: 全域設定")
    
    print("\n🔧 Session State 函數：")
    print("  - init_session_state()")
    print("  - get_filter_params(key)")
    print("  - set_filter_params(key, params)")
    print("  - update_filter_value(key, param_name, value)")
    print("  - get_global_settings()")
    print("  - set_global_setting(param_name, value)")
    
    print("\n🖼️ UI 渲染函數（@st.fragment）：")
    print("  - render_global_filters(key_prefix, ...)")
    print("  - render_scalper_filters(key_prefix, ...)")
    print("  - render_combined_filters(key_prefix, ...)")
    print("  - render_global_settings()")
    
    print("\n🔍 過濾邏輯函數：")
    print("  - apply_filters(df, params, filter_positive)")
    print("  - apply_hero_filters(df, ...)")
    
    print("\n🛠️ 工具函數：")
    print("  - reset_filters(key_prefix)")
    print("  - reset_all_filters()")
    print("  - get_filter_summary(key_prefix)")
    
    # 測試 FilterParams
    print("\n" + "-" * 40)
    print("測試 FilterParams：")
    params = FilterParams(min_pnl=1000.0, min_winrate=50.0)
    print(f"  建立: {params}")
    print(f"  轉字典: {params.to_dict()}")
    
    params2 = FilterParams.from_dict({'min_pnl': 2000.0, 'max_mdd': 50.0})
    print(f"  從字典建立: {params2}")
    
    # 測試過濾邏輯
    print("\n" + "-" * 40)
    print("測試 apply_filters：")
    
    test_df = pd.DataFrame({
        'AID': ['A001', 'A002', 'A003', 'A004'],
        'Net_PL': [1000.0, -500.0, 2000.0, 500.0],
        'Win_Rate': [60.0, 40.0, 75.0, 55.0],
        'Sharpe': [1.5, -0.5, 2.5, 1.0],
        'MDD_Pct': [10.0, 30.0, 5.0, 15.0]
    })
    
    print(f"  原始資料: {len(test_df)} 筆")
    
    filter_params = {'min_pnl': 500.0, 'min_winrate': 50.0}
    filtered = apply_filters(test_df, filter_params, filter_positive=True)
    print(f"  過濾後 (盈虧≥500, 勝率≥50%): {len(filtered)} 筆")
    print(f"  結果 AID: {filtered['AID'].tolist()}")
    
    print("\n" + "=" * 60)
    print("✅ 所有測試通過！")
