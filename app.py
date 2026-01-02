"""
パチスロ小役カウンター＆設定判別アプリ
Streamlit Webアプリケーション
4色の小役（黄・赤・緑・青）に対応
スマホ対応レイアウト
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from statistics_utils import (
    binomial_p_value,
    calculate_relative_likelihood,
    evaluate_setting,
)

# ページ設定
st.set_page_config(
    page_title="パチスロ設定判別ツール",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 小役の色設定
KOYAKU_COLORS = {
    "黄": {"color": "#FFD700", "bg": "#FFF8DC", "icon": "🟡"},
    "赤": {"color": "#FF4444", "bg": "#FFE4E1", "icon": "🔴"},
    "緑": {"color": "#32CD32", "bg": "#F0FFF0", "icon": "🟢"},
    "青": {"color": "#4169E1", "bg": "#F0F8FF", "icon": "🔵"},
}

# カスタムCSS（モバイル対応）
st.markdown("""
<style>
    /* 全体のパディングを減らす */
    .block-container {
        padding-top: 1rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    
    .main-header {
        font-size: 1.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* 4列カウンターを強制横並び */
    [data-testid="stHorizontalBlock"]:has(.counter-box) {
        display: flex !important;
        flex-wrap: nowrap !important;
        gap: 0.25rem !important;
    }
    
    [data-testid="stHorizontalBlock"]:has(.counter-box) > [data-testid="stColumn"] {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        width: 25% !important;
    }
    
    /* コンパクトなカウンターボックス */
    .counter-box {
        padding: 0.2rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        min-width: 0;
    }
    
    .counter-value {
        font-size: 1.5rem;
        font-weight: bold;
        line-height: 1.1;
    }
    
    .counter-icon {
        font-size: 1rem;
    }
    
    /* ボタンスタイル */
    .stButton > button {
        width: 100%;
        padding: 0.2rem 0;
        font-size: 1rem;
        min-height: 36px;
    }
    
    /* 判別結果 */
    .setting-result {
        padding: 0.5rem 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        font-size: 0.9rem;
        color: #333 !important;
    }
    .setting-positive {
        background-color: #C8E6C9;
        border-left: 4px solid #4CAF50;
        color: #1B5E20 !important;
    }
    .setting-neutral {
        background-color: #FFF9C4;
        border-left: 4px solid #FFC107;
        color: #6D4C00 !important;
    }
    .setting-negative {
        background-color: #FFCDD2;
        border-left: 4px solid #F44336;
        color: #B71C1C !important;
    }
    
    /* 確率入力行を強制横並び */
    .prob-input-row {
        display: flex;
        align-items: center;
        gap: 0.3rem;
        margin: 0.2rem 0;
    }
    
    /* 確率入力の3列も横並び強制 */
    [data-testid="stHorizontalBlock"]:has([data-testid="stTextInput"]) {
        display: flex !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
    }
    
    [data-testid="stHorizontalBlock"]:has([data-testid="stTextInput"]) > [data-testid="stColumn"] {
        flex: 0 0 auto !important;
        min-width: 0 !important;
    }
    
    /* タブを画面幅いっぱいに等間隔で配置 */
    .stTabs [data-baseweb="tab-list"] {
        width: 100%;
        display: flex !important;
        justify-content: stretch !important;
        gap: 0 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        flex: 1 1 0 !important;
        text-align: center;
        font-size: 1rem;
        padding: 0.6rem 0;
        justify-content: center;
    }
    
    /* 回転数入力の幅 */
    .spin-input input {
        font-size: 1rem;
    }
    
    /* 小さい画面でのレスポンシブ対応 */
    @media (max-width: 768px) {
        .counter-value {
            font-size: 1.3rem;
        }
        .counter-icon {
            font-size: 0.9rem;
        }
        .stButton > button {
            font-size: 0.9rem;
            min-height: 32px;
            padding: 0.15rem 0;
        }
        /* タブを小さく */
        .stTabs [data-baseweb="tab"] {
            font-size: 0.8rem;
            padding: 0.4rem 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# セッションステートの初期化
if 'counts' not in st.session_state:
    st.session_state.counts = {"黄": 0, "赤": 0, "緑": 0, "青": 0}

if 'total_spins' not in st.session_state:
    st.session_state.total_spins = 1000

if 'probabilities' not in st.session_state:
    # 各色の設定別確率（デフォルト値）
    default_probs = {
        "設定1": "7.0",
        "設定2": "6.8",
        "設定3": "6.5",
        "設定4": "6.2",
        "設定5": "5.8",
        "設定6": "5.5"
    }
    st.session_state.probabilities = {
        "黄": default_probs.copy(),
        "赤": default_probs.copy(),
        "緑": default_probs.copy(),
        "青": default_probs.copy(),
    }

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {"黄": None, "赤": None, "緑": None, "青": None}


def increment_count(color: str):
    """指定色のカウントを1増やす"""
    st.session_state.counts[color] += 1


def decrement_count(color: str):
    """指定色のカウントを1減らす（0未満にはならない）"""
    if st.session_state.counts[color] > 0:
        st.session_state.counts[color] -= 1


def reset_all_counts():
    """全てのカウントをリセット"""
    for color in st.session_state.counts:
        st.session_state.counts[color] = 0


def run_analysis(color: str, significance: float = 0.05):
    """指定色の分析を実行"""
    probs = st.session_state.probabilities[color]
    count = st.session_state.counts[color]
    n = st.session_state.total_spins
    
    # 確率をパース
    parsed_probs = {}
    for setting, prob_str in probs.items():
        try:
            denominator = float(prob_str)
            parsed_probs[setting] = 1.0 / denominator if denominator > 0 else 0
        except (ValueError, ZeroDivisionError):
            parsed_probs[setting] = 0
    
    # 尤度計算
    prob_list = list(parsed_probs.values())
    relative_likelihoods = calculate_relative_likelihood(n, count, prob_list)
    
    # 各設定の結果を計算（元の順番を維持）
    results = []
    for i, (setting, p) in enumerate(parsed_probs.items()):
        if p > 0:
            p_value = binomial_p_value(n, count, p)
            symbol, eval_text = evaluate_setting(p_value, significance)
            likelihood_pct = relative_likelihoods[i] * 100
        else:
            p_value = 0
            symbol = "?"
            eval_text = "確率未設定"
            likelihood_pct = 0
        
        results.append({
            "setting": setting,
            "probability": p,
            "p_value": p_value,
            "symbol": symbol,
            "eval_text": eval_text,
            "likelihood": likelihood_pct
        })
    
    # 設定6〜1の降順にソート
    results.sort(key=lambda x: int(x["setting"].replace("設定", "")), reverse=True)
    st.session_state.analysis_results[color] = {
        "results": results,
        "parsed_probs": parsed_probs,
        "count": count,
        "total_spins": n
    }


def generate_probability_distribution_graph(parsed_probs: dict, n: int, observed_count: int, color_info: dict):
    """
    X軸: 出現回数、Y軸: 各設定の確率密度（二項分布のPMF）
    """
    fig = go.Figure()
    
    setting_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', 
        '#96CEB4', '#FFEAA7', '#DDA0DD'
    ]
    
    # X軸の範囲を決定
    # 全設定の平均を考慮して範囲を設定
    all_means = [p * n for p in parsed_probs.values() if p > 0]
    if all_means:
        center = np.mean(all_means)
        # 標準偏差の4倍程度の範囲
        max_std = max(np.sqrt(p * (1-p) * n) for p in parsed_probs.values() if p > 0)
        x_min = max(0, int(center - 4 * max_std))
        x_max = int(center + 4 * max_std)
    else:
        x_min, x_max = 0, n
    
    x_values = np.arange(x_min, x_max + 1)
    
    # 各設定の確率分布を描画
    for i, (setting, p) in enumerate(parsed_probs.items()):
        if p > 0:
            # 二項分布の確率質量関数
            pmf = stats.binom.pmf(x_values, n, p)
            
            fig.add_trace(go.Scatter(
                x=x_values,
                y=pmf,
                name=setting,
                line=dict(color=setting_colors[i % len(setting_colors)], width=2),
                mode='lines',
                fill='tozeroy',
                fillcolor=f'rgba{tuple(list(int(setting_colors[i % len(setting_colors)].lstrip("#")[j:j+2], 16) for j in (0, 2, 4)) + [0.1])}'
            ))
    
    # 観測値をマーク
    fig.add_vline(
        x=observed_count,
        line=dict(color=color_info["color"], width=3, dash="dash"),
        annotation_text=f"観測値: {observed_count}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="各設定の確率分布",
        xaxis_title="出現回数",
        yaxis_title="確率密度",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=350,
        margin=dict(t=80, l=50, r=20, b=50)
    )
    
    return fig


def main():
    # ヘッダー
    st.markdown('<div class="main-header">🎰 パチスロ設定判別</div>', unsafe_allow_html=True)
    
    # ====== カウンターセクション ======
    # 総回転数入力（テキスト入力、プラマイボタンなし）
    spin_cols = st.columns([1, 2])
    with spin_cols[0]:
        st.markdown("<div style='line-height:38px;'>総回転数:</div>", unsafe_allow_html=True)
    with spin_cols[1]:
        spin_input = st.text_input(
            "総回転数",
            value=str(st.session_state.total_spins),
            key="spin_input",
            label_visibility="collapsed"
        )
        try:
            st.session_state.total_spins = int(spin_input) if spin_input else 1000
        except ValueError:
            pass
    
    # 4色のカウンター（横並び、コンパクト）
    cols = st.columns(4, gap="small")
    
    for i, (color_key, color_info) in enumerate(KOYAKU_COLORS.items()):
        with cols[i]:
            # ＋ボタン（上）
            if st.button("➕", key=f"inc_{color_key}", use_container_width=True, type="primary"):
                increment_count(color_key)
                st.rerun()
            
            # カウンター表示
            st.markdown(
                f"""<div class="counter-box" style="background-color: {color_info['bg']}; border: 2px solid {color_info['color']};">
                    <div class="counter-icon">{color_info['icon']}</div>
                    <div class="counter-value" style="color: {color_info['color']};">{st.session_state.counts[color_key]}</div>
                </div>""",
                unsafe_allow_html=True
            )
            
            # ーボタン（下）
            if st.button("➖", key=f"dec_{color_key}", use_container_width=True):
                decrement_count(color_key)
                st.rerun()
    
    # 全リセットボタン（カウンターの下）
    if st.button("🗑️ 全リセット", use_container_width=True):
        reset_all_counts()
        st.rerun()
    
    st.divider()
    
    # ====== 分析タブセクション ======
    tab_yellow, tab_red, tab_green, tab_blue = st.tabs([
        "🟡 黄", "🔴 赤", "🟢 緑", "🔵 青"
    ])
    
    tabs = [
        (tab_yellow, "黄"),
        (tab_red, "赤"),
        (tab_green, "緑"),
        (tab_blue, "青")
    ]
    
    for tab, color_key in tabs:
        with tab:
            color_info = KOYAKU_COLORS[color_key]
            
            # 確率入力セクション
            st.subheader(f"{color_info['icon']} 確率設定")
            
            # 1行ずつ表示、設定6〜1の降順
            settings = ["設定6", "設定5", "設定4", "設定3", "設定2", "設定1"]
            
            for setting in settings:
                sub_cols = st.columns([1.2, 0.5, 1.3])
                with sub_cols[0]:
                    st.markdown(f"<div style='line-height:38px;'><b>{setting}</b></div>", unsafe_allow_html=True)
                with sub_cols[1]:
                    st.markdown("<div style='line-height:38px; text-align:right;'>1/</div>", unsafe_allow_html=True)
                with sub_cols[2]:
                    st.session_state.probabilities[color_key][setting] = st.text_input(
                        f"{setting}",
                        value=st.session_state.probabilities[color_key][setting],
                        key=f"prob_{color_key}_{setting}",
                        label_visibility="collapsed"
                    )
            
            # 分析ボタン
            if st.button(
                f"🔍 分析",
                key=f"analyze_{color_key}",
                type="primary",
                use_container_width=True
            ):
                run_analysis(color_key)
                st.rerun()
            
            # 分析結果表示
            analysis = st.session_state.analysis_results[color_key]
            
            if analysis is not None:
                st.divider()
                st.subheader("🔍 判別結果")
                
                st.write(f"**{analysis['count']}回 / {analysis['total_spins']}回転**")
                
                # 各設定の結果（元の順序を維持）
                for r in analysis["results"]:
                    if r["symbol"] == "◎":
                        style_class = "setting-positive"
                    elif r["symbol"] in ["○", "△"]:
                        style_class = "setting-neutral"
                    else:
                        style_class = "setting-negative"
                    
                    if r["probability"] > 0:
                        prob_display = f"1/{1/r['probability']:.1f}"
                    else:
                        prob_display = "未設定"
                    
                    st.markdown(
                        f"""<div class="setting-result {style_class}">
                        <strong>{r["setting"]}</strong> ({prob_display}): 
                        {r["symbol"]} {r["eval_text"]} | 
                        尤度: {r["likelihood"]:.1f}%
                        </div>""",
                        unsafe_allow_html=True
                    )
                
                # 最も可能性の高い設定を表示
                best = max(analysis["results"], key=lambda x: x["likelihood"])
                st.success(f"📍 最も可能性が高い: **{best['setting']}** ({best['likelihood']:.1f}%)")
                
                # グラフ表示
                st.subheader("📈 確率分布")
                fig = generate_probability_distribution_graph(
                    analysis["parsed_probs"],
                    analysis["total_spins"],
                    analysis["count"],
                    color_info
                )
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
