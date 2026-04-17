# =============================================================================
# TechScore Stock Analyzer — Streamlit Edition
# Based on TechScore_v1.3 (PyQt5 Desktop Version)
#
# License: GNU General Public License v3
# Disclaimer: 本软件仅供学习与研究，不构成任何投资建议。
# =============================================================================

import os
import sys
import glob
import time
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# 页面基本配置（必须放在最前面）
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TechScore 技术指标综合评分系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 全局路径
# ---------------------------------------------------------------------------
DATA_ROOT = os.path.join(tempfile.gettempdir(), "TechScore_Data")
PREDS_DIR = os.path.join(DATA_ROOT, "Predictions")
for _d in [DATA_ROOT, PREDS_DIR]:
    os.makedirs(_d, exist_ok=True)

MIN_BARS = 50


# =============================================================================
# 技术指标计算引擎（与桌面版完全一致）
# =============================================================================
class TechnicalIndicatorEngine:
    """
    计算 10 个常见技术指标并输出综合评分 (0~100)。
    """

    WEIGHTS = {
        "RSI": 0.12, "MACD": 0.15, "KDJ": 0.12, "BB": 0.10, "MA": 0.12,
        "VOL": 0.10, "ATR": 0.07,  "OBV": 0.10, "WR": 0.06, "CCI": 0.06,
    }

    def calc(self, df: pd.DataFrame) -> dict:
        df = df.copy().reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df["close"] > 0].copy()
        df["volume"] = df["volume"].fillna(0)
        df = df.reset_index(drop=True)

        if len(df) < MIN_BARS:
            return self._empty_result()

        scores, values = {}, {}
        last_date = str(df["date"].iloc[-1]) if "date" in df.columns else ""

        rsi_val, rsi_score = self._rsi(df["close"], 14)
        values["RSI"] = rsi_val; scores["RSI"] = rsi_score

        macd_val, signal_val, hist_val, macd_score = self._macd(df["close"])
        values["MACD"] = macd_val; values["MACD_Signal"] = signal_val
        values["MACD_Hist"] = hist_val; scores["MACD"] = macd_score

        k_val, d_val, j_val, kdj_score = self._kdj(df["high"], df["low"], df["close"])
        values["K"] = k_val; values["D"] = d_val; values["J"] = j_val
        scores["KDJ"] = kdj_score

        bb_val, bb_score = self._bollinger(df["close"], 20, 2)
        values["BB_pct"] = bb_val; scores["BB"] = bb_score

        ma5_val, ma20_val, ma_score = self._ma_cross(df["close"], 5, 20)
        values["MA5"] = ma5_val; values["MA20"] = ma20_val; scores["MA"] = ma_score

        vr_val, vr_score = self._volume_ratio(df["volume"], 20)
        values["VolRatio"] = vr_val; scores["VOL"] = vr_score

        atr_val, atr_score = self._atr_pct(df["high"], df["low"], df["close"], 14)
        values["ATR_pct"] = atr_val; scores["ATR"] = atr_score

        obv_slope, obv_score = self._obv_trend(df["close"], df["volume"], 5)
        values["OBV_Slope"] = obv_slope; scores["OBV"] = obv_score

        wr_val, wr_score = self._williams_r(df["high"], df["low"], df["close"], 14)
        values["WR"] = wr_val; scores["WR"] = wr_score

        cci_val, cci_score = self._cci(df["high"], df["low"], df["close"], 14)
        values["CCI"] = cci_val; scores["CCI"] = cci_score

        total = sum(scores[k] * self.WEIGHTS[k] for k in scores)
        composite = round(total * 10, 2)

        return {
            "values": values, "scores": scores,
            "composite": composite, "last_date": last_date,
        }

    def _empty_result(self):
        return {
            "values": {}, "scores": {k: 0 for k in self.WEIGHTS},
            "composite": 0.0, "last_date": "",
        }

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def _rsi(self, close, period):
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - 100 / (1 + rs)
        val = round(rsi.iloc[-1], 2)
        if val <= 20: score = 10.0
        elif val <= 30: score = 9.0
        elif val <= 40: score = 7.5
        elif val <= 50: score = 6.0
        elif val <= 60: score = 5.0
        elif val <= 70: score = 3.5
        elif val <= 80: score = 2.0
        else: score = 1.0
        return val, score

    def _macd(self, close):
        ema12 = self._ema(close, 12)
        ema26 = self._ema(close, 26)
        macd = ema12 - ema26
        signal = self._ema(macd, 9)
        hist = macd - signal
        macd_v = round(macd.iloc[-1], 4)
        sig_v = round(signal.iloc[-1], 4)
        hist_v = round(hist.iloc[-1], 4)
        if macd_v > sig_v and macd_v > 0: score = 9.0
        elif macd_v > sig_v and macd_v <= 0: score = 7.0
        elif macd_v <= sig_v and macd_v > 0: score = 4.0
        else: score = 1.5
        if len(hist) >= 2:
            if hist.iloc[-1] > hist.iloc[-2]: score = min(10.0, score + 0.5)
            else: score = max(0.0, score - 0.5)
        return macd_v, sig_v, hist_v, round(score, 2)

    def _kdj(self, high, low, close):
        period = 9
        low_n = low.rolling(period).min()
        high_n = high.rolling(period).max()
        rsv = (close - low_n) / (high_n - low_n + 1e-9) * 100
        K = rsv.ewm(com=2, adjust=False).mean()
        D = K.ewm(com=2, adjust=False).mean()
        J = 3 * K - 2 * D
        k_v, d_v, j_v = round(K.iloc[-1], 2), round(D.iloc[-1], 2), round(J.iloc[-1], 2)
        if k_v < 20 and d_v < 20: score = 9.5
        elif k_v < 30 and K.iloc[-1] > K.iloc[-2] and K.iloc[-2] < D.iloc[-2]: score = 8.5
        elif k_v > 80 and d_v > 80: score = 1.5
        elif k_v > 70 and K.iloc[-1] < K.iloc[-2] and K.iloc[-2] > D.iloc[-2]: score = 2.0
        elif k_v > d_v: score = 6.5
        else: score = 4.0
        if j_v < 0: score = min(10.0, score + 1.0)
        elif j_v > 100: score = max(0.0, score - 1.0)
        return k_v, d_v, j_v, round(score, 2)

    def _bollinger(self, close, period, std_mult):
        ma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = ma + std_mult * std
        lower = ma - std_mult * std
        lc, lu, ll = close.iloc[-1], upper.iloc[-1], lower.iloc[-1]
        bw = lu - ll
        bb_pct = 0.5 if bw < 1e-9 else (lc - ll) / bw
        val = round(bb_pct, 4)
        if val < 0: score = 9.5
        elif val < 0.15: score = 8.5
        elif val < 0.35: score = 7.0
        elif val < 0.65: score = 5.5
        elif val < 0.85: score = 3.5
        elif val < 1.0: score = 2.0
        else: score = 1.0
        return val, round(score, 2)

    def _ma_cross(self, close, fast, slow):
        ma_fast = close.rolling(fast).mean()
        ma_slow = close.rolling(slow).mean()
        ma5_v = round(ma_fast.iloc[-1], 4)
        ma20_v = round(ma_slow.iloc[-1], 4)
        prev_diff = ma_fast.iloc[-2] - ma_slow.iloc[-2]
        curr_diff = ma_fast.iloc[-1] - ma_slow.iloc[-1]
        if prev_diff < 0 and curr_diff > 0: score = 10.0
        elif prev_diff > 0 and curr_diff < 0: score = 1.0
        elif curr_diff > 0:
            gap_pct = curr_diff / (ma_slow.iloc[-1] + 1e-9)
            score = 6.0 if gap_pct > 0.05 else 7.5
        else: score = 3.0
        return ma5_v, ma20_v, round(score, 2)

    @staticmethod
    def _volume_ratio(volume, period):
        avg_vol = volume.iloc[-period - 1:-1].mean()
        curr_vol = volume.iloc[-1]
        vr = 1.0 if avg_vol < 1e-9 else curr_vol / avg_vol
        val = round(vr, 3)
        if vr > 4.0: score = 8.5
        elif vr > 2.5: score = 9.0
        elif vr > 1.8: score = 8.0
        elif vr > 1.2: score = 6.5
        elif vr > 0.8: score = 5.0
        elif vr > 0.5: score = 3.0
        else: score = 1.5
        return val, round(score, 2)

    def _atr_pct(self, high, low, close, period):
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low, (high - prev_close).abs(), (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        atr_pct = atr / (close + 1e-9) * 100
        val = round(atr_pct.iloc[-1], 3)
        if 1.0 <= val <= 2.0: score = 8.0
        elif 0.5 <= val < 1.0 or 2.0 < val <= 3.0: score = 6.5
        elif 0.3 <= val < 0.5 or 3.0 < val <= 5.0: score = 4.5
        elif val > 5.0: score = 2.0
        else: score = 3.5
        return val, round(score, 2)

    @staticmethod
    def _obv_trend(close, volume, period):
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * volume).cumsum()
        recent = obv.iloc[-period:].values
        if len(recent) < 2:
            return 0.0, 5.0
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        mean_abs = np.mean(np.abs(recent)) + 1e-9
        norm_slope = slope / mean_abs
        if norm_slope > 0.05: score = 9.0
        elif norm_slope > 0.02: score = 7.5
        elif norm_slope > 0: score = 6.0
        elif norm_slope > -0.02: score = 4.5
        elif norm_slope > -0.05: score = 3.0
        else: score = 1.5
        return round(float(norm_slope), 5), round(score, 2)

    @staticmethod
    def _williams_r(high, low, close, period):
        high_n = high.rolling(period).max()
        low_n = low.rolling(period).min()
        wr = (high_n - close) / (high_n - low_n + 1e-9) * (-100)
        val = round(wr.iloc[-1], 2)
        if val <= -90: score = 9.5
        elif val <= -80: score = 8.0
        elif val <= -50: score = 5.5
        elif val <= -20: score = 3.5
        else: score = 1.5
        return val, round(score, 2)

    @staticmethod
    def _cci(high, low, close, period):
        typical = (high + low + close) / 3
        ma = typical.rolling(period).mean()
        md = typical.rolling(period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        cci = (typical - ma) / (0.015 * md + 1e-9)
        val = round(cci.iloc[-1], 2)
        if val < -200: score = 9.5
        elif val < -100: score = 8.0
        elif val < 0: score = 6.0
        elif val < 100: score = 4.5
        elif val < 200: score = 2.5
        else: score = 1.0
        return val, round(score, 2)


# =============================================================================
# 数据管理器（Streamlit 适配版）
# =============================================================================
class DataManager:
    def __init__(self):
        self.engine = TechnicalIndicatorEngine()

    def build_code_list(self, scopes, end_date):
        """构建股票代码列表"""
        import baostock as bs
        bs.login()
        codes = set()
        try:
            if scopes[0].startswith("single:"):
                codes.add(scopes[0].split(":")[1])
                return sorted(codes)

            if "test10" in scopes:
                rs = bs.query_hs300_stocks()
                cnt = 0
                while rs.next() and cnt < 10:
                    codes.add(rs.get_row_data()[1])
                    cnt += 1
                return sorted(codes)

            if "hs300" in scopes:
                rs = bs.query_hs300_stocks()
                while rs.next():
                    codes.add(rs.get_row_data()[1])
            if "zz500" in scopes:
                rs = bs.query_zz500_stocks()
                while rs.next():
                    codes.add(rs.get_row_data()[1])
            if "all" in scopes:
                rs = bs.query_all_stock(day=end_date)
                while rs.next():
                    c = rs.get_row_data()[0]
                    if c.startswith(("sh.", "sz.")):
                        codes.add(c)
        finally:
            bs.logout()
        return sorted(codes)

    def fetch_and_score(self, codes, start, end, progress_bar, status_text):
        """下载K线 → 计算评分 → 返回结果 DataFrame"""
        import baostock as bs
        bs.login()

        rows = []
        total = len(codes)
        skipped = 0

        for i, code in enumerate(codes):
            pct = (i + 1) / total
            progress_bar.progress(pct, text=f"[{i+1}/{total}] 正在处理: {code}")

            try:
                rs = bs.query_stock_basic(code=code)
                if not rs.data:
                    skipped += 1
                    continue
                info = rs.get_row_data()
                ipo = info[2] if len(info) > 2 else ""
                name = info[1] if len(info) > 1 else code
                if not ipo:
                    skipped += 1
                    continue

                fields = "date,open,high,low,close,volume"
                k_rs = bs.query_history_k_data_plus(code, fields, start, end, "d", "2")
                df = k_rs.get_data()

                if df is None or len(df) == 0:
                    skipped += 1
                    continue

                result = self.engine.calc(df)
                if result["composite"] == 0.0 and all(
                    v == 0 for v in result["scores"].values()
                ):
                    skipped += 1
                    continue

                vals = result.get("values", {})
                scores = result.get("scores", {})

                rows.append({
                    "code": code,
                    "name": name,
                    "ipo": ipo,
                    "last_date": result.get("last_date", ""),
                    "composite": result.get("composite", 0),
                    # 子得分
                    "Score_RSI": scores.get("RSI", 0),
                    "Score_MACD": scores.get("MACD", 0),
                    "Score_KDJ": scores.get("KDJ", 0),
                    "Score_BB": scores.get("BB", 0),
                    "Score_MA": scores.get("MA", 0),
                    "Score_VOL": scores.get("VOL", 0),
                    "Score_ATR": scores.get("ATR", 0),
                    "Score_OBV": scores.get("OBV", 0),
                    "Score_WR": scores.get("WR", 0),
                    "Score_CCI": scores.get("CCI", 0),
                    # 指标值
                    "RSI": vals.get("RSI", ""),
                    "MACD": vals.get("MACD", ""),
                    "MACD_Signal": vals.get("MACD_Signal", ""),
                    "MACD_Hist": vals.get("MACD_Hist", ""),
                    "K": vals.get("K", ""),
                    "D": vals.get("D", ""),
                    "J": vals.get("J", ""),
                    "BB_pct": vals.get("BB_pct", ""),
                    "MA5": vals.get("MA5", ""),
                    "MA20": vals.get("MA20", ""),
                    "VolRatio": vals.get("VolRatio", ""),
                    "ATR_pct": vals.get("ATR_pct", ""),
                    "OBV_Slope": vals.get("OBV_Slope", ""),
                    "WR": vals.get("WR", ""),
                    "CCI": vals.get("CCI", ""),
                })

            except Exception:
                skipped += 1
                continue

        bs.logout()

        if not rows:
            return None, skipped

        df_result = pd.DataFrame(rows).sort_values("composite", ascending=False)
        df_result = df_result.reset_index(drop=True)
        return df_result, skipped

    @staticmethod
    def save_csv(df: pd.DataFrame) -> str:
        today = datetime.now().strftime("%Y-%m-%d_%H%M")
        out_path = os.path.join(PREDS_DIR, f"TechScore_{today}.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path

    @staticmethod
    def refresh_quotes(df: pd.DataFrame) -> pd.DataFrame:
        """使用 easyquotation 刷新实时行情"""
        try:
            import easyquotation
            quotation = easyquotation.use("sina")
        except ImportError:
            st.warning("未安装 easyquotation，无法刷新实时行情。")
            return df

        df = df.copy()
        if "curr_price" not in df.columns:
            df["curr_price"] = 0.0
        if "curr_pct" not in df.columns:
            df["curr_pct"] = 0.0

        code_list = df["code"].tolist()
        num_list = [c[3:] for c in code_list]  # 去掉 "sh." / "sz." 前缀
        code_map = dict(zip(num_list, range(len(code_list))))

        try:
            for i in range(0, len(num_list), 800):
                sub = num_list[i:i + 800]
                data = quotation.real(sub)
                for code_num, info in data.items():
                    if code_num in code_map:
                        idx = code_map[code_num]
                        now_price = float(info.get("now", 0))
                        last_close = float(info.get("close", 0))
                        df.at[idx, "curr_price"] = now_price
                        if last_close > 0:
                            df.at[idx, "curr_pct"] = round(
                                (now_price - last_close) / last_close * 100, 2
                            )
        except Exception as e:
            st.warning(f"刷新行情时出错: {e}")

        return df


# =============================================================================
# 样式辅助函数
# =============================================================================
def color_composite(val):
    """综合分着色"""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    if v >= 75:
        return "background-color: #FFCDD2; color: #B71C1C; font-weight: bold"
    elif v >= 60:
        return "background-color: #FFE0B2; color: #E65100; font-weight: bold"
    elif v < 45:
        return "background-color: #E8F5E9; color: #2E7D32"
    return ""


def color_sub_score(val):
    """子得分着色"""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    if v >= 8.5:
        return "background-color: #FFCDD2; color: #B71C1C; font-weight: bold"
    return ""


def color_pct(val):
    """涨跌幅着色"""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    if v > 0:
        return "color: red; font-weight: bold"
    elif v < 0:
        return "color: green; font-weight: bold"
    return ""


def style_dataframe(df_display):
    """对展示用 DataFrame 应用条件格式"""
    score_cols = [c for c in df_display.columns if c.endswith("得分")]
    styled = df_display.style

    if "综合分" in df_display.columns:
        styled = styled.map(color_composite, subset=["综合分"])

    valid_score_cols = [c for c in score_cols if c in df_display.columns]
    if valid_score_cols:
        styled = styled.map(color_sub_score, subset=valid_score_cols)

    if "涨幅%" in df_display.columns:
        styled = styled.map(color_pct, subset=["涨幅%"])

    styled = styled.format(precision=2, na_rep="-")
    return styled


# =============================================================================
# 指标说明文本
# =============================================================================
INDICATOR_HELP = """
### 10 个技术指标评分说明
> 各指标子得分范围 **0~10**，综合得分范围 **0~100**

| 指标 | 权重 | 说明 |
|------|------|------|
| **MACD(12,26,9)** | 15% | 指数平滑异同移动平均 |
| **RSI(14)** | 12% | 相对强弱指标 |
| **KDJ(9,3,3)** | 12% | 随机指标 |
| **MA Cross(5/20)** | 12% | 均线交叉状态 |
| **Bollinger %B(20)** | 10% | 布林带位置百分比 |
| **量比(20日)** | 10% | 当日成交量/近20日均量 |
| **OBV Trend(5)** | 10% | 能量潮5日趋势斜率 |
| **ATR%(14)** | 7% | 真实波幅百分比 |
| **Williams %R(14)** | 6% | 威廉指标 |
| **CCI(14)** | 6% | 顺势指标 |

---

#### 评分规则详情

**RSI(14)** — 越低越看多  
- ≤20 → 10分 | ≤30 → 9分 | ≤40 → 7.5分 | ≤50 → 6分 | ≤60 → 5分 | ≤70 → 3.5分 | ≤80 → 2分 | >80 → 1分

**MACD(12,26,9)**  
- MACD>Signal 且 >0 → 9分 | MACD>Signal 且 ≤0 → 7分 | MACD≤Signal 且 >0 → 4分 | 其余 → 1.5分  
- 柱线扩大 +0.5，收缩 -0.5

**KDJ(9,3,3)**  
- K<20且D<20 → 9.5 | K<30且金叉 → 8.5 | K>80且D>80 → 1.5 | K>70且死叉 → 2  
- J<0 额外+1，J>100 额外-1

**Bollinger %B(20, 2σ)**  
- <0 → 9.5 | <0.15 → 8.5 | <0.35 → 7 | <0.65 → 5.5 | <0.85 → 3.5 | <1.0 → 2 | ≥1.0 → 1

**MA Cross(5/20)**  
- 金叉 → 10 | 死叉 → 1 | 5>20且偏离≤5% → 7.5 | 偏离>5% → 6 | 5<20 → 3

**量比(20日)**  
- 2.5~4.0 → 9 | >4.0 → 8.5 | 1.8~2.5 → 8 | 1.2~1.8 → 6.5 | 0.8~1.2 → 5 | 0.5~0.8 → 3 | <0.5 → 1.5

**ATR%(14)**  
- 1~2% → 8 | 0.5~1%/2~3% → 6.5 | 0.3~0.5%/3~5% → 4.5 | >5% → 2 | <0.3% → 3.5

**OBV Trend(5日斜率)**  
- >0.05 → 9 | >0.02 → 7.5 | >0 → 6 | >-0.02 → 4.5 | >-0.05 → 3 | ≤-0.05 → 1.5

**Williams %R(14)**  
- ≤-90 → 9.5 | ≤-80 → 8 | ≤-50 → 5.5 | ≤-20 → 3.5 | >-20 → 1.5

**CCI(14)**  
- <-200 → 9.5 | <-100 → 8 | <0 → 6 | <100 → 4.5 | <200 → 2.5 | ≥200 → 1
"""


# =============================================================================
# Streamlit 主界面
# =============================================================================
def main():
    # ---- 标题 ----
    st.title("📊 TechScore 技术指标综合评分系统")
    st.caption("基于 10 个技术指标的 A 股综合评分 · v1.3 Streamlit Edition")

    # ---- 初始化 session_state ----
    if "df_result" not in st.session_state:
        st.session_state.df_result = None
    if "status_msg" not in st.session_state:
        st.session_state.status_msg = ""

    dm = DataManager()

    # ==================================================================
    # 侧边栏
    # ==================================================================
    with st.sidebar:
        st.header("⚙️ 操作面板")

        # ---- Tab 1: 更新评分 ----
        tab_update, tab_load, tab_help = st.tabs(
            ["🔄 更新评分", "📂 加载/导出", "📖 指标说明"]
        )

        with tab_update:
            st.subheader("股票池选择")
            mode = st.radio(
                "模式", ["股票池模式", "单只股票"],
                horizontal=True, label_visibility="collapsed",
            )

            scopes = []
            if mode == "股票池模式":
                c_test = st.checkbox("测试 (10只)", value=False)
                if c_test:
                    scopes = ["test10"]
                else:
                    c_hs300 = st.checkbox("沪深300", value=True)
                    c_zz500 = st.checkbox("中证500", value=False)
                    c_all = st.checkbox("全部A股", value=False)
                    if c_hs300: scopes.append("hs300")
                    if c_zz500: scopes.append("zz500")
                    if c_all: scopes.append("all")
            else:
                single_code = st.text_input("股票代码", value="sz.002309")
                scopes = [f"single:{single_code}"]

            hist_days = st.slider(
                "历史K线天数（日历天）", 80, 600, 150,
                help="建议 ≥ 150，确保有足够有效交易日"
            )

            if st.button("🚀 开始更新行情 & 评分", type="primary", use_container_width=True):
                if not scopes:
                    st.warning("请至少选择一个股票池！")
                else:
                    end = datetime.now().strftime("%Y-%m-%d")
                    start = (datetime.now() - timedelta(days=hist_days)).strftime("%Y-%m-%d")

                    with st.spinner("正在获取股票列表…"):
                        codes = dm.build_code_list(scopes, end)

                    if not codes:
                        st.error("未获取到任何股票代码，请检查网络或配置。")
                    else:
                        st.info(f"共 {len(codes)} 只股票，开始下载K线并计算…")
                        progress_bar = st.progress(0, text="准备中…")
                        status_text = st.empty()

                        df_result, skipped = dm.fetch_and_score(
                            codes, start, end, progress_bar, status_text
                        )

                        if df_result is not None and len(df_result) > 0:
                            saved_path = dm.save_csv(df_result)
                            st.session_state.df_result = df_result
                            st.session_state.status_msg = (
                                f"✅ 完成: 分析 {len(df_result)} 只"
                                f"（跳过 {skipped} 只数据不足）\n"
                                f"已保存: {os.path.basename(saved_path)}"
                            )
                            progress_bar.progress(1.0, text="完成！")
                            st.success(st.session_state.status_msg)
                            st.rerun()
                        else:
                            st.error(
                                f"无有效数据（跳过 {skipped} 只）。"
                                f"请增大历史天数后重试。"
                            )

        with tab_load:
            st.subheader("加载历史结果")

            # 列出已有 CSV
            csv_files = sorted(
                glob.glob(os.path.join(PREDS_DIR, "TechScore_*.csv")),
                reverse=True,
            )
            if csv_files:
                file_names = [os.path.basename(f) for f in csv_files]
                selected = st.selectbox("选择历史文件", file_names)
                if st.button("📥 加载所选文件", use_container_width=True):
                    idx = file_names.index(selected)
                    df_loaded = pd.read_csv(csv_files[idx])
                    st.session_state.df_result = df_loaded
                    st.session_state.status_msg = (
                        f"✅ 已加载 {len(df_loaded)} 只股票 — {selected}"
                    )
                    st.success(st.session_state.status_msg)
                    st.rerun()
            else:
                st.info("暂无历史结果文件。")

            st.divider()

            # 上传 CSV
            st.subheader("上传 CSV")
            uploaded = st.file_uploader("上传 TechScore CSV", type="csv")
            if uploaded is not None:
                df_up = pd.read_csv(uploaded)
                st.session_state.df_result = df_up
                st.session_state.status_msg = f"✅ 已上传并加载 {len(df_up)} 只股票"
                st.success(st.session_state.status_msg)

            st.divider()

            # 导出下载
            if st.session_state.df_result is not None:
                st.subheader("导出当前结果")
                csv_data = st.session_state.df_result.to_csv(
                    index=False, encoding="utf-8-sig"
                )
                st.download_button(
                    "💾 下载 CSV",
                    data=csv_data,
                    file_name=f"TechScore_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with tab_help:
            st.markdown(INDICATOR_HELP)

    # ==================================================================
    # 主区域
    # ==================================================================

    # 顶部操作栏
    col1, col2, col3, col4 = st.columns([2, 2, 2, 6])

    with col1:
        btn_refresh_quote = st.button("🔃 刷新实时行情", use_container_width=True)
    with col2:
        if st.session_state.df_result is not None:
            st.metric("股票数量", len(st.session_state.df_result))
    with col3:
        if st.session_state.df_result is not None:
            avg_score = st.session_state.df_result["composite"].mean()
            st.metric("平均综合分", f"{avg_score:.1f}")

    if btn_refresh_quote:
        if st.session_state.df_result is not None:
            with st.spinner("正在刷新实时行情…"):
                st.session_state.df_result = dm.refresh_quotes(
                    st.session_state.df_result
                )
            st.success("✅ 行情刷新完成")
            st.rerun()
        else:
            st.warning("请先更新评分或加载历史结果。")

    # 状态消息
    if st.session_state.status_msg:
        st.info(st.session_state.status_msg)

    # ---- 主表格 ----
    if st.session_state.df_result is not None:
        df = st.session_state.df_result.copy()

        # -- 筛选区 --
        st.subheader("🔍 筛选 & 排序")
        fc1, fc2, fc3, fc4 = st.columns(4)

        with fc1:
            min_score = st.number_input("综合分 ≥", 0.0, 100.0, 0.0, 5.0)
        with fc2:
            max_score = st.number_input("综合分 ≤", 0.0, 100.0, 100.0, 5.0)
        with fc3:
            search_code = st.text_input("搜索代码/名称", "")
        with fc4:
            sort_col = st.selectbox(
                "排序依据",
                ["综合分(降序)", "综合分(升序)", "代码(升序)"],
            )

        # 应用筛选
        mask = (df["composite"] >= min_score) & (df["composite"] <= max_score)
        if search_code:
            mask = mask & (
                df["code"].str.contains(search_code, case=False, na=False)
                | df["name"].str.contains(search_code, case=False, na=False)
            )
        df_filtered = df[mask].copy()

        # 排序
        if sort_col == "综合分(降序)":
            df_filtered = df_filtered.sort_values("composite", ascending=False)
        elif sort_col == "综合分(升序)":
            df_filtered = df_filtered.sort_values("composite", ascending=True)
        else:
            df_filtered = df_filtered.sort_values("code", ascending=True)

        df_filtered = df_filtered.reset_index(drop=True)

        st.caption(f"共 {len(df_filtered)} 只股票（筛选后）")

        # 构建显示用 DataFrame
        display_cols = {
            "code": "代码",
            "name": "名称",
            "last_date": "行情截止日",
        }

        # 实时行情列（如果有）
        has_quote = "curr_price" in df_filtered.columns
        if has_quote:
            display_cols["curr_price"] = "现价"
            display_cols["curr_pct"] = "涨幅%"

        display_cols.update({
            "composite": "综合分",
            "Score_RSI": "RSI得分",
            "Score_MACD": "MACD得分",
            "Score_KDJ": "KDJ得分",
            "Score_BB": "BB得分",
            "Score_MA": "MA得分",
            "Score_VOL": "量比得分",
            "Score_ATR": "ATR得分",
            "Score_OBV": "OBV得分",
            "Score_WR": "WR得分",
            "Score_CCI": "CCI得分",
            "RSI": "RSI值",
            "MACD_Hist": "MACD柱",
            "K": "K",
            "D": "D",
            "J": "J",
            "BB_pct": "BB%",
            "MA5": "MA5",
            "MA20": "MA20",
            "VolRatio": "量比",
            "ATR_pct": "ATR%",
            "OBV_Slope": "OBV斜率",
            "WR": "WR",
            "CCI": "CCI",
        })

        # 只保留实际存在的列
        valid_cols = [c for c in display_cols if c in df_filtered.columns]
        df_display = df_filtered[valid_cols].rename(
            columns={c: display_cols[c] for c in valid_cols}
        )

        # 样式渲染
        styled = style_dataframe(df_display)

        st.dataframe(
            styled,
            use_container_width=True,
            height=min(700, 45 + 35 * len(df_display)),
            hide_index=True,
        )

        # ---- 分布图 ----
        st.subheader("📈 综合分分布")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # 分数区间统计
            bins = [0, 30, 45, 60, 75, 100]
            labels = ["0-30\n(弱)", "30-45\n(偏弱)", "45-60\n(中性)", "60-75\n(偏强)", "75-100\n(强)"]
            df_filtered["score_bin"] = pd.cut(
                df_filtered["composite"], bins=bins, labels=labels, right=True
            )
            bin_counts = df_filtered["score_bin"].value_counts().sort_index()
            st.bar_chart(bin_counts, use_container_width=True)

        with chart_col2:
            # 前10名
            top10 = df_filtered.nlargest(10, "composite")[["code", "name", "composite"]]
            top10["label"] = top10["name"] + " (" + top10["code"] + ")"
            st.dataframe(
                top10[["label", "composite"]].rename(
                    columns={"label": "股票", "composite": "综合分"}
                ),
                use_container_width=True,
                hide_index=True,
            )

    else:
        # 无数据时的引导页
        st.markdown("---")
        st.markdown(
            """
            ### 👈 开始使用

            1. 点击左侧 **🔄 更新评分** 选项卡
            2. 选择股票池和历史天数
            3. 点击 **🚀 开始更新行情 & 评分**

            或者点击 **📂 加载/导出** 选项卡加载以往保存的结果。

            ---

            #### ⚠️ 免责声明
            本软件仅供学习与研究使用，**不构成任何投资建议**。  
            使用者需自行承担一切投资决策及风险。
            """
        )

    # 页脚
    st.markdown("---")
    st.caption(
        "TechScore Stock Analyzer v1.3 · "
        "[GitHub](https://github.com/tsiwt/openstocktechscore) · "
        "License: GPL-3.0 · "
        "本软件仅供研究使用，不构成投资建议"
    )


# =============================================================================
# 启动
# =============================================================================
if __name__ == "__main__":
    main()

