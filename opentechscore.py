# =============================================================================
# TechScore Stock Analyzer
# Copyright (C) 2026  Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Project: https://github.com/tsiwt/openstocktechscore
# Version: 1.3.0
# Description:
#   A daily technical-indicator based stock scoring system.
#   Uses baostock for historical data, calculates 10 common indicators
#   (RSI, MACD, KDJ, Bollinger Bands, MA Cross, Volume Ratio, ATR,
#    OBV Trend, Williams %R, CCI) and generates a composite score 0-100.
#
# Disclaimer:
#   This software is for research and educational purposes only.
#   It does not constitute any investment advice or recommendation.
#   All investment decisions and associated risks are solely the
#   responsibility of the user.
# =============================================================================

import sys
import os
import pandas as pd
import numpy as np
import baostock as bs
import glob
import webbrowser
from datetime import datetime, timedelta

try:
    import easyquotation
except ImportError:
    print("错误: 缺少必要库。请运行: pip install pandas numpy baostock PyQt5 easyquotation")
    sys.exit(1)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar,
    QFileDialog, QMessageBox, QDialog, QCheckBox,
    QDialogButtonBox, QGroupBox, QSpinBox, QFrame,
    QAbstractItemView, QLineEdit, QRadioButton,
    QButtonGroup, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor

# =============================================================================
# 全局配置
# =============================================================================
SYSTEM_NAME = "TechScore_v1.3"
SCRIPT_DIR  = os.path.dirname(os.path.realpath(sys.argv[0]))
DATA_ROOT   = os.path.join(SCRIPT_DIR, "TechScore_Data")
PREDS_DIR   = os.path.join(DATA_ROOT, "Predictions")

for _d in [DATA_ROOT, PREDS_DIR]:
    if not os.path.exists(_d):
        os.makedirs(_d)

# 每只股票最少需要的有效 K 线条数
# MACD 最长回看 26 日，50 条足够所有指标稳定计算
MIN_BARS = 50


# =============================================================================
# 技术指标计算引擎
# =============================================================================
class TechnicalIndicatorEngine:
    """
    计算10个常见技术指标并输出综合评分(0~100)。

    指标列表:
      1. RSI(14)           - 相对强弱指标
      2. MACD(12,26,9)     - 指数平滑异同移动平均
      3. KDJ(9,3,3)        - 随机指标
      4. Bollinger %B(20)  - 布林带位置百分比
      5. MA Cross(5/20)    - 均线交叉状态
      6. Volume Ratio(20)  - 量比
      7. ATR%(14)          - 真实波幅百分比（归一化）
      8. OBV Trend(5)      - 能量潮5日趋势斜率
      9. Williams %R(14)   - 威廉指标
      10. CCI(14)          - 顺势指标

    综合评分 = 各指标子得分(0~10) 加权均值 × 10，映射至 0~100
    """

    # 各指标权重，合计 = 1.0
    WEIGHTS = {
        'RSI':  0.12,
        'MACD': 0.15,
        'KDJ':  0.12,
        'BB':   0.10,
        'MA':   0.12,
        'VOL':  0.10,
        'ATR':  0.07,
        'OBV':  0.10,
        'WR':   0.06,
        'CCI':  0.06,
    }

    def calc(self, df: pd.DataFrame) -> dict:
        """
        输入 df 必须包含列: date, open, high, low, close, volume
        至少 MIN_BARS 行（有效收盘价行），按日期升序排列。
        返回最新一行的各指标值、子得分及综合得分。
        """
        df = df.copy().reset_index(drop=True)

        # --- 数据清洗 ---
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 仅依赖 close > 0 过滤；volume 缺失时填 0，不删整行
        # baostock 对停牌日有时返回空字符串，不能用 dropna 一刀切
        df = df[df['close'] > 0].copy()
        df['volume'] = df['volume'].fillna(0)
        df = df.reset_index(drop=True)

        if len(df) < MIN_BARS:
            return self._empty_result()

        scores = {}
        values = {}

        # ★ 记录本次计算用到的最后一个交易日日期
        last_date = str(df['date'].iloc[-1]) if 'date' in df.columns else ""

        rsi_val, rsi_score                         = self._rsi(df['close'], 14)
        values['RSI']         = rsi_val
        scores['RSI']         = rsi_score

        macd_val, signal_val, hist_val, macd_score = self._macd(df['close'])
        values['MACD']        = macd_val
        values['MACD_Signal'] = signal_val
        values['MACD_Hist']   = hist_val
        scores['MACD']        = macd_score

        k_val, d_val, j_val, kdj_score             = self._kdj(df['high'], df['low'], df['close'])
        values['K']           = k_val
        values['D']           = d_val
        values['J']           = j_val
        scores['KDJ']         = kdj_score

        bb_val, bb_score                           = self._bollinger(df['close'], 20, 2)
        values['BB_pct']      = bb_val
        scores['BB']          = bb_score

        ma5_val, ma20_val, ma_score                = self._ma_cross(df['close'], 5, 20)
        values['MA5']         = ma5_val
        values['MA20']        = ma20_val
        scores['MA']          = ma_score

        vr_val, vr_score                           = self._volume_ratio(df['volume'], 20)
        values['VolRatio']    = vr_val
        scores['VOL']         = vr_score

        atr_val, atr_score                         = self._atr_pct(df['high'], df['low'], df['close'], 14)
        values['ATR_pct']     = atr_val
        scores['ATR']         = atr_score

        obv_slope, obv_score                       = self._obv_trend(df['close'], df['volume'], 5)
        values['OBV_Slope']   = obv_slope
        scores['OBV']         = obv_score

        wr_val, wr_score                           = self._williams_r(df['high'], df['low'], df['close'], 14)
        values['WR']          = wr_val
        scores['WR']          = wr_score

        cci_val, cci_score                         = self._cci(df['high'], df['low'], df['close'], 14)
        values['CCI']         = cci_val
        scores['CCI']         = cci_score

        total     = sum(scores[k] * self.WEIGHTS[k] for k in scores)
        composite = round(total * 10, 2)

        return {
            'values':    values,
            'scores':    scores,
            'composite': composite,
            'last_date': last_date,   # ★ 新增：行情截止日
        }

    def _empty_result(self):
        return {
            'values':    {},
            'scores':    {k: 0 for k in self.WEIGHTS},
            'composite': 0.0,
            'last_date': '',          # ★ 新增
        }

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def _rsi(self, close: pd.Series, period: int):
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs       = avg_gain / (avg_loss + 1e-9)
        rsi      = 100 - 100 / (1 + rs)
        val      = round(rsi.iloc[-1], 2)

        if   val <= 20: score = 10.0
        elif val <= 30: score =  9.0
        elif val <= 40: score =  7.5
        elif val <= 50: score =  6.0
        elif val <= 60: score =  5.0
        elif val <= 70: score =  3.5
        elif val <= 80: score =  2.0
        else:           score =  1.0
        return val, score

    def _macd(self, close: pd.Series):
        ema12  = self._ema(close, 12)
        ema26  = self._ema(close, 26)
        macd   = ema12 - ema26
        signal = self._ema(macd, 9)
        hist   = macd - signal

        macd_v = round(macd.iloc[-1],   4)
        sig_v  = round(signal.iloc[-1], 4)
        hist_v = round(hist.iloc[-1],   4)

        if   macd_v > sig_v and macd_v > 0:  score = 9.0
        elif macd_v > sig_v and macd_v <= 0: score = 7.0
        elif macd_v <= sig_v and macd_v > 0: score = 4.0
        else:                                score = 1.5

        if len(hist) >= 2:
            if hist.iloc[-1] > hist.iloc[-2]: score = min(10.0, score + 0.5)
            else:                             score = max( 0.0, score - 0.5)

        return macd_v, sig_v, hist_v, round(score, 2)

    def _kdj(self, high: pd.Series, low: pd.Series, close: pd.Series):
        period = 9
        low_n  = low.rolling(period).min()
        high_n = high.rolling(period).max()
        rsv    = (close - low_n) / (high_n - low_n + 1e-9) * 100
        K      = rsv.ewm(com=2, adjust=False).mean()
        D      = K.ewm(com=2, adjust=False).mean()
        J      = 3 * K - 2 * D

        k_v = round(K.iloc[-1], 2)
        d_v = round(D.iloc[-1], 2)
        j_v = round(J.iloc[-1], 2)

        if   k_v < 20 and d_v < 20:
            score = 9.5
        elif k_v < 30 and K.iloc[-1] > K.iloc[-2] and K.iloc[-2] < D.iloc[-2]:
            score = 8.5
        elif k_v > 80 and d_v > 80:
            score = 1.5
        elif k_v > 70 and K.iloc[-1] < K.iloc[-2] and K.iloc[-2] > D.iloc[-2]:
            score = 2.0
        elif k_v > d_v:
            score = 6.5
        else:
            score = 4.0

        if   j_v < 0:   score = min(10.0, score + 1.0)
        elif j_v > 100: score = max( 0.0, score - 1.0)

        return k_v, d_v, j_v, round(score, 2)

    def _bollinger(self, close: pd.Series, period: int, std_mult: float):
        ma    = close.rolling(period).mean()
        std   = close.rolling(period).std()
        upper = ma + std_mult * std
        lower = ma - std_mult * std

        lc = close.iloc[-1]
        lu = upper.iloc[-1]
        ll = lower.iloc[-1]
        bw = lu - ll

        bb_pct = 0.5 if bw < 1e-9 else (lc - ll) / bw
        val    = round(bb_pct, 4)

        if   val < 0:    score = 9.5
        elif val < 0.15: score = 8.5
        elif val < 0.35: score = 7.0
        elif val < 0.65: score = 5.5
        elif val < 0.85: score = 3.5
        elif val < 1.0:  score = 2.0
        else:            score = 1.0

        return val, round(score, 2)

    def _ma_cross(self, close: pd.Series, fast: int, slow: int):
        ma_fast = close.rolling(fast).mean()
        ma_slow = close.rolling(slow).mean()

        ma5_v  = round(ma_fast.iloc[-1], 4)
        ma20_v = round(ma_slow.iloc[-1], 4)

        prev_diff = ma_fast.iloc[-2] - ma_slow.iloc[-2]
        curr_diff = ma_fast.iloc[-1] - ma_slow.iloc[-1]

        if   prev_diff < 0 and curr_diff > 0: score = 10.0
        elif prev_diff > 0 and curr_diff < 0: score =  1.0
        elif curr_diff > 0:
            gap_pct = curr_diff / (ma_slow.iloc[-1] + 1e-9)
            score = 6.0 if gap_pct > 0.05 else 7.5
        else:
            score = 3.0

        return ma5_v, ma20_v, round(score, 2)

    @staticmethod
    def _volume_ratio(volume: pd.Series, period: int):
        avg_vol  = volume.iloc[-period - 1:-1].mean()
        curr_vol = volume.iloc[-1]
        vr       = 1.0 if avg_vol < 1e-9 else curr_vol / avg_vol
        val      = round(vr, 3)

        if   vr > 4.0: score = 8.5
        elif vr > 2.5: score = 9.0
        elif vr > 1.8: score = 8.0
        elif vr > 1.2: score = 6.5
        elif vr > 0.8: score = 5.0
        elif vr > 0.5: score = 3.0
        else:          score = 1.5

        return val, round(score, 2)

    def _atr_pct(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int):
        prev_close = close.shift(1)
        tr  = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)
        atr     = tr.rolling(period).mean()
        atr_pct = atr / (close + 1e-9) * 100
        val     = round(atr_pct.iloc[-1], 3)

        if   1.0 <= val <= 2.0:                      score = 8.0
        elif 0.5 <= val < 1.0 or 2.0 < val <= 3.0:  score = 6.5
        elif 0.3 <= val < 0.5 or 3.0 < val <= 5.0:  score = 4.5
        elif val > 5.0:                              score = 2.0
        else:                                        score = 3.5

        return val, round(score, 2)

    @staticmethod
    def _obv_trend(close: pd.Series, volume: pd.Series, period: int):
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv        = (direction * volume).cumsum()
        recent     = obv.iloc[-period:].values
        if len(recent) < 2:
            return 0.0, 5.0
        x          = np.arange(len(recent))
        slope      = np.polyfit(x, recent, 1)[0]
        mean_abs   = np.mean(np.abs(recent)) + 1e-9
        norm_slope = slope / mean_abs

        if   norm_slope >  0.05: score = 9.0
        elif norm_slope >  0.02: score = 7.5
        elif norm_slope >  0:    score = 6.0
        elif norm_slope > -0.02: score = 4.5
        elif norm_slope > -0.05: score = 3.0
        else:                    score = 1.5

        return round(float(norm_slope), 5), round(score, 2)

    @staticmethod
    def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int):
        high_n = high.rolling(period).max()
        low_n  = low.rolling(period).min()
        wr     = (high_n - close) / (high_n - low_n + 1e-9) * (-100)
        val    = round(wr.iloc[-1], 2)

        if   val <= -90: score = 9.5
        elif val <= -80: score = 8.0
        elif val <= -50: score = 5.5
        elif val <= -20: score = 3.5
        else:            score = 1.5

        return val, round(score, 2)

    @staticmethod
    def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int):
        typical = (high + low + close) / 3
        ma      = typical.rolling(period).mean()
        md      = typical.rolling(period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        cci = (typical - ma) / (0.015 * md + 1e-9)
        val = round(cci.iloc[-1], 2)

        if   val < -200: score = 9.5
        elif val < -100: score = 8.0
        elif val <    0: score = 6.0
        elif val <  100: score = 4.5
        elif val <  200: score = 2.5
        else:            score = 1.0

        return val, round(score, 2)


# =============================================================================
# UI 辅助类
# =============================================================================
class NumericItem(QTableWidgetItem):
    def __lt__(self, other):
        v1 = self.data(Qt.UserRole)
        v2 = other.data(Qt.UserRole)
        try:
            return float(v1) < float(v2)
        except Exception:
            return super().__lt__(other)


class ScopeSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("更新行情 & 评分配置")
        self.resize(380, 340)
        lay = QVBoxLayout(self)

        # --- 股票池 ---
        g1   = QGroupBox("1. 目标股票池")
        v1   = QVBoxLayout()
        self.bg = QButtonGroup(self)

        self.r_pool = QRadioButton("股票池模式")
        self.r_pool.setChecked(True)
        self.bg.addButton(self.r_pool)
        v1.addWidget(self.r_pool)

        self.f_pool  = QFrame()
        l_pool       = QVBoxLayout(self.f_pool)
        self.c_test  = QCheckBox("测试 (10只)")
        self.c_hs300 = QCheckBox("沪深300")
        self.c_zz500 = QCheckBox("中证500")
        self.c_all   = QCheckBox("全部A股")
        self.c_hs300.setChecked(True)
        self.all_checks = [self.c_test, self.c_hs300, self.c_zz500, self.c_all]
        for c in self.all_checks:
            l_pool.addWidget(c)
        v1.addWidget(self.f_pool)

        self.r_one = QRadioButton("单只股票模式")
        self.bg.addButton(self.r_one)
        v1.addWidget(self.r_one)
        self.e_code = QLineEdit("sz.002309")
        self.e_code.setEnabled(False)
        v1.addWidget(self.e_code)
        g1.setLayout(v1)
        lay.addWidget(g1)

        # --- 天数 ---
        g2 = QGroupBox("2. 历史K线天数（含周末节假日；建议 ≥ 150）")
        h2 = QHBoxLayout()
        self.s_days = QSpinBox()
        self.s_days.setRange(80, 600)
        self.s_days.setValue(150)
        self.s_days.setSuffix(" 日历天")
        h2.addWidget(QLabel("下载:"))
        h2.addWidget(self.s_days)
        g2.setLayout(h2)
        lay.addWidget(g2)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self.r_pool.toggled.connect(
            lambda: (self.f_pool.setEnabled(True), self.e_code.setEnabled(False))
        )
        self.r_one.toggled.connect(
            lambda: (self.f_pool.setEnabled(False), self.e_code.setEnabled(True))
        )
        self.c_test.toggled.connect(self._on_test)

    def _on_test(self, chk):
        if chk:
            for c in self.all_checks:
                if c != self.c_test:
                    c.setChecked(False)
                    c.setEnabled(False)
        else:
            for c in self.all_checks:
                c.setEnabled(True)

    def get_data(self):
        days = self.s_days.value()
        if self.r_one.isChecked():
            return ['single:' + self.e_code.text()], days
        if self.c_test.isChecked():
            return ['test10'], days
        s = []
        if self.c_hs300.isChecked(): s.append('hs300')
        if self.c_zz500.isChecked(): s.append('zz500')
        if self.c_all.isChecked():   s.append('all')
        return s, days


# =============================================================================
# 股票上下文
# =============================================================================
class Context:
    def __init__(self, code, name, ipo):
        self.code         = code
        self.name         = name
        self.ipo          = ipo
        self.curr_price   = "-"
        self.curr_pct     = "-"
        self.score_result = {}
        self.last_date    = ""   # ★ 新增：行情截止日（K线最后一个交易日）


# =============================================================================
# 数据管理
# =============================================================================
class DataManager:
    def __init__(self):
        self.engine     = TechnicalIndicatorEngine()
        self.ctxs: list = []
        self.quotation  = easyquotation.use('sina')

    # ------------------------------------------------------------------
    def _build_code_list(self, scopes, end_date):
        codes = set()
        if scopes[0].startswith('single:'):
            codes.add(scopes[0].split(':')[1])
            return codes

        if 'test10' in scopes:
            rs  = bs.query_hs300_stocks()
            cnt = 0
            while rs.next() and cnt < 10:
                codes.add(rs.get_row_data()[1])
                cnt += 1
            return codes

        if 'hs300' in scopes:
            rs = bs.query_hs300_stocks()
            while rs.next():
                codes.add(rs.get_row_data()[1])
        if 'zz500' in scopes:
            rs = bs.query_zz500_stocks()
            while rs.next():
                codes.add(rs.get_row_data()[1])
        if 'all' in scopes:
            rs = bs.query_all_stock(day=end_date)
            while rs.next():
                c = rs.get_row_data()[0]
                if c.startswith(('sh.', 'sz.')):
                    codes.add(c)
        return codes

    # ------------------------------------------------------------------
    def run_update_and_score(self, scopes, days, cb):
        """
        下载历史K线 → 清洗数据 → 计算技术指标评分 → 自动保存 CSV
        """
        bs.login()
        end   = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        codes   = self._build_code_list(scopes, end)
        self.ctxs = []
        clist   = sorted(list(codes))
        total   = len(clist)

        if total == 0:
            bs.logout()
            return "⚠️ 未获取到任何股票代码，请检查网络或股票池配置。"

        skipped = 0
        for i, code in enumerate(clist):
            if cb:
                cb(i, total, f"[{i+1}/{total}] 下载并计算: {code}")
            try:
                rs = bs.query_stock_basic(code=code)
                if not rs.data:
                    skipped += 1
                    continue
                info = rs.get_row_data()
                ipo  = info[2] if len(info) > 2 else ""
                name = info[1] if len(info) > 1 else code
                if not ipo:
                    skipped += 1
                    continue

                fields = "date,open,high,low,close,volume"
                k_rs   = bs.query_history_k_data_plus(
                    code, fields, start, end, "d", "2"
                )
                df = k_rs.get_data()

                if df is None or len(df) == 0:
                    skipped += 1
                    continue

                result = self.engine.calc(df)
                if result['composite'] == 0.0 and all(
                    v == 0 for v in result['scores'].values()
                ):
                    skipped += 1
                    continue

                ctx           = Context(code, name, ipo)
                ctx.score_result = result
                ctx.last_date    = result.get('last_date', '')  # ★ 写入截止日
                self.ctxs.append(ctx)

            except Exception:
                skipped += 1
                continue

        bs.logout()

        if not self.ctxs:
            return (
                f"⚠️ 无有效数据（跳过 {skipped} 只）：\n"
                f"所有股票有效交易日数不足 {MIN_BARS} 条，\n"
                f"请将历史天数设置为 150 以上后重试。"
            )

        saved_path = self._save_csv()
        count      = len(self.ctxs)
        return (
            f"✅ 完成: 共分析 {count} 只股票"
            f"（跳过 {skipped} 只数据不足）。\n"
            f"结果已保存至: {saved_path}"
        )

    # ------------------------------------------------------------------
    def _save_csv(self) -> str:
        """将当前评分结果保存为带时间戳的 CSV，返回保存路径。"""
        today = datetime.now().strftime("%Y-%m-%d_%H%M")
        rows  = []
        for ctx in self.ctxs:
            res    = ctx.score_result
            vals   = res.get('values',  {})
            scores = res.get('scores',  {})
            row = {
                'code':        ctx.code,
                'name':        ctx.name,
                'ipo':         ctx.ipo,
                'last_date':   ctx.last_date,          # ★ 新增列
                'composite':   res.get('composite', 0),
                # 指标值
                'RSI':         vals.get('RSI',        ''),
                'MACD':        vals.get('MACD',        ''),
                'MACD_Signal': vals.get('MACD_Signal', ''),
                'MACD_Hist':   vals.get('MACD_Hist',   ''),
                'K':           vals.get('K',           ''),
                'D':           vals.get('D',           ''),
                'J':           vals.get('J',           ''),
                'BB_pct':      vals.get('BB_pct',      ''),
                'MA5':         vals.get('MA5',         ''),
                'MA20':        vals.get('MA20',        ''),
                'VolRatio':    vals.get('VolRatio',    ''),
                'ATR_pct':     vals.get('ATR_pct',     ''),
                'OBV_Slope':   vals.get('OBV_Slope',   ''),
                'WR':          vals.get('WR',          ''),
                'CCI':         vals.get('CCI',         ''),
                # 子得分
                'Score_RSI':   scores.get('RSI',  0),
                'Score_MACD':  scores.get('MACD', 0),
                'Score_KDJ':   scores.get('KDJ',  0),
                'Score_BB':    scores.get('BB',   0),
                'Score_MA':    scores.get('MA',   0),
                'Score_VOL':   scores.get('VOL',  0),
                'Score_ATR':   scores.get('ATR',  0),
                'Score_OBV':   scores.get('OBV',  0),
                'Score_WR':    scores.get('WR',   0),
                'Score_CCI':   scores.get('CCI',  0),
            }
            rows.append(row)

        df       = pd.DataFrame(rows).sort_values('composite', ascending=False)
        out_path = os.path.join(PREDS_DIR, f"TechScore_{today}.csv")
        df.to_csv(out_path, index=False, encoding='utf-8-sig')
        return out_path

    # ------------------------------------------------------------------
    def save_csv_manual(self) -> str:
        """手动触发保存（表格已有数据时使用）"""
        if not self.ctxs:
            return "⚠️ 无数据，请先执行「更新行情 & 评分」。"
        path = self._save_csv()
        return f"✅ 已保存: {path}  ({len(self.ctxs)} 只)"

    # ------------------------------------------------------------------
    def load_from_csv(self, path: str) -> str:
        """从 CSV 重建 ctxs，加载后可刷新实时行情。"""
        try:
            df        = pd.read_csv(path)
            self.ctxs = []
            for _, row in df.iterrows():
                ctx = Context(
                    str(row.get('code', '')),
                    str(row.get('name', '')),
                    str(row.get('ipo',  ''))
                )
                ctx.last_date    = str(row.get('last_date', ''))   # ★ 读取截止日
                ctx.score_result = {
                    'composite': float(row.get('composite', 0)),
                    'last_date': str(row.get('last_date', '')),     # ★ 同步写入 result
                    'scores': {
                        'RSI':  float(row.get('Score_RSI',  0)),
                        'MACD': float(row.get('Score_MACD', 0)),
                        'KDJ':  float(row.get('Score_KDJ',  0)),
                        'BB':   float(row.get('Score_BB',   0)),
                        'MA':   float(row.get('Score_MA',   0)),
                        'VOL':  float(row.get('Score_VOL',  0)),
                        'ATR':  float(row.get('Score_ATR',  0)),
                        'OBV':  float(row.get('Score_OBV',  0)),
                        'WR':   float(row.get('Score_WR',   0)),
                        'CCI':  float(row.get('Score_CCI',  0)),
                    },
                    'values': {
                        'RSI':       row.get('RSI',        ''),
                        'MACD_Hist': row.get('MACD_Hist',  ''),
                        'K':         row.get('K',          ''),
                        'D':         row.get('D',          ''),
                        'J':         row.get('J',          ''),
                        'BB_pct':    row.get('BB_pct',     ''),
                        'MA5':       row.get('MA5',        ''),
                        'MA20':      row.get('MA20',       ''),
                        'VolRatio':  row.get('VolRatio',   ''),
                        'ATR_pct':   row.get('ATR_pct',    ''),
                        'OBV_Slope': row.get('OBV_Slope',  ''),
                        'WR':        row.get('WR',         ''),
                        'CCI':       row.get('CCI',        ''),
                    }
                }
                self.ctxs.append(ctx)
            return (
                f"✅ 已加载 {len(self.ctxs)} 只股票，"
                f"可点击「② 刷新实时行情」获取最新报价。"
            )
        except Exception as e:
            return f"❌ 加载失败: {e}"

    # ------------------------------------------------------------------
    def refresh_quotes(self):
        """刷新实时行情（更新评分后或加载历史 CSV 后均可使用）"""
        if not self.ctxs:
            return
        code_map = {c.code[3:]: c for c in self.ctxs}
        try:
            all_nums = list(code_map.keys())
            for i in range(0, len(all_nums), 800):
                sub  = all_nums[i:i + 800]
                data = self.quotation.real(sub)
                for code_num, info in data.items():
                    if code_num in code_map:
                        ctx            = code_map[code_num]
                        ctx.curr_price = info.get('now',   0)
                        last           = float(info.get('close', 0))
                        now            = float(info.get('now',   0))
                        if last > 0:
                            ctx.curr_pct = (now - last) / last * 100
        except Exception:
            pass


# =============================================================================
# 后台工作线程
# =============================================================================
class Worker(QThread):
    prog = pyqtSignal(int, int, str)
    done = pyqtSignal(str)

    def __init__(self, func, *args):
        super().__init__()
        self.f = func
        self.a = args

    def run(self):
        self.done.emit(self.f(*self.a, self.prog.emit))


# =============================================================================
# 主窗口
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(SYSTEM_NAME + " — 技术指标综合评分系统")
        self.resize(1750, 880)
        self.dm = DataManager()

        w   = QWidget()
        self.setCentralWidget(w)
        lay = QVBoxLayout(w)

        # 菜单
        menu = self.menuBar().addMenu("系统功能")
        menu.addAction("关于", self.show_about)

        # 顶部状态说明
        self.lbl_mode = QLabel(
            "就绪  |  点击「① 更新行情 & 评分」下载K线并自动计算指标、保存结果"
        )
        self.lbl_mode.setStyleSheet(
            "color:#1565C0; font-weight:bold; font-size:13px;"
        )
        lay.addWidget(self.lbl_mode)

        # 操作按钮行1
        h1 = QHBoxLayout()
        b_update = QPushButton("① 更新行情 & 评分")
        b_update.setToolTip("下载历史K线，计算10个技术指标综合评分，并自动保存结果 CSV")
        b_update.clicked.connect(self.do_update)

        b_quote = QPushButton("② 刷新实时行情")
        b_quote.setToolTip("更新现价与涨跌幅（更新评分后或加载历史结果后均可使用）")
        b_quote.clicked.connect(self.do_quote)

        b_save = QPushButton("③ 手动保存 CSV")
        b_save.setToolTip("将当前表格结果再次保存为带时间戳的 CSV")
        b_save.clicked.connect(self.do_save_csv)

        h1.addWidget(b_update)
        h1.addWidget(b_quote)
        h1.addWidget(b_save)
        h1.addStretch()
        lay.addLayout(h1)

        # 操作按钮行2
        h2 = QHBoxLayout()
        b_load_csv = QPushButton("④ 加载历史结果")
        b_load_csv.setToolTip("加载以往保存的 CSV，加载后可刷新实时行情")
        b_load_csv.clicked.connect(self.do_load_csv)

        b_open_dir = QPushButton("⑤ 打开结果目录")
        b_open_dir.setToolTip(f"打开 CSV 保存目录: {PREDS_DIR}")
        b_open_dir.clicked.connect(self.do_open_dir)

        b_help = QPushButton("⑥ 指标说明")
        b_help.clicked.connect(self.show_indicator_help)

        h2.addWidget(b_load_csv)
        h2.addWidget(b_open_dir)
        h2.addWidget(b_help)
        h2.addStretch()
        lay.addLayout(h2)

        # 状态 + 进度
        self.stat = QLabel("就绪")
        lay.addWidget(self.stat)
        self.pb = QProgressBar()
        lay.addWidget(self.pb)

        # 主表格
        self.tab = QTableWidget()
        self.tab.verticalHeader().setVisible(False)
        self.tab.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tab.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tab.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.tab.doubleClicked.connect(self.on_double_click)
        lay.addWidget(self.tab)

        # 启动时自动加载最新历史结果
        self._try_auto_load_latest()

    # ------------------------------------------------------------------
    def _try_auto_load_latest(self):
        files = sorted(glob.glob(os.path.join(PREDS_DIR, "TechScore_*.csv")))
        if files:
            latest = files[-1]
            msg    = self.dm.load_from_csv(latest)
            self.lbl_mode.setText(
                f"已自动加载最新结果: {os.path.basename(latest)}"
                f"  |  可刷新行情或重新更新评分"
            )
            self.stat.setText(msg)
            self.refresh_table()

    def on_double_click(self, index):
        """双击行 → 打开新浪K线页面"""
        row       = index.row()
        code_item = self.tab.item(row, 1)
        if code_item:
            code = code_item.text().replace('.', '')
            webbrowser.open(
                f"https://finance.sina.com.cn/realstock/company/{code}/nc.shtml"
            )

    # ------------------------------------------------------------------
    def do_update(self):
        d = ScopeSelectionDialog(self)
        if d.exec_():
            scopes, days = d.get_data()
            self.lbl_mode.setText("正在下载K线并计算指标，请稍候…")
            self.start_work(self.dm.run_update_and_score, scopes, days)

    def do_quote(self):
        if not self.dm.ctxs:
            QMessageBox.information(
                self, "提示",
                "请先执行「① 更新行情 & 评分」或「④ 加载历史结果」。"
            )
            return
        self.stat.setText("正在刷新实时行情…")
        QApplication.processEvents()
        self.dm.refresh_quotes()
        self.refresh_table()
        self.stat.setText("✅ 行情刷新完成")

    def do_save_csv(self):
        msg = self.dm.save_csv_manual()
        self.stat.setText(msg)
        QMessageBox.information(self, "保存结果", msg)

    def do_load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "加载历史结果 CSV", PREDS_DIR, "*.csv"
        )
        if path:
            msg = self.dm.load_from_csv(path)
            self.stat.setText(msg)
            self.lbl_mode.setText(
                f"已加载: {os.path.basename(path)}"
                f"  |  可点击「② 刷新实时行情」获取最新报价"
            )
            self.refresh_table()

    def do_open_dir(self):
        import subprocess, platform
        if platform.system() == "Windows":
            os.startfile(PREDS_DIR)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", PREDS_DIR])
        else:
            subprocess.Popen(["xdg-open", PREDS_DIR])

    def show_about(self):
        QMessageBox.about(
            self, "关于 " + SYSTEM_NAME,
            "<b>TechScore Stock Analyzer v1.3</b><br>"
            "基于10个技术指标的股票综合评分系统<br><br>"
            "Copyright © 2024 Contributors<br>"
            "License: GNU General Public License v3<br>"
            "GitHub: https://github.com/your-org/techscore-stock-analyzer<br><br>"
            "<i>本软件仅供学习与研究使用，不构成任何投资建议。<br>"
            "使用者需自行承担一切投资决策及风险。</i>"
        )

    def show_indicator_help(self):
        txt = (
            "<b>10个技术指标评分说明</b><br>"
            "<small>各指标子得分范围 0~10，综合得分范围 0~100</small><br><br>"

            "<b>1. RSI(14)</b> — 相对强弱指标<br>"
            "· RSI ≤ 20：得分 10　· RSI ≤ 30：得分 9<br>"
            "· RSI ≤ 40：得分 7.5　· RSI ≤ 50：得分 6<br>"
            "· RSI ≤ 60：得分 5　· RSI ≤ 70：得分 3.5<br>"
            "· RSI ≤ 80：得分 2　· RSI &gt; 80：得分 1<br><br>"

            "<b>2. MACD(12,26,9)</b> — 指数平滑异同移动平均<br>"
            "· MACD &gt; Signal 且 MACD &gt; 0：得分 9<br>"
            "· MACD &gt; Signal 且 MACD ≤ 0：得分 7<br>"
            "· MACD ≤ Signal 且 MACD &gt; 0：得分 4<br>"
            "· MACD ≤ Signal 且 MACD ≤ 0：得分 1.5<br>"
            "· 柱状线扩大额外 +0.5，收缩额外 -0.5<br><br>"

            "<b>3. KDJ(9,3,3)</b> — 随机指标<br>"
            "· K &lt; 20 且 D &lt; 20：得分 9.5<br>"
            "· K &lt; 30 且 K 上穿 D：得分 8.5<br>"
            "· K &gt; 80 且 D &gt; 80：得分 1.5<br>"
            "· K &gt; 70 且 K 下穿 D：得分 2<br>"
            "· K &gt; D（其余）：得分 6.5　· K ≤ D（其余）：得分 4<br>"
            "· J &lt; 0 额外 +1，J &gt; 100 额外 -1<br><br>"

            "<b>4. Bollinger %B(20, 2σ)</b> — 布林带位置<br>"
            "· %B &lt; 0（下轨以外）：得分 9.5<br>"
            "· %B &lt; 0.15：得分 8.5　· %B &lt; 0.35：得分 7<br>"
            "· %B &lt; 0.65（中轨附近）：得分 5.5<br>"
            "· %B &lt; 0.85：得分 3.5　· %B &lt; 1.0：得分 2<br>"
            "· %B ≥ 1.0（上轨以外）：得分 1<br><br>"

            "<b>5. MA Cross(5日/20日)</b> — 均线交叉<br>"
            "· 当日 5日上穿 20日：得分 10<br>"
            "· 当日 5日下穿 20日：得分 1<br>"
            "· 5日 &gt; 20日（偏离 ≤5%）：得分 7.5<br>"
            "· 5日 &gt; 20日（偏离 &gt;5%）：得分 6<br>"
            "· 5日 &lt; 20日：得分 3<br><br>"

            "<b>6. 量比(20日均量)</b> — 当日成交量 / 近20日均量<br>"
            "· 量比 2.5~4.0：得分 9　· 量比 &gt; 4.0：得分 8.5<br>"
            "· 量比 1.8~2.5：得分 8　· 量比 1.2~1.8：得分 6.5<br>"
            "· 量比 0.8~1.2：得分 5　· 量比 0.5~0.8：得分 3<br>"
            "· 量比 &lt; 0.5：得分 1.5<br><br>"

            "<b>7. ATR%(14)</b> — 真实波幅占收盘价比例<br>"
            "· ATR% 1~2%：得分 8<br>"
            "· ATR% 0.5~1% 或 2~3%：得分 6.5<br>"
            "· ATR% 0.3~0.5% 或 3~5%：得分 4.5<br>"
            "· ATR% &lt; 0.3%：得分 3.5　· ATR% &gt; 5%：得分 2<br><br>"

            "<b>8. OBV Trend(5日斜率)</b> — 能量潮趋势<br>"
            "· 归一化斜率 &gt; 0.05：得分 9　· &gt; 0.02：得分 7.5<br>"
            "· &gt; 0：得分 6　· &gt; -0.02：得分 4.5<br>"
            "· &gt; -0.05：得分 3　· ≤ -0.05：得分 1.5<br><br>"

            "<b>9. Williams %R(14)</b> — 威廉指标<br>"
            "· %R ≤ -90：得分 9.5　· %R ≤ -80：得分 8<br>"
            "· %R ≤ -50：得分 5.5　· %R ≤ -20：得分 3.5<br>"
            "· %R &gt; -20：得分 1.5<br><br>"

            "<b>10. CCI(14)</b> — 顺势指标<br>"
            "· CCI &lt; -200：得分 9.5　· CCI &lt; -100：得分 8<br>"
            "· CCI &lt; 0：得分 6　· CCI &lt; 100：得分 4.5<br>"
            "· CCI &lt; 200：得分 2.5　· CCI ≥ 200：得分 1<br><br>"

            "<b>综合得分权重</b><br>"
            "MACD 15% | RSI/KDJ/MA 各12% | BB/VOL/OBV 各10% | ATR 7% | WR/CCI 各6%"
        )
        dlg    = QDialog(self)
        dlg.setWindowTitle("技术指标评分说明")
        dlg.resize(620, 700)
        lay    = QVBoxLayout(dlg)
        lbl    = QLabel(txt)
        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.RichText)
        scroll = QScrollArea()
        scroll.setWidget(lbl)
        scroll.setWidgetResizable(True)
        lay.addWidget(scroll)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        dlg.exec_()

    # ------------------------------------------------------------------
    def start_work(self, func, *args):
        self.tab.setSortingEnabled(False)
        self.worker = Worker(func, *args)
        self.worker.prog.connect(
            lambda c, t, s: (
                self.pb.setMaximum(max(t, 1)),
                self.pb.setValue(c),
                self.stat.setText(s)
            )
        )
        self.worker.done.connect(self._on_work_done)
        self.worker.start()

    def _on_work_done(self, msg):
        self.stat.setText(msg)
        self.pb.setValue(self.pb.maximum())
        self.lbl_mode.setText(
            "评分完成，结果已自动保存  |  可点击「② 刷新实时行情」获取最新报价"
        )
        self.refresh_table()

    # ------------------------------------------------------------------
    def refresh_table(self):
        self.tab.setSortingEnabled(False)
        self.tab.clear()
        self._build_live_view()
        self.tab.resizeColumnsToContents()
        self.tab.setSortingEnabled(True)

    def _build_live_view(self):
        if not self.dm.ctxs:
            self.tab.setColumnCount(1)
            self.tab.setHorizontalHeaderLabels(["提示"])
            self.tab.setRowCount(1)
            self._si(
                0, 0,
                "无数据，请点击「① 更新行情 & 评分」"
                "（历史天数建议设置 150 日历天以上）"
            )
            return

        # ★ 表头新增「行情截止日」列（紧跟名称之后）
        headers = [
            "序号", "代码", "名称", "行情截止日",        # ★ col 3
            "现价", "涨幅%",
            "综合分(0~100)",
            "RSI得分", "MACD得分", "KDJ得分", "BB得分", "MA得分",
            "量比得分", "ATR得分", "OBV得分", "WR得分", "CCI得分",
            "RSI值", "MACD柱", "K", "D", "J",
            "BB%", "MA5", "MA20", "量比", "ATR%",
            "OBV斜率", "WR", "CCI",
        ]
        self.tab.setColumnCount(len(headers))
        self.tab.setHorizontalHeaderLabels(headers)
        self.tab.setRowCount(len(self.dm.ctxs))

        for i, ctx in enumerate(self.dm.ctxs):
            # col 0~3
            self._si(i, 0, str(i + 1), i + 1)
            self._si(i, 1, ctx.code)
            self._si(i, 2, ctx.name)
            self._si(i, 3, ctx.last_date if ctx.last_date else "-")  # ★ 行情截止日

            # col 4~5  现价 / 涨幅
            p   = 0.0
            pct = 0.0
            try:
                p_raw = ctx.curr_price
                if str(p_raw) not in ['-', '', '0', '0.0']:
                    p = float(p_raw)
            except Exception:
                pass
            try:
                pct_raw = ctx.curr_pct
                if str(pct_raw) not in ['-', '']:
                    pct = float(pct_raw)
            except Exception:
                pass

            self._si(i, 4, f"{p:.2f}" if p else "-", p)
            fg_pct = (
                QColor("red")   if pct > 0 else
                QColor("green") if pct < 0 else None
            )
            self._si(i, 5, f"{pct:+.2f}%" if p else "-", pct, fg=fg_pct)

            # col 6  综合分
            res    = ctx.score_result
            comp   = res.get('composite', 0)
            scores = res.get('scores',    {})
            vals   = res.get('values',    {})

            if   comp >= 75: bg_comp = QColor("#FFCDD2"); fg_comp = QColor("#B71C1C")
            elif comp >= 60: bg_comp = QColor("#FFE0B2"); fg_comp = QColor("#E65100")
            elif comp >= 45: bg_comp = None;              fg_comp = None
            else:            bg_comp = QColor("#E8F5E9"); fg_comp = QColor("#2E7D32")

            self._si(i, 6, f"{comp:.1f}", comp, bg=bg_comp, fg=fg_comp)

            # col 7~16  子得分
            score_keys = ['RSI', 'MACD', 'KDJ', 'BB', 'MA', 'VOL', 'ATR', 'OBV', 'WR', 'CCI']
            for j, key in enumerate(score_keys):
                sv = scores.get(key, 0)
                try:    sv = float(sv)
                except: sv = 0.0
                bg_s = QColor("#FFCDD2") if sv >= 8.5 else None
                fg_s = QColor("#B71C1C") if sv >= 8.5 else None
                self._si(i, 7 + j, f"{sv:.1f}", sv, bg=bg_s, fg=fg_s)

            # col 17~29  指标原始值
            val_map = [
                ('RSI',        '.2f'),
                ('MACD_Hist',  '.4f'),
                ('K',          '.2f'),
                ('D',          '.2f'),
                ('J',          '.2f'),
                ('BB_pct',     '.3f'),
                ('MA5',        '.2f'),
                ('MA20',       '.2f'),
                ('VolRatio',   '.2f'),
                ('ATR_pct',    '.3f'),
                ('OBV_Slope',  '.5f'),
                ('WR',         '.2f'),
                ('CCI',        '.2f'),
            ]
            for j, (vkey, fmt) in enumerate(val_map):
                v = vals.get(vkey, '')
                try:
                    fv  = float(v)
                    txt = f"{fv:{fmt}}"
                    self._si(i, 17 + j, txt, fv)
                except Exception:
                    self._si(i, 17 + j, "-", 0)

    # ------------------------------------------------------------------
    def _si(self, r, c, txt, sort_val=None, bg=None, fg=None):
        it = NumericItem(str(txt))
        if sort_val is not None:
            it.setData(Qt.UserRole, sort_val)
        if bg: it.setBackground(bg)
        if fg: it.setForeground(fg)
        it.setTextAlignment(Qt.AlignCenter)
        self.tab.setItem(r, c, it)


# =============================================================================
# 入口
# =============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w   = MainWindow()
    w.show()
    sys.exit(app.exec_())

