"""
report_document.py — генератор LaTeX-исходника отчёта (XeLaTeX + polyglossia).

Документ собирается из четырёх разделов, как задумано:

    1. Симулятор мира          — справедливая цена, якорь, лимитные биржи;
    2. Агентская структура     — непрерывная биржа и четыре роли агентов;
    3. Графики прогона         — все аналитические фигуры с подписями;
    4. Аналитика симулятора    — числа прогона, выводы, контроль корректности;
    приложение                 — полный дамп конфигурации для воспроизведения.

Числа в тексте и таблицах не зашиты: они подставляются из сводки прогона
(report_data.summary), поэтому документ всегда описывает именно тот прогон,
по которому построены графики.

Шрифты DejaVu берутся из поставки matplotlib и копируются в каталог отчёта,
поэтому исходник собирается без установки системных шрифтов.

Внутри шаблонов математика набрана \\( ... \\) и \\[ ... \\]: знак доллара
не используется, чтобы подстановка через string.Template была безопасной.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict
from datetime import date
from pathlib import Path
from string import Template

import matplotlib
import numpy as np

import report_data as D

FONT_FILES = (
    "DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf", "DejaVuSerif-Italic.ttf",
    "DejaVuSerif-BoldItalic.ttf", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
    "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf",
    "DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf",
    "DejaVuSansMono-Oblique.ttf", "DejaVuSansMono-BoldOblique.ttf",
)


# --------------------------------------------------------------------------- #
# Мелкие помощники форматирования
# --------------------------------------------------------------------------- #

def num(x, digits: int = 2) -> str:
    """Число с фиксированной точностью и неразрывными разделителями тысяч."""
    text = f"{x:,.{digits}f}".replace(",", "\\,")
    return text


def sci(x) -> str:
    """Число в научной записи для математического режима."""
    if x == 0:
        return "0"
    mantissa, exponent = f"{x:.1e}".split("e")
    return f"{mantissa}\\cdot 10^{{{int(exponent)}}}"


def signed(x, digits: int = 3) -> str:
    return ("+" if x >= 0 else "\u2212") + num(abs(x), digits)


def escape(text: str) -> str:
    """Экранирование обычного текста (подписи фигур, значения конфигурации).

    Спецсимволы, которые сами превращаются в команды с фигурными скобками,
    подставляются последними — иначе их скобки экранировались бы повторно.
    """
    for bad, good in (("\\", "\x00"), ("~", "\x01"), ("^", "\x02")):
        text = text.replace(bad, good)
    for bad in ("_", "%", "&", "#", "$", "{", "}"):
        text = text.replace(bad, "\\" + bad)
    for bad, good in (("\x00", "\\textbackslash{}"),
                      ("\x01", "\\textasciitilde{}"),
                      ("\x02", "\\textasciicircum{}")):
        text = text.replace(bad, good)
    return text


def figure(spec, specs_by_key: dict, width: str = "0.97\\textwidth") -> str:
    """Плавающая фигура с подписью «Заголовок. Развёрнутое описание».

    Заголовок и подпись приходят из report_figures как обычный текст, поэтому
    экранируются целиком: незакрытый процент в подписи проглотил бы остаток
    строки и обвалил сборку.
    """
    spec = specs_by_key[spec] if isinstance(spec, str) else spec
    title, caption = escape(spec.title), escape(spec.caption)
    return "\n".join((
        # [H] вместо плавающего размещения: фигуры идут строго по тексту и
        # плотно пакуются по две на полосу, а не разъезжаются по страницам
        "\\begin{figure}[H]",
        "\\centering",
        f"\\includegraphics[width={width}]{{figures/{spec.key}.pdf}}",
        f"\\caption[{title}]{{\\textbf{{{title}.}} {caption}}}",
        f"\\label{{fig:{spec.key}}}",
        "\\end{figure}",
    ))


# --------------------------------------------------------------------------- #
# Преамбула
# --------------------------------------------------------------------------- #

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}

\usepackage{fontspec}
\usepackage{polyglossia}
\setmainlanguage{russian}
\setotherlanguage{english}

\setmainfont{DejaVuSerif}[
  Path=fonts/, Extension=.ttf,
  UprightFont=*, BoldFont=*-Bold,
  ItalicFont=*-Italic, BoldItalicFont=*-BoldItalic,
  Scale=0.92]
\setsansfont{DejaVuSans}[
  Path=fonts/, Extension=.ttf,
  UprightFont=*, BoldFont=*-Bold,
  ItalicFont=*-Oblique, BoldItalicFont=*-BoldOblique,
  Scale=0.92]
\setmonofont{DejaVuSansMono}[
  Path=fonts/, Extension=.ttf,
  UprightFont=*, BoldFont=*-Bold,
  ItalicFont=*-Oblique, BoldItalicFont=*-BoldOblique,
  Scale=0.86]

\usepackage[a4paper,left=2cm,right=2cm,top=2.1cm,bottom=2.1cm]{geometry}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{enumitem}
\usepackage[font=footnotesize,labelfont=bf]{caption}
\usepackage[unicode,hidelinks]{hyperref}

\setlist{nosep,leftmargin=1.4em}
\setlength{\parskip}{0.4em}
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.15}

% фигуры крупные, и по умолчанию LaTeX пускал бы на страницу лишь одну,
% оставляя половину полосы пустой; ослабляем ограничения на долю плавающих
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.95}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.7}
\setlength{\floatsep}{8pt plus 2pt minus 2pt}
\setlength{\textfloatsep}{10pt plus 2pt minus 2pt}

\newcommand{\term}[1]{\textbf{#1}}
\newcommand{\code}[1]{\texttt{\small #1}}
"""


# --------------------------------------------------------------------------- #
# Таблицы
# --------------------------------------------------------------------------- #

def table_venues(run: D.RunData, s: dict) -> str:
    cfg = run.cfg
    rows = []
    for name, venue in (("1", cfg.venue1), ("2", cfg.venue2)):
        w = s["world"][name]
        rows.append(" & ".join((
            f"биржа {escape(name)}",
            num(venue.arrival_rate, 1),
            num(venue.order_ttl, 0),
            num(venue.price_std, 2),
            num(w["spread_mean"], 3),
            num(w["depth_mean"], 1),
            num(w["vol_mean"] * 1e4, 1),
            num(w["volume_mean"], 2),
            num(w["trade_share"] * 100, 1),
        )) + " \\\\")
    return "\n".join(rows)


def table_kinds(run: D.RunData, s: dict) -> str:
    rows = []
    for kind in D.KINDS:
        k = s["kinds"][kind]
        rows.append(" & ".join((
            kind,
            f"{k['alive']} / {k['n']}",
            signed(k["pnl_total"], 3),
            signed(k["reval"], 3),
            signed(k["hedge"], 3),
            num(k["turnover"], 1),
            num(k["inventory_abs"], 3),
        )) + " \\\\")
    return "\n".join(rows)


def table_phases(run: D.RunData, s: dict) -> str:
    rows = []
    for label, rng_ in s["phases_obj"].named:
        p = s["phases"][label]
        rows.append(" & ".join((
            label,
            f"{rng_[0]}--{rng_[1]}",
            signed(p["err1_mean"], 1) + " / " + num(p["err1_std"], 1),
            signed(p["err2_mean"], 1) + " / " + num(p["err2_std"], 1),
            num(p["basis_x_std"], 2),
            num(p["basis_y_std"], 2),
            num(p["gross_mean"], 3),
        )) + " \\\\")
    return "\n".join(rows)


def table_hyper(run: D.RunData) -> str:
    """Средний PnL копии по значениям гиперпараметров и число выживших."""
    ph = D.phases(run)
    warm = D.phase_pnl(run, ph.warmup)
    rows = []
    for key, title in (("h_m", "h_m"), ("h_r", "h_R")):
        values = sorted({a[key] for a in run.agents if a[key] is not None})
        for value in values:
            trans = [i for i, a in enumerate(run.agents)
                     if a[key] == value and a["kind"] in ("T1", "T2")]
            arbs = [i for i, a in enumerate(run.agents)
                    if a[key] == value and a["kind"] in ("AX", "AY")]
            alive_t = sum(run.agents[i]["death_tick"] is None for i in trans)
            alive_a = sum(run.agents[i]["death_tick"] is None for i in arbs)
            cell_a = ("---" if not arbs else
                      signed(sum(run.equity[i, run.T] for i in arbs) / len(arbs), 4))
            rows.append(" & ".join((
                f"\\({title} = {value:g}\\)",
                signed(sum(warm[i] for i in trans) / len(trans), 4),
                signed(sum(run.equity[i, run.T] for i in trans) / len(trans), 4),
                f"{alive_t} / {len(trans)}",
                cell_a,
                (f"{alive_a} / {len(arbs)}" if arbs else "---"),
            )) + " \\\\")
    return "\n".join(rows)


def table_config(run: D.RunData) -> str:
    """Полный дамп SimConfig, включая вложенные параметры площадок."""
    cfg = asdict(run.cfg)
    comments = {
        "h_m_values": "сетка агрессивности",
        "h_r_values": "сетка чувствительности к риску",
        "arb_copies": "копий арбитражёра на значение h_m",
        "c0": "стартовый виртуальный капитал, X1",
        "kappa_t": "мощность семейства трансляторов, 1/тик",
        "kappa_a": "мощность арбитражёра от капитала, 1/тик",
        "gamma": "неприятие риска (None -- калибруется)",
        "q_max_fraction": "целевой хедж-порог как доля капитала",
        "capital_floor_frac": "пол капитала в риск-коэффициенте",
        "shading_clamp": "предохранитель на |g q| в экспоненте",
        "arb_shading": "шейдинг котировки арбитражёров",
        "arb_vol_half_life": "полураспад EWMA волатильности базиса",
        "total_steps": "длина прогона, тиков",
        "evolution_start": "тик включения отбора",
        "window": "окно оценки убытка при отборе",
        "book_snapshot_every": "период срезов стакана",
        "book_band": "полуширина полосы среза",
        "book_bin": "ширина ценового бина среза",
        "report_dir": "каталог отчёта",
        "seed": "зерно генератора",
        "warmup": "прогрев лимитных бирж, тиков",
        "tick_size": "шаг ценовой сетки",
        "initial_price": "стартовая цена",
        "depth_band": "полоса подсчёта глубины у мида",
        "anchor_half_life": "полураспад EWMA якоря",
        "fundamental_vol": "волатильность справедливой цены за тик",
        "progress_every": "период печати прогресса",
    }
    rows = []
    for key, value in cfg.items():
        if key in ("venue1", "venue2"):
            continue
        text = ", ".join(f"{v:g}" for v in value) if isinstance(value, (list, tuple)) \
            else escape(str(value))
        rows.append(f"\\code{{{escape(key)}}} & {text} & "
                    f"{comments.get(key, '')} \\\\")
    for venue_key in ("venue1", "venue2"):
        venue = cfg[venue_key]
        text = ", ".join(f"{k}={v}" for k, v in venue.items())
        rows.append(f"\\code{{{escape(venue_key)}}} & \\multicolumn{{2}}{{p{{0.62\\textwidth}}}}"
                    f"{{{escape(text)}}} \\\\")
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Значения для подстановки
# --------------------------------------------------------------------------- #

def _basis_correlation(run: D.RunData) -> float:
    """Корреляция базисов арбитражёров — мера их зеркальности."""
    basis = D.basis_bp(run)[1:]
    return float(np.corrcoef(basis[:, 0], basis[:, 1])[0, 1])


def values(run: D.RunData, s: dict) -> dict:
    cfg = run.cfg
    ph = s["phases_obj"]
    w1, w2 = s["world"]["1"], s["world"]["2"]
    warm, stat = s["phases"]["прогрев"], s["phases"]["стационар"]
    kinds = s["kinds"]
    hedge_total = w1["hedge_count"] + w2["hedge_count"]
    return {
        "date": date.today().strftime("%d.%m.%Y"),
        "T": f"{run.T:,}".replace(",", "\\,"),
        "seed": str(cfg.seed),
        "warmup_ticks": str(cfg.warmup),
        "n_agents": str(run.n),
        "n_alive": str(s["alive"]),
        "n_deaths": str(s["deaths"]),
        "elapsed": num(run.elapsed, 1),
        "gamma": num(run.gamma, 0),
        "c0": num(cfg.c0, 0),
        "kappa_t": f"{cfg.kappa_t:g}",
        "kappa_a": f"{cfg.kappa_a:g}",
        "q_max": f"{cfg.q_max_fraction:g}",
        "q_max_x1": num(cfg.q_max_fraction * cfg.c0, 1),
        "tick_size": f"{cfg.tick_size:g}",
        "initial_price": f"{cfg.initial_price:g}",
        "depth_band": f"{cfg.depth_band:g}",
        "anchor_hl": f"{cfg.anchor_half_life:g}",
        "fund_vol_bp": num(cfg.fundamental_vol * 1e4, 0),
        "fund_vol_run": num(cfg.fundamental_vol * (run.T ** 0.5) * 100, 1),
        "evolution_start": f"{cfg.evolution_start:,}".replace(",", "\\,"),
        "window": f"{cfg.window:,}".replace(",", "\\,"),
        "rate1": f"{cfg.venue1.arrival_rate:g}",
        "rate2": f"{cfg.venue2.arrival_rate:g}",
        "ttl": f"{cfg.venue1.order_ttl:g}",
        "price_std": f"{cfg.venue1.price_std:g}",
        "ewma_hl": f"{cfg.venue1.ewma_half_life:g}",
        # фазы
        "ph_warm_a": str(ph.warmup[0]), "ph_warm_b": str(ph.warmup[1]),
        "ph_sel_a": str(ph.selection[0]), "ph_sel_b": str(ph.selection[1]),
        "ph_stat_a": str(ph.stationary[0]), "ph_stat_b": str(ph.stationary[1]),
        "ph_note": ph.note,
        # мир
        "spread1": num(w1["spread_mean"], 3), "spread2": num(w2["spread_mean"], 3),
        "depth1": num(w1["depth_mean"], 1), "depth2": num(w2["depth_mean"], 1),
        "vol1": num(w1["vol_mean"] * 1e4, 1), "vol2": num(w2["vol_mean"] * 1e4, 1),
        "volume1": num(w1["volume_mean"], 2), "volume2": num(w2["volume_mean"], 2),
        "oneside2": num(w2["oneside_share"] * 100, 1),
        "dev1": num(abs(w1["mid_vs_fund_bp"]), 0),
        "hedge_n": num(hedge_total, 0),
        "hedge_n1": num(w1["hedge_count"], 0), "hedge_n2": num(w2["hedge_count"], 0),
        "hedge_notional": num(w1["hedge_notional"] + w2["hedge_notional"], 1),
        "hedge_res1": signed(w1["hedge_result"], 3),
        "hedge_res2": signed(w2["hedge_result"], 3),
        "volume_total": num(sum(run.volume.sum(axis=0)), 0),
        # непрерывная биржа
        "ce_gross_mean": num(s["ce"]["gross_mean"], 3),
        "ce_gross_total": num(s["ce"]["gross_total"], 0),
        "ce_orders": num(s["ce"]["orders_mean"], 1),
        "ce_rank": str(s["ce"]["rank_mode"]),
        "imbalance": sci(s["ce"]["imbalance_max"]),
        "residual": sci(s["ce"]["pnl_residual_max"]),
        "err1_warm": signed(warm["err1_mean"], 1),
        "err1_warm_sd": num(warm["err1_std"], 1),
        "err1_stat": signed(stat["err1_mean"], 1),
        "err1_stat_sd": num(stat["err1_std"], 1),
        "err2_warm_sd": num(warm["err2_std"], 1),
        "err2_stat_sd": num(stat["err2_std"], 1),
        "basis_x_stat": num(stat["basis_x_std"], 2),
        "basis_y_stat": num(stat["basis_y_std"], 2),
        "basis_corr": num(_basis_correlation(run), 6),
        # агенты
        "pnl_t1": signed(kinds["T1"]["pnl_total"], 3),
        "pnl_t2": signed(kinds["T2"]["pnl_total"], 3),
        "pnl_ax": signed(kinds["AX"]["pnl_total"], 3),
        "pnl_ay": signed(kinds["AY"]["pnl_total"], 3),
        "reval_t2": signed(kinds["T2"]["reval"], 3),
        "hedge_t2": signed(kinds["T2"]["hedge"], 3),
        "alive_t1": str(kinds["T1"]["alive"]), "alive_t2": str(kinds["T2"]["alive"]),
        "alive_ax": str(kinds["AX"]["alive"]), "alive_ay": str(kinds["AY"]["alive"]),
        # таблицы
        "table_venues": table_venues(run, s),
        "table_kinds": table_kinds(run, s),
        "table_phases": table_phases(run, s),
        "table_hyper": table_hyper(run),
        "table_config": table_config(run),
    }


# --------------------------------------------------------------------------- #
# Разделы документа
# --------------------------------------------------------------------------- #

TITLE = r"""
\begin{titlepage}
\centering
\vspace*{3cm}
{\sffamily\Huge\bfseries Непрерывная биржа портфельных заявок\par}
\vspace{0.6cm}
{\sffamily\LARGE симулятор мира, агентская экосистема и аналитика прогона\par}
\vspace{2.5cm}
{\large Отчёт по прогону от $date\par}
\vspace{1.2cm}
\begin{tabular}{rl}
длина прогона & $T тиков \\
зерно генератора & $seed \\
агентов в популяции & $n_agents (выжило $n_alive) \\
время счёта & $elapsed с \\
\end{tabular}
\vfill
{\small Документ и все графики построены автоматически по записи прогона;\\
конфигурация эксперимента задаётся только в \code{SimConfig}.\par}
\end{titlepage}

\tableofcontents
\clearpage

\section*{О чём этот отчёт}
\addcontentsline{toc}{section}{О чём этот отчёт}

Модель состоит из трёх слоёв, и отчёт устроен так же.

\term{Мир} --- две обычные биржи с лимитными стаканами и батч-аукционом,
цены которых связаны общим якорем генерации заявок. Над якорем стоит
экзогенная \emph{справедливая цена}: случайные <<новости>>, которые никто
из участников не наблюдает напрямую.

\term{Непрерывная биржа} (далее CE) --- площадка портфельных заявок,
которая клирится не сведением встречных заявок, а решением линейной задачи
в логарифмах цен. На ней торгуются четыре актива: \(X_1, Y_1\) с первой
площадки и \(X_2, Y_2\) со второй.

\term{Агенты} --- четыре семейства по 16 копий. Трансляторы \(T_1, T_2\)
переносят цену и ликвидность своей лимитной биржи на CE и хеджируют
накопленный инвентарь об домашний стакан; арбитражёры \(A_X, A_Y\)
торгуют только на CE, стягивая цены двойников с разных площадок.
Копии отличаются гиперпараметрами, и эволюционный отбор постепенно
выключает проигрывающих.

Разделы 1 и 2 описывают устройство модели, с формулами и небольшими
примерами <<на пальцах>>. Раздел 3 --- галерея графиков конкретного
прогона. Раздел 4 --- аналитика: что эти графики означают в числах,
включая проверки корректности. Полная конфигурация прогона вынесена
в приложение.
"""

SECTION_WORLD = r"""
\clearpage
\section{Симулятор мира}

\subsection{Справедливая цена и якорь}

Экзогенная справедливая цена --- это накопленное произведение лог-нормальных
шоков:
\[
  \Phi_t = \Phi_{t-1}\exp(\varepsilon_t),\qquad
  \varepsilon_t \sim \mathcal{N}(0, \sigma_F^2),\qquad
  \sigma_F = $fund_vol_bp\ \text{б.п. за тик.}
\]
За весь прогон это даёт блуждание уровня примерно на
\(\sigma_F\sqrt{T} \approx $fund_vol_run\%\). Ни один участник \(\Phi_t\) не
видит: шок входит в модель только через якорь генерации фонового потока.

Якорь --- это то, как рынок отслеживает справедливую цену. Он равен
экспоненциальному среднему взвешенных мидов, к которому каждый тик
применяется тот же шок:
\[
  \bar m_t = \frac{\sum_v m_{v,t}\, d_{v,t}}{\sum_v d_{v,t}},\qquad
  a_t = \bigl[(1-\alpha)\,a_{t-1} + \alpha\,\bar m_t\bigr]\exp(\varepsilon_t),
  \qquad \alpha = 1 - 2^{-1/h},\ h = $anchor_hl .
\]
Веса \(d_{v,t}\) --- это глубина у мида в полосе
\(\pm$depth_band\): тонкая площадка получает малый вес и не может утащить
общий уровень за собой. Обе биржи в одном тике видят \emph{один и тот же}
якорь --- в этом и состоит их связь; никакого прямого арбитража между
лимитными биржами в модели нет.

$fig_ex_coupling

\subsection{Одна лимитная биржа}

Внутри тика биржа проходит четыре шага: истечение старых заявок, генерация
фонового потока, батч-аукцион, обновление статистики.

\term{Фоновый поток.} Число заявок за тик \(N \sim \text{Пуассон}(\lambda_v)\).
Для каждой заявки \emph{независимо} разыгрываются сторона (монетка 50/50) и
цена \(p \sim \mathcal{N}(a_t, \sigma_p^2)\), округляемая к ценовой сетке с
шагом $tick_size. Именно независимость стороны и цены создаёт сделки: часть
покупок оказывается выше части продаж. Каждая заявка живёт $ttl тиков.

$fig_ex_book_formation

\term{Батч-аукцион.} Все сделки тика проходят по единой цене \(p^*\),
максимизирующей исполняемый объём:
\[
  p^* = \arg\max_p \min\bigl(D(p),\, S(p)\bigr),\qquad
  D(p) = \!\!\sum_{\text{покупки } p_i \ge p}\!\! q_i,\qquad
  S(p) = \!\!\sum_{\text{продажи } p_i \le p}\!\! q_i .
\]
При нескольких оптимумах берётся цена, ближайшая к предыдущей. На уровне,
где объёма не хватает всем, заявки исполняются pro-rata --- преимущества
от порядка прихода нет. Если пересечения нет, сделок в тике не происходит
и цена последнего аукциона не меняется.

$fig_ex_auction

\term{Наблюдаемые величины.} Мид \(m = (b + a)/2\) (при пустой стороне ---
цена последнего аукциона), спред \(s = a - b\), глубина у мида
\(d = \sum_{|p_i - m| \le $depth_band} q_i\) и волатильность как EWMA
квадратов лог-приростов мида с полураспадом $ewma_hl тиков.

\subsection{Две площадки прогона}

Площадки отличаются только густотой потока: у первой
\(\lambda = $rate1\) заявок за тик, у второй \(\lambda = $rate2\).
Всё остальное совпадает. Этого достаточно, чтобы получить <<толстую>> и
<<тонкую>> биржу с качественно разной микроструктурой.

\begin{table}[H]
\centering
\caption{Параметры площадок и то, во что они превращаются в прогоне
(средние по всем $T тикам).}
\small
\begin{tabular}{lrrrrrrrr}
\toprule
& \multicolumn{3}{c}{задано} & \multicolumn{5}{c}{получилось} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-9}
площадка & поток & TTL & \(\sigma_p\) & спред & глубина & вол., б.п.
& объём & \% тиков со сделкой \\
\midrule
$table_venues
\bottomrule
\end{tabular}
\end{table}

Тонкая биржа платит за разреженность потока втрое: спред шире, глубина
втрое меньше, а в $oneside2\% тиков в стакане вообще нет одной из сторон ---
в такие тики её транслятор молчит, а хедж невозможен.

\subsection{Оценка хеджа: <<бумажное>> исполнение}

Агентам нужен способ сбросить инвентарь об лимитный стакан. В текущей
версии это делается без влияния на рынок: заявка проходит по книге от
лучшей цены вглубь, средняя цена исполнения возвращается агенту, но сам
стакан не меняется. Такое приближение допустимо ровно потому, что
хеджи малы на фоне оборота площадки (панель г рис.~\ref{fig:world_volume}):
за прогон трансляторы отхеджировали $hedge_notional X1 при суммарном
объёме аукционов $volume_total штук.
"""

SECTION_AGENTS = r"""
\clearpage
\section{Агентская структура на непрерывной бирже}

\subsection{Непрерывная биржа портфельных заявок}

Состояние биржи --- вектор логарифмов цен \(S \in \mathbb{R}^n\),
\(P = \exp(S)\). Заявка --- это тройка \((w, z, \lambda)\):

\begin{itemize}
\item \(w\) --- веса по стоимости, \(\sum_i w_i = 0\) (заявка
самофинансируема: сколько стоимости вошло, столько и вышло);
\item \(z\) --- курс безразличия, уровень геометрического индекса
\(\Pi(P) = \prod_i P_i^{w_i}\), при котором заявка ничего не торгует;
\item \(\lambda\) --- агрессивность: нотионал за единицу времени на единицу
лог-расхождения, [X1/тик].
\end{itemize}

Сигнал заявки --- лог-расхождение \(f = \ln z - w \cdot S\); заявка торгует
нотионал \(\lambda f\,dt\) (при \(f > 0\) покупает портфель). Условие
клиринга --- нулевой поток стоимости по каждому активу:
\[
  \sum_a \lambda_a f_a\, w_{a,i} = 0 \quad \forall i
  \qquad\Longleftrightarrow\qquad
  A S = b,\quad A = \sum_a \lambda_a w_a w_a^{\mathsf{T}},\quad
  b = \sum_a \lambda_a \ln z_a\, w_a .
\]
Это в точности взвешенная задача наименьших квадратов
\(S^* = \arg\min_S \sum_a \lambda_a (w_a\cdot S - \ln z_a)^2\): цена
клиринга --- \(\lambda\)-взвешенное среднее котировок в логарифмах.
Поскольку \(\sum_i w_{a,i} = 0\) у каждой заявки, \(A\mathbf{1} = 0\), и
решение определено с точностью до общего масштаба цен --- этот произвол
снимается выбором единицы счёта (здесь \(P_{X_1} \equiv 1\)).

Система решается в приращениях от текущих цен, а направления, на которые
никто не выставился, сохраняют прежнюю цену точно --- без регуляризации.
Ранг матрицы \(A\) и есть число ценовых направлений, которые книга
определяет в этот тик (в прогоне обычно $ce_rank из 4).

$fig_ex_ce_clearing

\subsection{Четыре роли}

\begin{table}[H]
\centering
\caption{Портфели заявок. Нога \(+1\) --- покупаемый актив, \(-1\) --- чем
за него платят. У трансляторов индекс портфеля равен цене \(Y\) в кассовом
активе площадки и напрямую сравним с мидом домашней биржи; у арбитражёров
безразличие достигается при курсе 1.}
\small
\begin{tabular}{llll}
\toprule
роль & портфель & курс безразличия \(z\) & что наблюдает \\
\midrule
\(T_1\) & \(+1\,Y_1,\ -1\,X_1\) & мид биржи 1 со сдвигом на позицию
  & стакан биржи 1 \\
\(T_2\) & \(+1\,Y_2,\ -1\,X_2\) & мид биржи 2 со сдвигом на позицию
  & стакан биржи 2 \\
\(A_X\) & \(+1\,X_1,\ -1\,X_2\) & 1 со сдвигом на позицию
  & только цены CE \\
\(A_Y\) & \(+1\,Y_1,\ -1\,Y_2\) & 1 со сдвигом на позицию
  & только цены CE \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Торговая мощность}

Транслятор опирается на домашний стакан: всё, что он наторгует на CE, ему
рано или поздно хеджировать об эту книгу. Качество книги измеряется
величиной с размерностью денег
\[
  H = \frac{d\, m\, \delta}{\max(s,\ \delta)},\qquad \delta = $tick_size,
\]
где \(d\,m\) --- нотионал, лежащий у мида, а \(\delta/s\) --- безразмерная
теснота книги (спред в один тик даёт 1). Мощность всего семейства
\(\Lambda = \kappa_T H\) с \(\kappa_T = $kappa_t\) делится между копиями
пропорционально капиталу и обратно пропорционально агрессивности:
\[
  \lambda_i = \Lambda \cdot \frac{\max(C_i,0)}{\sum_j \max(C_j,0)}
              \cdot \frac{1}{h_m^{(i)}},\qquad
  C_i = C_0 + \mathrm{PnL}_i,\quad C_0 = $c0 .
\]
Арбитражёру опереться не на что, кроме собственного капитала, поэтому у него
\(\lambda_i = \kappa_A \max(C_i,0)/h_m^{(i)}\) с \(\kappa_A = $kappa_a\).
Мёртвые копии выпадают из суммы, и их доля мощности автоматически перетекает
живым.

\subsection{Риск, котировка и хедж}

Риск-коэффициент агента:
\[
  g = \frac{h_R\,\gamma\,\sigma^2}{\max(C,\ C_{\min})}\quad [1/X_1],
\]
где \(\gamma\) --- общее для системы неприятие риска, \(\sigma^2\) --- EWMA
дисперсии лог-приростов релевантной цены. Богатый агент спокойнее держит
ту же позицию; пол \(C_{\min}\) не даёт разорившемуся делить на ноль.

Котировка сдвигается против собственной позиции (шейдинг):
\[
  z = z_0 \exp(-g\,q),
\]
где \(z_0\) --- базовая оценка курса (мид для транслятора, 1 для
арбитражёра), а \(q\) --- стоимость позиции по торгуемой ноге. Лонг опускает
котировку --- агент охотнее продаёт; сила пружины пропорциональна отклонению
и исчезает при \(q \to 0\).

$fig_ex_shading

Хедж включается, когда риск последней единицы позиции превысил потерю на её
сбросе:
\[
  \text{держать невыгодно, если } g|q| > c,\quad c = \tfrac12\,\frac{s}{m},
  \qquad\text{сбрасываем излишек } |q| - \frac{c}{g}.
\]
Никаких дополнительных констант здесь нет: и порог, и цель следуют из
сравнения риска и потери в одних единицах, а гистерезис встроен --- сразу
после хеджа риск равен потере, и повторный триггер не сработает, пока
позиция снова не вырастет.

$fig_ex_hedge

Единственная свободная константа \(\gamma\) калибруется по целевому порогу:
если при нейтральном \(h_R = 1\) требовать \(|q^*| = $q_max\,C\), то
\(\gamma = c/(\sigma^2 \cdot $q_max)\). По состоянию рынка после прогрева
это дало \(\gamma = $gamma\), то есть порог около $q_max_x1 X1 при
стартовом капитале.

\subsection{Гиперпараметры и эволюционный отбор}

Каждая копия несёт свои \(h_m\) (агрессивность: чем меньше, тем больше
\(\lambda\)) и --- у трансляторов --- \(h_R\) (чувствительность к риску: чем
больше, тем раньше и глубже хедж). Сетка \(4\times4\) даёт 16 копий на
семейство, всего $n_agents агентов.

С тика $evolution_start каждый тик деактивируется не более одного агента:
берётся тот, у кого убыток за последние $window тиков наибольший, но только
если (а) он убыточен и за всё время, и (б) он не последний живой в своём
семействе. Деактивированный навсегда перестаёт торговать, его прибыль
замораживается.

\subsection{Учёт и точное разложение прибыли}

Инвентарь агента живёт в балансах CE --- это единственный источник правды.
Прибыль считается по рынку: \(\mathrm{PnL} = \sum_i q_i P_i\) в единицах
\(X_1\). Стартовый капитал \(C_0\) на баланс не кладётся, он живёт только
в формулах доли и риска, поэтому прибыль равна ровно стоимости реального
портфеля.

Внутри тика цены CE меняются ровно один раз --- в клиринге. Отсюда точное
разложение изменения прибыли:
\[
  \Delta \mathrm{PnL}_t
  = \underbrace{\sum_i q_{i,t-1}\bigl(P_{i,t} - P_{i,t-1}\bigr)}
    _{\text{переоценка позиции}}
  + \underbrace{\sum_i \Delta q^{\text{сделки}}_{i}\,P_{i,t}}_{= 0}
  + \underbrace{\sum_i \Delta q^{\text{хедж}}_{i}\,P_{i,t}}
    _{\text{результат хеджа}} .
\]
Средний член равен нулю тождественно: \(\sum_i w_i = 0\), поэтому сделка на
CE в момент исполнения не создаёт и не уничтожает стоимость. Значит, весь
PnL --- это переоценка накопленной позиции плюс то, что агент выиграл или
потерял, скидывая инвентарь по встречной стороне книги. Невязка этого
тождества за весь прогон не превысила \($residual\) X1 --- то есть
разложение на рис.~\ref{fig:agents_structure} не приближённое, а точное.
"""

SECTION_FIGURES = r"""
\clearpage
\section{Графики прогона}

Ниже --- полная галерея по прогону длиной $T тиков (зерно $seed).
Прогон делится на три фазы, они размечены фоном на графиках по времени:

\begin{itemize}
\item \term{прогрев}, тики $ph_warm_a--$ph_warm_b: отбор ещё не включён,
живы все $n_agents копий, гиперпараметры никак не отобраны;
\item \term{отбор}, тики $ph_sel_a--$ph_sel_b: состав популяции меняется,
доли мощности перетекают выжившим;
\item \term{стационар}, тики $ph_stat_a--$ph_stat_b: состав зафиксирован,
режим установившийся.
\end{itemize}

Границу стационара выбирает не человек, а данные: $ph_note.

\subsection{Мир}

$fig_world_price

$fig_world_book

$fig_world_volume

$fig_world_liquidity

\subsection{Непрерывная биржа}

$fig_ce_price

$fig_ce_basis

$fig_ce_clearing

\subsection{Агенты}

$fig_agents_pnl

$fig_agents_turnover

$fig_agents_structure

$fig_agents_inventory

$fig_agents_hyper

$fig_agents_hyper_pnl
"""

SECTION_ANALYTICS = r"""
\clearpage
\section{Аналитика симулятора}

\subsection{Сводка прогона}

За $T тиков ($elapsed с счёта) непрерывная биржа провернула
$ce_gross_total X1 суммарного оборота, в среднем $ce_gross_mean X1 за тик
при $ce_orders исполнившихся заявках. Трансляторы сделали $hedge_n хеджей
($hedge_n1 на первой площадке, $hedge_n2 на второй) на $hedge_notional X1.
Отбор выключил $n_deaths копий из $n_agents.

\begin{table}[H]
\centering
\caption{Итоги по семействам: выжившие, прибыль и её точное разложение,
оборот на CE и средний размер удерживаемой позиции.}
\small
\begin{tabular}{lrrrrrr}
\toprule
семейство & выжило & PnL, X1 & переоценка & хедж & оборот, X1
& средняя \(|q|\) \\
\midrule
$table_kinds
\bottomrule
\end{tabular}
\end{table}

\subsection{Насколько точно CE переносит цену}

\begin{table}[H]
\centering
\caption{По фазам: ошибка трансляции (среднее / стандартное отклонение,
базисные пункты), разброс базисов арбитражёров и оборот CE.}
\small
\begin{tabular}{llrrrrr}
\toprule
фаза & тики & \(Y_1\) к миду 1 & \(Y_2\) к миду 2 &
\(\sigma\) базиса \(X\) & \(\sigma\) базиса \(Y\) & оборот/тик \\
\midrule
$table_phases
\bottomrule
\end{tabular}
\end{table}

Главный результат виден в правой части таблицы. Ошибка трансляции для
толстой площадки падает с $err1_warm_sd б.п. в прогреве до $err1_stat_sd
б.п. в стационаре, для тонкой --- с $err2_warm_sd до $err2_stat_sd б.п.
Смещения при этом почти нет ($err1_stat б.п. в стационаре): CE не
систематически дороже или дешевле лимитной биржи, она просто шумит вокруг
неё, и отбор гиперпараметров этот шум уменьшает.

Базисы арбитражёров держатся на порядок теснее: стандартное отклонение
$basis_x_stat б.п. по паре \(X_1/X_2\) и $basis_y_stat б.п. по
\(Y_1/Y_2\). Это ожидаемо: базис --- разница двух цен \emph{внутри} CE, и
чтобы его закрыть, арбитражёру не нужно ходить ни на какой внешний стакан.
То, что оба числа в каждой строке таблицы совпадают, --- не совпадение
и не ошибка; разбор в п.~4.3.

\subsection{Кольцевое тождество оборотов}

На панели а рис.~\ref{fig:agents_turnover} четыре линии совпадают. Это не
артефакт: так устроен клиринг. Обозначим \(V_k\) суммарный нотионал
семейства \(k\) со знаком. Условие нулевого потока стоимости по каждому из
четырёх активов даёт четыре уравнения:
\[
  X_1:\ -V_{T_1} + V_{A_X} = 0,\qquad
  Y_1:\ +V_{T_1} + V_{A_Y} = 0,\qquad
  X_2:\ -V_{T_2} - V_{A_X} = 0,\qquad
  Y_2:\ +V_{T_2} - V_{A_Y} = 0,
\]
откуда \(V_{A_X} = V_{T_1} = -V_{T_2} = -V_{A_Y}\). Все четыре нотионала
равны по модулю, поэтому каждое семейство даёт ровно четверть оборота CE,
как бы ни отличались его \(\lambda\). Отличие возможно лишь в те тики,
когда копии внутри семейства торгуют в разные стороны и валовой оборот
превышает чистый; в прогоне медианное расхождение между семействами
в точности нулевое.

Практическое следствие: <<увеличить долю рынка>> отдельному семейству
нельзя --- можно только перераспределить оборот между копиями внутри
семейства. Именно это и делает отбор.

То же кольцо объясняет и вторую пару совпадающих чисел в таблице~4:
разбросы базисов \(X\) и \(Y\) равны, и это не опечатка. Из
\(V_{A_X} = -V_{A_Y}\) и \(V = \lambda f\) следует
\(\lambda_{A_X} f_{A_X} = -\lambda_{A_Y} f_{A_Y}\); семейства арбитражёров
устроены одинаково и имеют равную суммарную мощность, поэтому
\(f_{A_X} = -f_{A_Y}\) --- базисы оказываются точными зеркалами. Измеренная
на прогоне корреляция равна \($basis_corr\). Экономический смысл: разрыв
между площадками один, и виден он в двух парах активов с разными знаками;
закрыть их независимо нельзя.

\subsection{Кто зарабатывает и на чём}

Итог прогона: арбитражёры $pnl_ax и $pnl_ay X1, трансляторы $pnl_t1 и
$pnl_t2 X1. Асимметрия объясняется разложением PnL.

У арбитражёра компонента хеджа тождественно нулевая --- он не выходит на
лимитные биржи. Весь его заработок --- переоценка позиции: он набирает
базис, когда тот расходится, и получает прибыль, когда трансляторы
возвращают цены к паритету. Его позиция ограничена только шейдингом.

У транслятора картина обратная. Он вынужден периодически скидывать
инвентарь по встречной стороне книги, теряя полуспред; на тонкой площадке
спред шире ($spread2 против $spread1), и это прямо переносится в результат.
Показательно семейство \(T_2\): переоценка позиции дала $reval_t2 X1, а
хедж --- $hedge_t2 X1; итоговый $pnl_t2 X1 --- разность двух больших
величин разного знака. Отсюда и разная выживаемость: из 16 копий каждого
семейства трансляторов осталось $alive_t1 и $alive_t2, а арбитражёров ---
$alive_ax и $alive_ay из 16.

Стоит назвать это прямо: в текущей конфигурации отбор выкашивает
трансляторов почти полностью. Это не поломка механики (тождество разложения
PnL выполняется с машинной точностью), а свойство параметров: расплата за
хедж систематически больше, чем премия за перенос цены. Правило отбора
защищает лишь последнюю копию семейства, поэтому трансляция цен на CE
не исчезает --- но держится на одном агенте на площадку.

\subsection{Что отобрала эволюция}

\begin{table}[H]
\centering
\caption{Средний PnL копии в разрезе значений гиперпараметров: у
трансляторов --- за фазу прогрева (именно по ней отбор и принимает решение)
и за весь прогон, у арбитражёров --- за весь прогон.}
\small
\begin{tabular}{lrrrrr}
\toprule
& \multicolumn{3}{c}{трансляторы} & \multicolumn{2}{c}{арбитражёры} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
значение & PnL за прогрев & PnL за прогон & выжило & PnL за прогон & выжило \\
\midrule
$table_hyper
\bottomrule
\end{tabular}
\end{table}

У арбитражёров зависимость от \(h_m\) монотонная и чистая: чем агрессивнее
копия (меньше \(h_m\)), тем больше её \(\lambda\), тем большую долю
семейного оборота она забирает и тем больше зарабатывает. Отбор здесь
никого не тронул --- прибыльны все.

У трансляторов сигнал слабее и сильно зашумлён: разброс PnL между копиями
одного семейства сопоставим с разбросом между значениями гиперпараметра,
поэтому отбор по окну в $window тиков выключает копии в порядке, который
ближе к случайному, чем к <<по заслугам>>. Панель г
рис.~\ref{fig:agents_pnl} показывает это прямо: связь между прибылью
в прогреве и прибылью в стационаре у трансляторов слабая.

\subsection{Контроль корректности}

Три независимые проверки, все на данных этого прогона:

\begin{enumerate}
\item \term{Клиринг не теряет стоимость.} Максимальная невязка условия
нулевого потока за весь прогон --- \($imbalance\) X1 при среднем обороте
$ce_gross_mean X1 за тик, то есть машинный ноль.
\item \term{Разложение PnL точное.} Максимальное расхождение между
изменением прибыли и суммой её компонент --- \($residual\) X1.
\item \term{Единица счёта неподвижна.} Цена \(X_1\) равна 1 по построению
калибра, и все величины отчёта выражены в ней; ранг книги ($ce_rank из 4)
показывает, что три ценовых направления определяются заявками, а четвёртое
и есть выбор единицы счёта.
\end{enumerate}

\subsection{Границы применимости}

Что в модели заведомо упрощено и на что это влияет:

\begin{itemize}
\item \term{Хедж не двигает рынок.} Заявка агента проходит по книге, но
стакан остаётся прежним. Приближение опирается на малость хеджей
относительно оборота площадки; при росте \(\kappa_T\) или числа копий его
придётся заменить настоящим исполнением.
\item \term{Заявка живёт один клиринг.} Агенты перевыставляются каждый тик,
поэтому книга CE не накапливает историю --- динамика полностью
определяется текущими наблюдениями.
\item \term{Отбор однонаправлен.} Копии только выключаются, новые не
рождаются и параметры не мутируют. Это отвечает на вопрос <<кто
проигрывает>>, но не на вопрос <<какие параметры оптимальны>>: для второго
нужен механизм воспроизводства.
\item \term{Один прогон --- одна реализация.} Все числа отчёта получены при
зерне $seed. Выводы о механике (тождества, разложения, соотношения
площадок) устойчивы, а конкретные PnL и порядок выбытия копий --- нет.
\end{itemize}
"""

APPENDIX = r"""
\clearpage
\appendix
\section{Конфигурация прогона}

Таблица ниже --- полный дамп \code{SimConfig}. Этого достаточно, чтобы
воспроизвести и прогон, и весь отчёт: при неизменных конфиге и исходниках
симуляции результат берётся из кэша, при любом изменении --- пересчитывается.

\begin{longtable}{p{0.24\textwidth}p{0.20\textwidth}p{0.46\textwidth}}
\toprule
параметр & значение & смысл \\
\midrule
\endhead
$table_config
\bottomrule
\end{longtable}
"""


# --------------------------------------------------------------------------- #
# Сборка
# --------------------------------------------------------------------------- #

def build_tex(run: D.RunData, specs: list, outdir: Path,
              verbose: bool = True) -> Path:
    """Пишет report.tex и кладёт рядом шрифты; возвращает путь к .tex."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _copy_fonts(outdir / "fonts")

    by_key = {spec.key: spec for spec in specs}
    subst = values(run, D.summary(run))
    for key in by_key:
        subst[f"fig_{key}"] = figure(key, by_key)

    body = "\n".join((TITLE, SECTION_WORLD, SECTION_AGENTS, SECTION_FIGURES,
                      SECTION_ANALYTICS, APPENDIX))
    text = "\n".join((PREAMBLE, "\\begin{document}",
                      Template(body).substitute(subst), "\\end{document}", ""))

    path = outdir / "report.tex"
    path.write_text(text, encoding="utf-8")
    if verbose:
        print(f"  LaTeX-исходник -> {path}")
    return path


def _copy_fonts(target: Path) -> None:
    """Шрифты DejaVu из поставки matplotlib — рядом с исходником."""
    source = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    target.mkdir(parents=True, exist_ok=True)
    for name in FONT_FILES:
        dst = target / name
        if not dst.exists():
            shutil.copyfile(source / name, dst)


if __name__ == "__main__":
    import report_figures as R

    run_data = D.load_or_build()
    out = Path(run_data.cfg.report_dir)
    figure_specs = R.build_all(run_data, out / "figures")
    build_tex(run_data, figure_specs, out)
