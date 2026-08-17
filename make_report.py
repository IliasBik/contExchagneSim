"""
make_report.py — собрать полный отчёт по симуляции.

Запуск:  python make_report.py

Никаких параметров у сборщика нет и быть не должно: эксперимент настраивается
только в SimConfig (agent_simulation.py), включая каталог отчёта. Сборка
делает четыре вещи:

    1) берёт прогон из кэша или считает заново (кэш инвалидируется и при
       правке конфига, и при правке исходников симуляции);
    2) строит все фигуры в report/figures/*.pdf;
    3) генерирует report/report.tex (XeLaTeX + polyglossia + DejaVu);
    4) собирает report/report.pdf доступным движком.

Движок ищется в PATH и в стандартных местах установки MiKTeX/TeX Live;
порядок предпочтения — xelatex, lualatex, tectonic. Если движка нет, .tex
и фигуры всё равно остаются на диске, а в конце печатается, что именно
поставить.

В конце печатается короткая сводка прогона. Подробный терминальный отчёт
по агентам никуда не делся: он по-прежнему выводится по
`python agent_simulation.py`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import agent_simulation as A
import report_data as D
import report_document as Doc
import report_figures as R

# движки в порядке предпочтения: (имя, аргументы сборки)
ENGINES = (
    ("xelatex", ["-interaction=nonstopmode", "-file-line-error"]),
    ("lualatex", ["-interaction=nonstopmode", "-file-line-error"]),
    ("tectonic", ["--keep-logs"]),
)

# куда MiKTeX и TeX Live ставятся, если их не прописали в PATH текущей сессии
EXTRA_BIN_DIRS = (
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/MiKTeX/miktex/bin/x64",
    Path("C:/Program Files/MiKTeX/miktex/bin/x64"),
    Path("C:/texlive/2025/bin/windows"),
    Path("C:/texlive/2024/bin/windows"),
)


def find_engine():
    """Первый доступный движок: (имя, путь, аргументы) или None."""
    for name, args in ENGINES:
        found = shutil.which(name)
        if found:
            return name, found, args
        for folder in EXTRA_BIN_DIRS:
            candidate = folder / f"{name}.exe"
            if candidate.exists():
                return name, str(candidate), args
    return None


def compile_pdf(tex: Path, verbose: bool = True) -> Path | None:
    """Собирает PDF. Два прохода — оглавление и ссылки на фигуры."""
    engine = find_engine()
    if engine is None:
        return None
    name, path, args = engine
    passes = 1 if name == "tectonic" else 2
    for run_no in range(1, passes + 1):
        if verbose:
            print(f"  {name}, проход {run_no} из {passes} ...")
        result = subprocess.run([path, *args, tex.name], cwd=tex.parent,
                                capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        pdf = tex.with_suffix(".pdf")
        if not pdf.exists():
            print(f"\n!! {name} не смог собрать PDF. Последние строки вывода:")
            print("\n".join(result.stdout.strip().splitlines()[-25:]))
            return None
    return tex.with_suffix(".pdf")


def print_summary(run: D.RunData) -> None:
    """Короткая сводка прогона в терминал — то же, что в таблицах отчёта."""
    s = D.summary(run)
    ph = s["phases_obj"]
    print("=" * 78)
    print(f"Фазы: прогрев {ph.warmup[0]}–{ph.warmup[1]}, "
          f"отбор {ph.selection[0]}–{ph.selection[1]}, "
          f"стационар {ph.stationary[0]}–{ph.stationary[1]}")
    print(f"gamma={s['gamma']:.4g}, выжило {s['alive']} из {s['n_agents']}, "
          f"оборот CE {s['ce']['gross_mean']:.4f} X1/тик, "
          f"хеджей {int(s['world']['1']['hedge_count'] + s['world']['2']['hedge_count'])}")
    print(f"{'семейство':<10}{'живых':>8}{'PnL':>12}{'переоценка':>14}"
          f"{'хедж':>12}{'оборот':>12}")
    for kind in D.KINDS:
        k = s["kinds"][kind]
        print(f"{kind:<10}{k['alive']:>4}/{k['n']:<3}{k['pnl_total']:>+12.4f}"
              f"{k['reval']:>+14.4f}{k['hedge']:>+12.4f}{k['turnover']:>12.1f}")
    print(f"Невязки: клиринг {s['ce']['imbalance_max']:.1e}, "
          f"разложение PnL {s['ce']['pnl_residual_max']:.1e} X1")
    print("=" * 78)


def main() -> int:
    cfg = A.SimConfig()
    outdir = Path(cfg.report_dir)
    started = time.time()

    print("=" * 78)
    print("Сборка отчёта")
    print("=" * 78)

    run = D.load_or_build(cfg)

    print("Фигуры:")
    specs = R.build_all(run, outdir / "figures")

    print("Документ:")
    tex = Doc.build_tex(run, specs, outdir)
    pdf = compile_pdf(tex)

    print_summary(run)
    print(f"Фигур: {len(specs)};  время сборки: {time.time() - started:.1f} с")
    if pdf is not None:
        print(f"Готово: {pdf.resolve()}")
    else:
        print(f"PDF не собран: LaTeX-движок не найден.\n"
              f"Исходник и фигуры на месте: {tex.resolve()}\n"
              f"Поставьте любой из движков и запустите сборку снова:\n"
              f"  winget install MiKTeX.MiKTeX\n"
              f"  winget install TectonicProject.Tectonic")
    print("=" * 78)
    return 0 if pdf is not None else 1


if __name__ == "__main__":
    sys.exit(main())
