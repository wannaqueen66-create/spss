#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd

FILES = [
    Path('/root/问卷/附图表/表4_B1-B3的描述性统计与修正后LMM结果.xlsx'),
    Path('/root/问卷/附图表/表4_B1-B3的描述性统计与模型拟合状态.xlsx'),
]

SUBDOMAIN_MAP = {
    'B1': 'Functional equipment appraisal',
    'B2': 'Functional equipment appraisal',
    'B3': 'Functional equipment appraisal',
    'Bmean': 'Functional equipment appraisal (composite)',
}


def rebuild_main(path: Path) -> None:
    df = pd.read_excel(path)
    for c in ['ConstructZH', 'ConstructEN']:
        if c in df.columns:
            df = df.drop(columns=[c])

    df['Measure'] = df['Measure'].astype(str).str.strip()
    df.insert(0, 'Domain', 'Questionnaire')
    df.insert(1, 'Subdomain', df['Measure'].map(lambda x: SUBDOMAIN_MAP.get(x, 'Other')))

    cols = ['Domain', 'Subdomain', 'Measure', 'Model formula', 'Overall M ± SD', 'High M ± SD', 'Low M ± SD',
            'WWR (F, p, FDR p)', 'ExperienceGroup (F, p, FDR p)', 'WWR × ExperienceGroup (F, p, FDR p)']
    keep = [c for c in cols if c in df.columns]
    df = df[keep]

    order = {'B1': 1, 'B2': 2, 'B3': 3, 'Bmean': 4}
    df['_o'] = df['Measure'].map(order).fillna(999)
    df = df.sort_values('_o').drop(columns=['_o']).reset_index(drop=True)

    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Table 4', index=False)


def rebuild_status(path: Path) -> None:
    df = pd.read_excel(path)
    for c in ['ConstructZH', 'ConstructEN']:
        if c in df.columns:
            df = df.drop(columns=[c])

    df['Measure'] = df['Measure'].astype(str).str.strip()
    df.insert(0, 'Domain', 'Questionnaire')
    df.insert(1, 'Subdomain', df['Measure'].map(lambda x: SUBDOMAIN_MAP.get(x, 'Other')))

    cols = ['Domain', 'Subdomain', 'Measure', 'Overall M ± SD', '95% CI', 'Model status', 'Failure reason', 'Note']
    keep = [c for c in cols if c in df.columns]
    df = df[keep]

    order = {'B1': 1, 'B2': 2, 'B3': 3, 'Bmean': 4}
    df['_o'] = df['Measure'].map(order).fillna(999)
    df = df.sort_values('_o').drop(columns=['_o']).reset_index(drop=True)

    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Table 4', index=False)


def main() -> int:
    rebuild_main(FILES[0])
    rebuild_status(FILES[1])
    for p in FILES:
        print(p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
