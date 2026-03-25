#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd

SRC = Path('/root/问卷/附图表/表2_S1-S5的LMM_fixed_effects结果.xlsx')
OUT = Path('/root/问卷/附图表/表2_S1-S5的LMM_fixed_effects结果.xlsx')

SUBDOMAIN_MAP = {
    'S1': 'Task supportiveness in space',
    'S2': 'Task supportiveness in space',
    'S3': 'Task supportiveness in space',
    'S4': 'Affective and behavioral outcomes',
    'S5': 'Affective and behavioral outcomes',
}


def main() -> int:
    xl = pd.ExcelFile(SRC)
    df = pd.read_excel(SRC, sheet_name=xl.sheet_names[0])

    df['Outcome'] = df['Outcome'].astype(str).str.strip()
    if 'ConstructZH' in df.columns:
        df = df.drop(columns=['ConstructZH'])
    if 'ConstructEN' in df.columns:
        df = df.drop(columns=['ConstructEN'])

    domain = pd.Series(['Questionnaire'] * len(df), index=df.index)
    subdomain = df['Outcome'].map(lambda x: SUBDOMAIN_MAP.get(x, 'Other'))

    df.insert(0, 'Domain', domain)
    df.insert(1, 'Subdomain', subdomain)

    effect_order = {
        'WWR': 0,
        'Complexity': 1,
        'ExperienceGroup': 2,
        'WWR × Complexity': 3,
        'WWR × ExperienceGroup': 4,
        'Complexity × ExperienceGroup': 5,
        'WWR × Complexity × ExperienceGroup': 6,
    }
    outcome_order = {'S1': 1, 'S2': 2, 'S3': 3, 'S4': 4, 'S5': 5}
    df['_o'] = df['Outcome'].map(outcome_order).fillna(999)
    df['_e'] = df['Effect'].map(effect_order).fillna(999)
    df = df.sort_values(['_o', '_e']).drop(columns=['_o', '_e']).reset_index(drop=True)

    with pd.ExcelWriter(OUT, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='表2', index=False)

    print(OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
