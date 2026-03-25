#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd

TABLE5 = Path('/root/问卷/附图表/表5_按构念与经验高低组的WWR描述统计与配对比较.xlsx')
TABLE6 = Path('/root/问卷/附图表/表6_按构念与经验高低组的Complexity描述统计与组内比较.xlsx')

SUBDOMAIN_MAP = {
    'S1': 'Task supportiveness in space',
    'S2': 'Task supportiveness in space',
    'S3': 'Task supportiveness in space',
    'S4': 'Affective and behavioral outcomes',
    'S5': 'Affective and behavioral outcomes',
    'B1': 'Functional equipment appraisal',
    'B2': 'Functional equipment appraisal',
    'B3': 'Functional equipment appraisal',
}

GROUP_MAP = {'High': 'High', 'Low': 'Low'}


def rebuild_table5() -> None:
    df = pd.read_excel(TABLE5)
    for c in ['ConstructZH', 'ConstructEN']:
        if c in df.columns:
            df = df.drop(columns=[c])
    df['Measure'] = df['Measure'].astype(str).str.strip()
    df['Group'] = df['Group'].astype(str).map(lambda x: GROUP_MAP.get(x, x))
    df.insert(0, 'Domain', 'Questionnaire')
    df.insert(1, 'Subdomain', df['Measure'].map(lambda x: SUBDOMAIN_MAP.get(x, 'Other')))
    cols = ['Domain', 'Subdomain', 'Measure', 'Group', 'WWR15 M ± SD', 'WWR45 M ± SD', 'WWR75 M ± SD', 'Significant pairwise contrasts']
    df = df[[c for c in cols if c in df.columns]]
    order = {'S1':1,'S2':2,'S3':3,'S4':4,'S5':5,'B1':6,'B2':7,'B3':8}
    group_order = {'High': 1, 'Low': 2}
    df['_m'] = df['Measure'].map(order).fillna(999)
    df['_g'] = df['Group'].map(group_order).fillna(999)
    df = df.sort_values(['_m','_g']).drop(columns=['_m','_g']).reset_index(drop=True)
    with pd.ExcelWriter(TABLE5, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Table 5', index=False)


def rebuild_table6() -> None:
    df = pd.read_excel(TABLE6)
    for c in ['ConstructZH', 'ConstructEN']:
        if c in df.columns:
            df = df.drop(columns=[c])
    df['Measure'] = df['Measure'].astype(str).str.strip()
    df['Group'] = df['Group'].astype(str).map(lambda x: GROUP_MAP.get(x, x))
    df.insert(0, 'Domain', 'Questionnaire')
    df.insert(1, 'Subdomain', df['Measure'].map(lambda x: SUBDOMAIN_MAP.get(x, 'Other')))
    cols = ['Domain', 'Subdomain', 'Measure', 'Group', 'C0 M ± SD', 'C1 M ± SD', 'Mean difference (C0–C1)', 'p', 'Effect summary']
    df = df[[c for c in cols if c in df.columns]]
    order = {'S1':1,'S2':2,'S3':3,'S4':4,'S5':5,'B1':6,'B2':7,'B3':8}
    group_order = {'High': 1, 'Low': 2}
    df['_m'] = df['Measure'].map(order).fillna(999)
    df['_g'] = df['Group'].map(group_order).fillna(999)
    df = df.sort_values(['_m','_g']).drop(columns=['_m','_g']).reset_index(drop=True)
    with pd.ExcelWriter(TABLE6, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Table 6', index=False)


def main() -> int:
    rebuild_table5()
    rebuild_table6()
    print(TABLE5)
    print(TABLE6)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
