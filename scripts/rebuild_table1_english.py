#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd

SRC = Path('/root/问卷/附图表/表1_所有问卷指标的总体描述性统计.xlsx')
OUT = Path('/root/问卷/附图表/表1_Overall descriptive statistics (English).xlsx')

SUBDOMAIN_MAP = {
    'S1': 'Task supportiveness in space',
    'S2': 'Task supportiveness in space',
    'S3': 'Task supportiveness in space',
    'S4': 'Affective and behavioral outcomes',
    'S5': 'Affective and behavioral outcomes',
    'B1': 'Functional equipment appraisal',
    'B2': 'Functional equipment appraisal',
    'B3': 'Functional equipment appraisal',
    'Bmean': 'Functional equipment appraisal (composite)',
    'IPQ1': 'Presence',
    'IPQ2': 'Presence',
    'IPQ3': 'Presence',
    'IPQ4': 'Presence',
    'IPQ5': 'Presence',
    'IPQ6': 'Presence',
    'IPQ_mean': 'Presence (composite)',
}

DOMAIN_MAP = {
    'S1': 'Questionnaire',
    'S2': 'Questionnaire',
    'S3': 'Questionnaire',
    'S4': 'Questionnaire',
    'S5': 'Questionnaire',
    'B1': 'Questionnaire',
    'B2': 'Questionnaire',
    'B3': 'Questionnaire',
    'Bmean': 'Questionnaire',
    'IPQ1': 'Presence (IPQ)',
    'IPQ2': 'Presence (IPQ)',
    'IPQ3': 'Presence (IPQ)',
    'IPQ4': 'Presence (IPQ)',
    'IPQ5': 'Presence (IPQ)',
    'IPQ6': 'Presence (IPQ)',
    'IPQ_mean': 'Presence (IPQ)',
}


def main() -> int:
    xl = pd.ExcelFile(SRC)
    df = pd.read_excel(SRC, sheet_name=xl.sheet_names[0])

    out = pd.DataFrame()
    out['Domain'] = df['Measure'].astype(str).map(lambda x: DOMAIN_MAP.get(x, 'Other'))
    out['Subdomain'] = df['Measure'].astype(str).map(lambda x: SUBDOMAIN_MAP.get(x, 'Other'))
    out['Measure'] = df['Measure']
    out['n'] = df['n']
    out['Mean'] = df['Mean'].map(lambda x: f'{float(x):.3f}' if pd.notna(x) else '')
    out['SD'] = df['SD'].map(lambda x: f'{float(x):.3f}' if pd.notna(x) else '')
    out['95% CI'] = df['95% CI']
    out['Median'] = df['Median'].map(lambda x: f'{float(x):.3f}' if pd.notna(x) else '')

    domain_order = {'Questionnaire': 0, 'Presence (IPQ)': 1, 'Other': 2}
    sub_order = {
        'Task supportiveness in space': 0,
        'Affective and behavioral outcomes': 1,
        'Functional equipment appraisal': 2,
        'Functional equipment appraisal (composite)': 3,
        'Presence': 4,
        'Presence (composite)': 5,
        'Other': 6,
    }
    out['_d'] = out['Domain'].map(domain_order).fillna(99)
    out['_s'] = out['Subdomain'].map(sub_order).fillna(99)
    out['_m'] = out['Measure'].astype(str)
    out = out.sort_values(['_d', '_s', '_m']).drop(columns=['_d', '_s', '_m']).reset_index(drop=True)

    with pd.ExcelWriter(OUT, engine='openpyxl') as writer:
        out.to_excel(writer, sheet_name='Table 1', index=False)

    print(OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
