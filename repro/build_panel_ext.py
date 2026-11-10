# -*- coding: utf-8 -*-
import os, sys, io, glob, json
import numpy as np
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = r'H:\Claw_backup\财报分析模型\repro\data_ext'
EXTRA = r'H:\Claw_backup\财报分析模型\repro\data_ext\extra'
PV_PQ = r'h:\Claw_backup\vp_probe_data\stock_pv.parquet'
OUT_DIR = BASE

INCOME = ['ts_code', 'end_date', 'ann_date', 'report_type', 'n_income', 'n_income_attr_p',
          'total_revenue', 'revenue', 'oper_cost', 'operate_profit', 'non_oper_income',
          'total_profit', 'income_tax', 'rd_exp']
BAL = ['ts_code', 'end_date', 'ann_date', 'report_type', 'total_assets', 'total_liab',
       'total_hldr_eqy_exc_min_int', 'total_hldr_eqy_inc_min_int', 'total_cur_assets',
       'total_cur_liab', 'total_ncl', 'total_share', 'fix_assets', 'money_cap', 'trad_asset',
       'oth_cur_assets', 'oth_cur_liab', 'oth_assets', 'oth_nca', 'oth_ncl', 'acct_payable',
       'div_payable', 'oth_eqt_tools', 'inventories',
       'accounts_receiv', 'notes_receiv', 'prepayment', 'notes_payable', 'adv_receipts']
CF = ['ts_code', 'end_date', 'ann_date', 'report_type', 'n_cashflow_act', 'n_cashflow_inv_act',
      'n_cash_flows_fnc_act', 'c_pay_dist_dpcp_int_exp', 'c_pay_acq_const_fiolta',
      'n_recp_disp_fiolta', 'c_paid_to_for_empl', 'c_cash_equ_end_period']
FIA = ['ts_code', 'end_date', 'ann_date', 'report_type', 'grossprofit_margin', 'roe', 'roe_dt',
       'invest_capital', 'extra_item', 'profit_dedt', 'ar_turn', 'or_yoy', 'roe_yoy',
       'netprofit_yoy', 'debt_to_assets', 'current_ratio']

FLOW = {'n_income': 'ni', 'revenue': 'sale', 'oper_cost': 'cost', 'n_cashflow_act': 'cfo'}


def load(api, cols, d):
    fs = sorted(glob.glob(os.path.join(d, 'batch_*_%s.parquet' % api)))
    if not fs:
        return pd.DataFrame(columns=cols)
    out = []
    for f in fs:
        df = pd.read_parquet(f)
        out.append(df[[c for c in cols if c in df.columns]])
    return pd.concat(out, ignore_index=True)


def latest(df):
    df = df.drop_duplicates(subset=['ts_code', 'end_date', 'ann_date'])
    return df.sort_values(['ts_code', 'end_date', 'ann_date']).groupby(
        ['ts_code', 'end_date'], as_index=False).tail(1)


def build(d, tag):
    inc = latest(load('income', INCOME, d))
    bal = latest(load('balancesheet', BAL, d))
    cf = latest(load('cashflow', CF, d))
    fia = latest(load('fina_indicator', FIA, d))
    key = ['ts_code', 'end_date']
    rep = inc.rename(columns={'ann_date': 'a_inc'})
    for other, nm in [(bal, 'a_bal'), (cf, 'a_cf'), (fia, 'a_fia')]:
        rep = rep.merge(other.rename(columns={'ann_date': nm}), on=key, how='outer',
                        suffixes=('', '_%s' % nm))
    rep = rep.loc[:, ~rep.columns.duplicated()]
    ann_cols = [c for c in ['a_inc', 'a_bal', 'a_cf', 'a_fia'] if c in rep.columns]
    for c in ann_cols:
        rep[c] = pd.to_datetime(rep[c], errors='coerce')
    rep['ann_visible'] = rep[ann_cols].max(axis=1)
    rep['end_date'] = pd.to_datetime(rep['end_date'])
    rep = rep.dropna(subset=['ann_visible']).sort_values(['ts_code', 'end_date']).reset_index(drop=True)

    for src, tg in FLOW.items():
        rep['%s_cum' % tg] = pd.to_numeric(rep.get(src), errors='coerce')
    g = rep.groupby('ts_code', sort=False)
    prev_end = g['end_date'].shift(1)
    m3 = prev_end + pd.offsets.MonthEnd(3)
    for tg in FLOW.values():
        prev = g['%s_cum' % tg].shift(1)
        cur = rep['%s_cum' % tg]
        em = rep['end_date'].dt.month
        q = np.where(em == 3, cur, cur - prev)
        ok = (em == 3) | (prev_end.notna() & (rep['end_date'] == m3))
        rep['%s_q' % tg] = pd.Series(np.where(ok, q, np.nan), index=rep.index)
    for tg in FLOW.values():
        rep['%s_ttm' % tg] = (rep.groupby('ts_code', sort=False)['%s_q' % tg]
                              .rolling(4, min_periods=4).sum().reset_index(level=0, drop=True))

    rep['IT'] = np.where(rep.get('inventories', pd.Series(index=rep.index)).notna() & (rep.get('inventories') != 0),
                         rep['cost_ttm'] / rep.get('inventories'), np.nan)
    rep['RD'] = np.where(rep['sale_ttm'] != 0, pd.to_numeric(rep.get('rd_exp'), errors='coerce') / rep['sale_ttm'], np.nan)

    FIN = {'AT': 'total_assets', 'SEQ': 'total_hldr_eqy_inc_min_int', 'ICAPT': 'invest_capital',
           'PSTK': 'oth_eqt_tools', 'PPENT': 'fix_assets', 'LT': 'total_liab', 'CEQ': 'total_share',
           'AO': None, 'DLTT': 'total_ncl', 'LO': 'oth_ncl', 'ACO': 'oth_cur_assets',
           'CHE': None, 'LCO': 'oth_cur_liab', 'AP': 'acct_payable', 'DV': 'div_payable',
           'SALE': 'sale_ttm', 'XIDO': 'extra_item', 'IB': 'profit_dedt', 'NI': 'ni_ttm',
           'PI': 'total_profit', 'TXT': 'income_tax', 'NOPI': 'non_oper_income',
           'DVCASH': 'c_pay_dist_dpcp_int_exp', 'FINACT': 'n_cash_flows_fnc_act',
           'PURCHASE': 'c_pay_acq_const_fiolta', 'RECOVER': 'n_recp_disp_fiolta',
           'INVACT': 'n_cashflow_inv_act', 'EMPLOYEE': 'c_paid_to_for_empl',
           'CASH': 'c_cash_equ_end_period'}
    for v, src in FIN.items():
        if src is None:
            rep[v] = np.nan
        else:
            rep[v] = pd.to_numeric(rep.get(src), errors='coerce')
    rep['AO'] = pd.to_numeric(rep.get('oth_assets'), errors='coerce').fillna(0) + \
                pd.to_numeric(rep.get('oth_nca'), errors='coerce').fillna(0)
    rep['CHE'] = pd.to_numeric(rep.get('money_cap'), errors='coerce').fillna(0) + \
                 pd.to_numeric(rep.get('trad_asset'), errors='coerce').fillna(0)
    rep['GPR'] = pd.to_numeric(rep.get('grossprofit_margin'), errors='coerce')
    rep['ART'] = pd.to_numeric(rep.get('ar_turn'), errors='coerce')

    num = lambda c: pd.to_numeric(rep.get(c), errors='coerce')
    nsum = lambda *cs: pd.concat([num(c) for c in cs], axis=1).sum(axis=1, min_count=1)
    rep['CUSTOMER_LVL'] = nsum('notes_receiv', 'accounts_receiv', 'prepayment')
    rep['SUPPLIER_LVL'] = nsum('notes_payable', 'acct_payable', 'adv_receipts')
    py = rep[['ts_code', 'end_date', 'CUSTOMER_LVL', 'SUPPLIER_LVL']].copy()
    py['end_date'] = py['end_date'] + pd.DateOffset(years=1)
    py = py.rename(columns={'CUSTOMER_LVL': 'CUST_py', 'SUPPLIER_LVL': 'SUP_py'})
    rep = rep.merge(py, on=['ts_code', 'end_date'], how='left')
    rep['CUSTOMER'] = rep['CUSTOMER_LVL'] - rep['CUST_py']
    rep['SUPPLIER'] = rep['SUPPLIER_LVL'] - rep['SUP_py']

    rep = rep[rep['end_date'] >= '2008-01-01'].copy()
    rep['month_end'] = rep['end_date']
    rep.to_parquet(os.path.join(OUT_DIR, 'fin_events_%s.parquet' % tag), index=False)
    print('[%s] 事件表 %d 行 | 报表期 %s..%s | 股票 %d' % (
        tag, len(rep), rep['end_date'].min().date(), rep['end_date'].max().date(), rep['ts_code'].nunique()))
    cov = {v: round(100 * rep[v].notna().mean(), 1) for v in list(FIN) + ['AO', 'CHE', 'GPR', 'ART', 'IT', 'RD', 'CUSTOMER', 'SUPPLIER']}
    print('    财务变量非空率%%:', cov)
    return rep


MAPPING = {
    'AT': 'balancesheet.total_assets', 'SEQ': 'balancesheet.total_hldr_eqy_inc_min_int',
    'ICAPT': 'fina_indicator.invest_capital', 'PSTK': 'balancesheet.oth_eqt_tools(~)',
    'PPENT': 'balancesheet.fix_assets', 'LT': 'balancesheet.total_liab', 'CEQ': 'balancesheet.total_share',
    'AO': 'oth_assets+oth_nca(~)', 'DLTT': 'balancesheet.total_ncl', 'LO': 'balancesheet.oth_ncl',
    'ACO': 'balancesheet.oth_cur_assets', 'CHE': 'money_cap+trad_asset(~)', 'LCO': 'balancesheet.oth_cur_liab',
    'AP': 'balancesheet.acct_payable', 'DV': 'balancesheet.div_payable', 'SALE': 'income.revenue(TTM)',
    'XIDO': 'fina_indicator.extra_item', 'IB': 'fina_indicator.profit_dedt', 'NI': 'income.n_income(TTM)',
    'PI': 'income.total_profit', 'TXT': 'income.income_tax', 'NOPI': 'income.non_oper_income',
    'DVCASH': 'cashflow.c_pay_dist_dpcp_int_exp', 'FINACT': 'cashflow.n_cash_flows_fnc_act',
    'PURCHASE': 'cashflow.c_pay_acq_const_fiolta', 'RECOVER': 'cashflow.n_recp_disp_fiolta',
    'INVACT': 'cashflow.n_cashflow_inv_act', 'EMPLOYEE': 'cashflow.c_paid_to_for_empl',
    'CASH': 'cashflow.c_cash_equ_end_period', 'GPR': 'fina_indicator.grossprofit_margin',
    'ART': 'fina_indicator.ar_turn', 'IT': 'oper_cost/inventories(compute)', 'RD': 'income.rd_exp/revenue',
    'HHI_10': 'top10_holders HHI(compute)', 'PLEDGE': 'pledge_stat.pledge_count',
    'CSPLEDGE': 'pledge_stat.pledge_ratio', 'RESTRICTED': 'share_float.float_ratio',
    'SOESHARES': 'top10_holders.state(compute~)', 'INST': 'top10_floatholders.institution(compute~)',
    'NUM': 'industry count(compute)', 'HHI': 'industry concentration(compute)',
    'MARKETSHARE': 'revenue/industry(compute)', 'BOARD': 'NOT AVAILABLE', 'SUPERVISORS': 'NOT AVAILABLE',
    'SENIORMAN': 'NOT AVAILABLE',
    'CUSTOMER': 'PROXY: dYoY(notes_receiv+accounts_receiv+prepayment)',
    'SUPPLIER': 'PROXY: dYoY(notes_payable+acct_payable+adv_receipts)',
    'OPERATE': 'revenue/employees NOT AVAILABLE',
    'MEDIA': 'PROXY: month up-day ratio (daily pv)',
    'NEWS': 'PROXY: ln(month avg amount) (daily pv)',
}


def build_media_proxy():
    pv = pd.read_parquet(PV_PQ, columns=['ts_code', 'trade_date', 'amount', 'pct_chg'])
    pv['trade_date'] = pd.to_datetime(pv['trade_date'])
    pv['month_end'] = pv['trade_date'] + pd.offsets.MonthEnd(0)
    g = pv.groupby(['ts_code', 'month_end'])
    m = g.agg(amt_mean=('amount', 'mean'),
              up_ratio=('pct_chg', lambda s: (s > 0).mean())).reset_index()
    m['NEWS'] = np.log1p(m['amt_mean'].clip(lower=0))
    m['MEDIA'] = m['up_ratio']
    m = m[['ts_code', 'month_end', 'NEWS', 'MEDIA']]
    m.to_parquet(os.path.join(OUT_DIR, 'media_proxy_monthly.parquet'), index=False)
    print('[media] 月度代理 %d 行 | %s..%s' % (len(m), m['month_end'].min().date(), m['month_end'].max().date()))
    return m


def main():
    pilot = '--pilot' in sys.argv
    d = os.path.join(BASE, 'pilot') if pilot else os.path.join(BASE, 'by_batch')
    tag = 'pilot' if pilot else 'full'
    rep = build(d, tag)
    build_media_proxy()
    pd.DataFrame([{'var': k, 'source': v} for k, v in MAPPING.items()]).to_csv(
        os.path.join(OUT_DIR, 'var_mapping.csv'), index=False, encoding='utf-8-sig')
    print('mapping saved:', len(MAPPING), 'vars')


if __name__ == '__main__':
    main()
