"""Build a refined EO panel from SEC XBRL facts.

For each US-listed firm we pull, from SEC companyconcept endpoints:
 - quarterly revenue (RevenueFromContractWithCustomerExcludingAssessedTax)
 - remaining performance obligations (RPO, point-in-time backlog proxy)
 - period-end common shares outstanding (falls back across tag variants)
 - total debt and cash (to build enterprise value)
Market cap uses the quarter-end price times period-end shares (not a single
current snapshot), and EV = mktcap + debt - cash. EV/Sales uses TTM revenue.
All values are public. This script writes eo_panel_refined.csv.
"""
import urllib.request, json, time
import pandas as pd, numpy as np
import yfinance as yf

HEADERS = {'User-Agent': 'Academic Research research@example.edu'}
FIRMS = {'PL': '0001836833', 'BKSY': '0001753539',
         'SATL': '0001874315', 'SPIR': '0001816017'}


def concept(cik, tag, taxonomy='us-gaap'):
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{tag}.json"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        return json.loads(urllib.request.urlopen(req, timeout=20).read())
    except Exception:
        return None


def duration_facts(cik, tag, lo=80, hi=100, taxonomy='us-gaap'):
    """Quarterly (≈3-month) duration facts."""
    d = concept(cik, tag, taxonomy)
    if not d:
        return pd.DataFrame()
    rows = []
    for unit_key, arr in d.get('units', {}).items():
        for u in arr:
            if u.get('form') in ('10-Q', '10-K') and u.get('start') and u.get('end'):
                s, e = pd.to_datetime(u['start']), pd.to_datetime(u['end'])
                if lo <= (e - s).days <= hi:
                    rows.append({'end': e, 'val': u['val']})
    return (pd.DataFrame(rows).drop_duplicates(subset=['end'])
            .sort_values('end')) if rows else pd.DataFrame()


def instant_facts(cik, tags, taxonomy='us-gaap'):
    """Point-in-time (instant) facts, trying tag variants in order."""
    for tag in tags:
        d = concept(cik, tag, taxonomy)
        if not d:
            continue
        rows = []
        for unit_key, arr in d.get('units', {}).items():
            for u in arr:
                if u.get('form') in ('10-Q', '10-K') and u.get('end') and not u.get('start'):
                    rows.append({'end': pd.to_datetime(u['end']), 'val': u['val']})
        if rows:
            df = (pd.DataFrame(rows).drop_duplicates(subset=['end'], keep='last')
                  .sort_values('end'))
            df.columns = ['end', tag]
            return df, tag
    return pd.DataFrame(columns=['end']), None


def build_firm(t, cik):
    rev = duration_facts(cik, 'RevenueFromContractWithCustomerExcludingAssessedTax')
    rev = rev.rename(columns={'val': 'revenue'})
    rpo, _ = instant_facts(cik, ['RevenueRemainingPerformanceObligation'])
    if not rpo.empty:
        rpo.columns = ['end', 'rpo']
    shares, stag = instant_facts(cik, ['CommonStockSharesOutstanding',
                                       'EntityCommonStockSharesOutstanding'])
    if not shares.empty:
        shares.columns = ['end', 'shares']
    # Fallback: weighted-average diluted shares (a duration fact present for
    # every firm-quarter). Used only where the instant share count is missing.
    wavg = duration_facts(cik, 'WeightedAverageNumberOfDilutedSharesOutstanding')
    if not wavg.empty:
        wavg = wavg.rename(columns={'val': 'shares_wavg'})
    debt, dtag = instant_facts(cik, ['LongTermDebt', 'LongTermDebtNoncurrent',
                                     'DebtInstrumentCarryingAmount'])
    if not debt.empty:
        debt.columns = ['end', 'debt']
    cash, ctag = instant_facts(cik, ['CashAndCashEquivalentsAtCarryingValue'])
    if not cash.empty:
        cash.columns = ['end', 'cash']

    df = rev
    for extra in [rpo, shares, debt, cash]:
        if not extra.empty:
            df = df.merge(extra, on='end', how='left')
    if not wavg.empty:
        df = df.merge(wavg, on='end', how='left')
    else:
        df['shares_wavg'] = np.nan
    for col in ['rpo', 'shares', 'debt', 'cash', 'shares_wavg']:
        if col not in df.columns:
            df[col] = np.nan
    # Prefer reported instant share count; fall back to diluted weighted avg.
    df['shares'] = df['shares'].fillna(df['shares_wavg'])
    df['ticker'] = t
    return df


def main():
    frames = [build_firm(t, cik) for t, cik in FIRMS.items()]
    for _ in FIRMS:
        time.sleep(0.2)
    panel = pd.concat(frames, ignore_index=True).sort_values(['ticker', 'end'])

    # TTM revenue and YoY growth
    panel['ttm_rev'] = panel.groupby('ticker')['revenue'].transform(
        lambda s: s.rolling(4).sum())
    panel['rev_yoy'] = panel.groupby('ticker')['revenue'].transform(
        lambda s: s.pct_change(4))

    # Prices as of each quarter end
    prices = {}
    for t in FIRMS:
        p = yf.download(t, start='2020-01-01', end='2026-08-01',
                        progress=False, auto_adjust=True)['Close']
        if hasattr(p, 'columns'):
            p = p.iloc[:, 0]
        prices[t] = p.dropna()

    def price_asof(row):
        s = prices[row['ticker']][prices[row['ticker']].index <= row['end']]
        return float(s.iloc[-1]) if len(s) else np.nan

    panel['price'] = panel.apply(price_asof, axis=1)
    # Forward-fill shares within firm (shares reported on cover, dense enough)
    panel['shares'] = panel.groupby('ticker')['shares'].ffill()
    panel['mktcap'] = panel['price'] * panel['shares']
    panel['debt'] = panel['debt'].fillna(0.0)
    panel['cash'] = panel.groupby('ticker')['cash'].ffill()
    panel['ev'] = panel['mktcap'] + panel['debt'] - panel['cash']
    panel['ev_sales'] = panel['ev'] / panel['ttm_rev']
    panel['ps_ttm'] = panel['mktcap'] / panel['ttm_rev']
    panel['rpo_to_rev'] = panel['rpo'] / panel['ttm_rev']
    panel['year'] = panel['end'].dt.year

    panel.to_csv('data/eo_panel_refined.csv', index=False)
    print("Refined panel written:", panel.shape)
    print(panel.groupby('ticker').agg(
        n=('revenue', 'size'),
        ev_sales_n=('ev_sales', lambda x: np.isfinite(x).sum()),
        rpo_n=('rpo', lambda x: x.notna().sum())))
    return panel


if __name__ == '__main__':
    main()
