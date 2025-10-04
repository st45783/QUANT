import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('top_50_zscore_rank_3Y.csv')

# 제거할 ticker 리스트
tickers_to_remove = ['AACT', 'AACT U', 'AACT WS', 'DMYY U', 'DMYY', 'DMYY WS', 'VHC', 
                     'ACVA', 'BGFV', 'BCLI', 'IROHR', 'IROHU', 'BASE', 'CGBS', 'CGBSW', 
                     'DALN', 'DBGIW', 'FMTO', 'GOVXW', 'GECCZ', 'HYMCL', 'LSB', 'LSBPW', 
                     'LIPO', 'NERD', 'XAGE', 'XAGEW', 'MODV', 'COOP', 'OUSTW', 'PTPI', 
                     'PEV', 'SYTAW', 'KHC25', 'THTX', 'TCBX', 'TTNP', 'VXRT', 'WLGS', 'ZVSA']

# 제거 전 행 개수
print(f"제거 전 행 개수: {len(df)}")

# 제거할 ticker 중 실제로 존재하는 ticker 확인
existing_tickers = df[df['Ticker'].isin(tickers_to_remove)]['Ticker'].tolist()
print(f"제거될 ticker 개수: {len(existing_tickers)}")
print(f"제거될 ticker: {existing_tickers}")

# 해당 ticker들을 제외한 데이터프레임 생성
df_filtered = df[~df['Ticker'].isin(tickers_to_remove)]

# 제거 후 행 개수
print(f"제거 후 행 개수: {len(df_filtered)}")

# 결과를 새로운 CSV 파일로 저장
df_filtered.to_csv('all_stocks_raw_factors_filtered.csv', index=False)
print("필터링된 데이터가 'all_stocks_raw_factors_filtered.csv'에 저장되었습니다.")
