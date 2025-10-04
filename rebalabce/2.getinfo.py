import yfinance as yf
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime, timedelta

# --- 설정값 ---
INPUT_FILENAME = "all_us_tickers.csv"
OUTPUT_FILENAME = "all_stocks_raw_factors.csv"

print("✅ 파이프라인 2단계: 7가지 팩터 원본 데이터 수집을 시작합니다.")

# 📌 수동 제외 티커 리스트
# 상장폐지 예정, 거래 정지, 기타 위험 종목을 여기에 추가하세요
# NASDAQ: https://www.nasdaq.com/market-activity/stocks/issuers-pending-suspension-delisting
# NYSE: https://www.nyse.com/regulation/delistings
MANUAL_EXCLUDE_TICKERS = set(['AACT', 'AACT U', 'AACT WS', 'DMYY U', 'DMYY', 'DMYY WS', 'VHC', 'ACVA', 'BGFV', 'BCLI', 'IROHR', 'IROHU', 'BASE', 'CGBS', 'CGBSW', 'DALN', 'DBGIW', 'FMTO', 'GOVXW', 'GECCZ', 'HYMCL', 'LSB', 'LSBPW', 'LIPO', 'NERD', 'XAGE', 'XAGEW', 'MODV', 'COOP', 'OUSTW', 'PTPI', 'PEV', 'SYTAW', 'KHC25', 'THTX', 'TCBX', 'TTNP', 'VXRT', 'WLGS', 'ZVSA']
)

print(f"\n📋 수동 제외 티커 설정")
if MANUAL_EXCLUDE_TICKERS:
    print(f"  ⚠️ 제외할 종목 {len(MANUAL_EXCLUDE_TICKERS)}개: {sorted(MANUAL_EXCLUDE_TICKERS)}")
    print(f"  ✅ 이들 종목은 분석에서 제외됩니다.")
else:
    print(f"  ℹ️ 현재 수동 제외 종목이 설정되지 않았습니다.")
    print(f"  💡 제외가 필요한 경우 MANUAL_EXCLUDE_TICKERS 변수에 티커를 추가하세요.")

delisting_tickers = MANUAL_EXCLUDE_TICKERS

# 1. 저장된 티커 목록을 불러옵니다.
try:
    tickers_df = pd.read_csv(INPUT_FILENAME)
    tickers = tickers_df['Ticker'].tolist()
    print(f"\n'{INPUT_FILENAME}'에서 총 {len(tickers)}개 티커를 불러왔습니다.")
    
    # 📌 유효한 티커만 필터링 (문자열이 아닌 값 제거)
    original_count = len(tickers)
    tickers = [t for t in tickers if isinstance(t, str) and t.strip()]
    invalid_count = original_count - len(tickers)
    if invalid_count > 0:
        print(f"  ⚠️ 유효하지 않은 티커 {invalid_count}개를 제외했습니다.")
    
    # 📌 상장폐지 예정 종목 필터링
    if delisting_tickers:
        original_count = len(tickers)
        tickers = [t for t in tickers if t.upper() not in delisting_tickers]
        filtered_count = original_count - len(tickers)
        if filtered_count > 0:
            print(f"  ⚠️ 상장폐지 예정 종목 {filtered_count}개를 제외했습니다.")
        print(f"  ✅ 필터링 후 {len(tickers)}개 티커로 진행합니다.")
    else:
        print(f"  ✅ {len(tickers)}개 티커로 진행합니다.")
    
    # # ⚠️ 테스트 시 아래 주석을 풀어 100개만 실행하는 것을 권장합니다.
    # tickers = tickers[:10]
    # print(f"테스트를 위해 {len(tickers)}개로 제한하여 실행합니다.")
except FileNotFoundError:
    print(f"❌ 오류: '{INPUT_FILENAME}' 파일을 찾을 수 없습니다. 이전 단계를 먼저 실행해주세요.")
    tickers = []

# 팩터 데이터를 저장할 리스트
factor_data = []

if tickers:
    # 기간 설정 (오늘부터 3년 전까지)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=3*365)

    print(f"\n✅ 데이터 수집 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    print("\n✅ 각 주식의 팩터 원본 데이터를 수집합니다. (모멘텀/변동성은 1년, 3년 모두 계산)")

    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] '{ticker}' 처리 중...")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # 1차 필터링: 주식이 아니거나, 랭킹에 필수적인 정보가 없으면 제외
            if (info.get('quoteType') != 'EQUITY' or 
                info.get('priceToBook') is None or 
                info.get('beta') is None):
                print(f"  -> 주식이 아니거나 필수 정보(PBR, Beta)가 없어 제외합니다.")
                continue

            # 3년치 과거 주가 데이터를 가져옵니다.
            hist = stock.history(start=start_date, end=end_date)

            # 2차 필터링: 3년치 데이터가 충분하지 않으면 제외
            if len(hist) < 750:
                print(f"  -> 주가 데이터가 3년 미만이라 제외합니다.")
                continue

            # 팩터 값 계산
            beta_val = info.get('beta')
            pbr_val = info.get('priceToBook')
            market_cap_val = info.get('marketCap')
            roe_val = info.get('returnOnEquity')

            # 📌 수정된 부분: 1년 및 3년 모멘텀/변동성 모두 계산
            # 3년 모멘텀
            momentum_3y_val = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
            # 1년 모멘텀
            momentum_1y_val = (hist['Close'].iloc[-1] / hist['Close'].iloc[-252]) - 1
            
            # 3년 변동성
            volatility_3y_val = hist['Close'].pct_change().std()
            # 1년 변동성
            volatility_1y_val = hist['Close'].iloc[-252:].pct_change().std()

            # 총자산 증가율 계산
            balance_sheet = stock.balance_sheet
            if len(balance_sheet.columns) > 1 and 'Total Assets' in balance_sheet.index:
                asset_growth_val = (balance_sheet.loc['Total Assets'][0] - balance_sheet.loc['Total Assets'][1]) / abs(balance_sheet.loc['Total Assets'][1])
            else:
                asset_growth_val = None # 자산 정보 부족

            # 모든 데이터가 수집된 경우에만 최종 리스트에 추가
            all_factors = [beta_val, pbr_val, market_cap_val, roe_val, 
                           momentum_1y_val, momentum_3y_val, 
                           volatility_1y_val, volatility_3y_val, 
                           asset_growth_val]
            if all(v is not None for v in all_factors):
                factor_data.append({
                    "Ticker": ticker,
                    "Beta": beta_val,
                    "Momentum_1Y": momentum_1y_val,
                    "Momentum_3Y": momentum_3y_val,
                    "Value_PBR": pbr_val,
                    "Volatility_1Y": volatility_1y_val,
                    "Volatility_3Y": volatility_3y_val,
                    "Size_MarketCap": market_cap_val,
                    "Profitability_ROE": roe_val,
                    "Investment_AssetGrowth": asset_growth_val
                })
                print(f"  -> '{ticker}' 데이터 수집 완료.")
            else:
                print(f"  -> 일부 팩터 값이 누락되어 제외합니다.")

        except Exception as e:
            print(f"  -> '{ticker}' 처리 중 오류 발생: {e}")
        
        finally:
            # IP 차단을 피하기 위한 랜덤 지연 시간 (1~2초)
            time.sleep(random.uniform(1, 2))

print(f"\n✅ 3. 총 {len(factor_data)}개 주식의 데이터 수집 완료. CSV 파일로 저장합니다.")
if factor_data:
    all_stocks_df = pd.DataFrame(factor_data)
    all_stocks_df.to_csv(OUTPUT_FILENAME, index=False)
    print(f"  -> '{OUTPUT_FILENAME}' 저장 완료!")
else:
    print("\n수집된 데이터가 없어 CSV 파일을 생성하지 않습니다.")