import pandas as pd
from finvizfinance.screener.overview import Overview

print("✅ 파이프라인 1단계: 미국 전체 티커 목록 수집을 시작합니다.")

try:
    foverview = Overview()
    all_tickers = []
    
    # 분석 대상 거래소 리스트
    exchanges = ['NASDAQ', 'NYSE', 'AMEX']
    
    # 3대 거래소의 티커를 모두 가져옵니다.
    for exchange in exchanges:
        print(f"...{exchange} 거래소의 티커 목록을 가져오는 중...")
        filters_dict = {'Exchange': exchange}
        foverview.set_filter(filters_dict=filters_dict)
        df_screener = foverview.screener_view()
        
        # 'Ticker' 열의 데이터를 리스트에 추가합니다.
        all_tickers.extend(df_screener['Ticker'].tolist())
    
    # 중복을 제거하고 알파벳순으로 정렬합니다.
    tickers = sorted(list(set(all_tickers)))
    
    print(f"\n총 {len(tickers)}개의 고유 티커를 수집했습니다.")

    # 수집된 티커 리스트를 Pandas DataFrame으로 변환합니다.
    tickers_df = pd.DataFrame(tickers, columns=['Ticker'])
    
    # DataFrame을 CSV 파일로 저장합니다.
    output_filename = "all_us_tickers.csv"
    tickers_df.to_csv(output_filename, index=False)
    
    print(f"✅ 성공: 전체 티커 목록이 '{output_filename}' 파일로 저장되었습니다.")

except Exception as e:
    print(f"\n❌ 오류: 티커 목록을 가져오는 데 실패했습니다: {e}")