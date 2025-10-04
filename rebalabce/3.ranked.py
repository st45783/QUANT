import pandas as pd
import numpy as np

# --- 설정값 ---
INPUT_FILENAME = "all_stocks_raw_factors.csv"
OUTPUT_SIMPLE_RANK_FILENAME = "top_50_simple_rank.csv"
OUTPUT_ZSCORE_RANK_FILENAME = "top_50_zscore_rank.csv"

print("✅ 파이프라인 3단계: 순위 산정 및 포트폴리오 선정을 시작합니다.")

# 1. 2단계에서 저장한 팩터 원본 데이터 파일을 불러옵니다.
try:
    df = pd.read_csv(INPUT_FILENAME)
    # 분석 전, 누락된 데이터가 있는 행은 제거합니다.
    df.dropna(inplace=True)
    print(f"'{INPUT_FILENAME}'에서 총 {len(df)}개 주식의 데이터를 불러왔습니다.")
except FileNotFoundError:
    print(f"❌ 오류: '{INPUT_FILENAME}' 파일을 찾을 수 없습니다. 이전 단계를 먼저 실행해주세요.")
    df = pd.DataFrame()

if not df.empty:
    # --- 팩터 특성 정의 ---
    # 값이 낮을수록 좋은 팩터들
    lower_is_better = [
        'Beta',
        'Value_PBR',
        'Volatility_1Y',
        'Volatility_3Y',
        'Size_MarketCap',
        'Investment_AssetGrowth'
    ]
    # 값이 높을수록 좋은 팩터들
    higher_is_better = [
        'Momentum_1Y',
        'Momentum_3Y',
        'Profitability_ROE'
    ]
    all_factors = lower_is_better + higher_is_better
    
    # =================================================================
    # 방법 1: 단순 순위 (Simple Rank) 기준
    # =================================================================
    print("\n✅ 방법 1: 단순 순위(Simple Rank) 계산을 시작합니다.")
    
    df_simple = df.copy()
    
    # 각 팩터의 순위를 계산하여 새로운 컬럼에 저장
    for factor in lower_is_better:
        df_simple[f'{factor}_Rank'] = df_simple[factor].rank(ascending=True)
    for factor in higher_is_better:
        df_simple[f'{factor}_Rank'] = df_simple[factor].rank(ascending=False)
        
    # 각 팩터 순위를 모두 더해 종합 순위 점수를 계산
    rank_columns = [f'{col}_Rank' for col in all_factors]
    df_simple['Composite_Rank_Score'] = df_simple[rank_columns].sum(axis=1)
    
    # 종합 점수가 낮은 순서(좋은 순서)대로 정렬
    final_simple_rank = df_simple.sort_values(by='Composite_Rank_Score', ascending=True)
    
    # 상위 50개 선정 및 파일 저장
    top_50_simple = final_simple_rank.head(50)
    top_50_simple.to_csv(OUTPUT_SIMPLE_RANK_FILENAME, index=False)
    print(f"  -> '{OUTPUT_SIMPLE_RANK_FILENAME}' 저장 완료!")

    # =================================================================
    # 방법 2: Z-스코어 (Z-Score) 기준
    # =================================================================
    print("\n✅ 방법 2: Z-스코어(Z-Score) 계산을 시작합니다.")
    
    df_zscore = df.copy()
    
    # 각 팩터의 Z-스코어를 계산
    for factor in all_factors:
        df_zscore[f'{factor}_Z'] = (df_zscore[factor] - df_zscore[factor].mean()) / df_zscore[factor].std()
        
    # 방향성 통일: 낮을수록 좋은 팩터의 Z-스코어에 -1을 곱함
    for factor in lower_is_better:
        df_zscore[f'{factor}_Z'] = df_zscore[f'{factor}_Z'] * -1
        
    # 방향성이 통일된 Z-스코어를 모두 더해 종합 점수를 계산
    z_score_columns = [f'{col}_Z' for col in all_factors]
    df_zscore['Composite_Z_Score'] = df_zscore[z_score_columns].sum(axis=1)
    
    # 종합 점수가 높은 순서(좋은 순서)대로 정렬
    final_zscore_rank = df_zscore.sort_values(by='Composite_Z_Score', ascending=False)
    
    # 상위 50개 선정 및 파일 저장
    top_50_zscore = final_zscore_rank.head(50)
    top_50_zscore.to_csv(OUTPUT_ZSCORE_RANK_FILENAME, index=False)
    print(f"  -> '{OUTPUT_ZSCORE_RANK_FILENAME}' 저장 완료!")

else:
    print("\n분석할 데이터가 없습니다.")

print("\n✅ 모든 작업이 완료되었습니다.")