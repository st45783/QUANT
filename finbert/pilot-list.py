import os
import subprocess
import platform
import pandas as pd
from transformers import pipeline
from edgar import Company, set_identity 
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ===================================================================
# 1. 신원 설정
# ===================================================================
set_identity("Seungtae Kim st45783@skku.edu")

# ===================================================================
# 2. GPU 선택 함수 (기존 코드 유지)
# ===================================================================
def select_device():
    # Mac M1/V를 사용하는지 체크
    if platform.system() == 'Darwin' and 'arm' in platform.machine().lower():
        print("Mac M1 환경 감지: MPS 사용")
        return 'mps'
    else:
        # 기존 GPU 선택 함수 호출 또는 디폴트 GPU 반환
        try:
            gpu_info_cmd = "nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits"
            result = subprocess.run(gpu_info_cmd.split(), capture_output=True, text=True, check=True)
            index_to_uuid = {int(index): uuid.strip() for index, uuid in (line.split(', ') for line in result.stdout.strip().split('\n') if line)}
            busy_apps_cmd = "nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits"
            result = subprocess.run(busy_apps_cmd.split(), capture_output=True, text=True, check=True)
            busy_uuids = {line.strip() for line in result.stdout.strip().split('\n') if line}
            for index, uuid in index_to_uuid.items():
                if uuid not in busy_uuids:
                    print(f"✅ 실행 중인 프로세스가 없는 GPU {index}번을 선택했습니다.")
                    return str(index)
            return "0"
        except Exception:
            print("✅ GPU 상태 확인 불가. 0번 GPU를 기본으로 선택합니다.")
            return "0"

# ===================================================================
# 3. MD&A 추출 함수 (기존 코드 유지)
# ===================================================================
def get_10q_mda_text(ticker):
    try:
        print(f"📋 {ticker}의 최근 10-Q 보고서를 검색하는 중...")
        company = Company(ticker)
        filing_10q = company.get_filings(form="10-Q").latest()
        if filing_10q is None:
            print(f"❗️ {ticker}의 10-Q 보고서를 찾을 수 없습니다.")
            return None
        print(f"✅ {filing_10q.form} 보고서를 찾았습니다 (제출일: {filing_10q.filing_date})")
        print("📄 MD&A 섹션 텍스트를 추출하는 중...")
        tenq_obj = filing_10q.obj()
        mda_text = tenq_obj['Item 2']
        if not mda_text:
             print(f"❗️ 해당 보고서에서 MD&A 섹션을 찾을 수 없습니다.")
             return None
        print(f"✅ MD&A 텍스트를 성공적으로 추출했습니다 (총 {len(mda_text):,}자)")
        return mda_text
    except Exception as e:
        print(f"❗️ 10-Q 보고서 처리 중 오류: {e}")
        return None

# ===================================================================
# 메인 실행 로직
# ===================================================================
device_choice = select_device()
if device_choice == 'mps':
    os.environ["DEVICE_TYPE"] = "mps"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = device_choice

print("모델을 로딩하는 중입니다...")
sentiment_analyzer = pipeline("text-classification", model="ProsusAI/finbert", device=0)
print("✅ 모델 로딩이 완료되었습니다.")

target_tickers = ["ABBV", "ACIC", "APP", "ASM", "CLS", "FTAI", "HCA", "HRTG", "IAG", "LMB", "MAMA", "MFH", "PLTR", "POWL", "PSIX", "SLNO", "SMMT", "STRL", "TSSI", "UAMY", "VRNA", "MONEY"]
  # 여러 티커를 리스트로 지정
# target_ticker = "psix"  # 기존 단일 티커 주석 처리

# 결과를 저장할 리스트 초기화
analysis_results = []

# 결과를 저장할 리스트 초기화
analysis_results = []

# 여러 티커에 대해 반복 처리
for ticker in target_tickers:
    print(f"\n{'='*60}")
    print(f"📊 {ticker.upper()} 분석 시작")
    print(f"{'='*60}")
    
    mda_full_text = get_10q_mda_text(ticker)

    if not mda_full_text:
        print(f"⚠️  {ticker.upper()}의 MD&A 텍스트를 가져오지 못해 해당 티커를 건너뜁니다.")
        # 실패한 경우에도 결과에 추가 (None 값으로)
        analysis_results.append({
            'ticker': ticker.upper(),
            'positive': None,
            'neutral': None,
            'negative': None,
            'polarity_score': None,
            'filtered_positive': None,
            'filtered_neutral': None,
            'filtered_negative': None
        })
        continue

    print("\n⚙️  긴 텍스트를 분석 가능한 조각으로 분할하는 중...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
    text_chunks = text_splitter.split_text(mda_full_text)
    print(f"✅ 텍스트가 총 {len(text_chunks)}개의 조각으로 분할되었습니다.")

    print(f"\n🔬 {len(text_chunks)}개의 모든 조각에 대한 감성 분석을 시작합니다...")
    results = sentiment_analyzer(text_chunks, top_k=None)
    print("✅ 모든 조각 분석 완료!")

    # --- (기존 결과 종합 로직) ---
    total_scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
    total_chunks = len(text_chunks)
    if total_chunks == 0:
        print(f"❗️ {ticker.upper()}의 분석할 텍스트 조각이 없습니다. 해당 티커를 건너뜁니다.")
        analysis_results.append({
            'ticker': ticker.upper(),
            'positive': None,
            'neutral': None,
            'negative': None,
            'polarity_score': None,
            'filtered_positive': None,
            'filtered_neutral': None,
            'filtered_negative': None
        })
        continue

    for chunk_result in results:
        for score_info in chunk_result:
            label = score_info['label'].lower()
            score = score_info['score']
            if label in total_scores:
                total_scores[label] += score

    average_scores = {label: score / total_chunks for label, score in total_scores.items()}

    print(f"\n--- [{ticker.upper()} 10-Q MD&A] 전체 텍스트 평균 감성 점수 ---")
    print(f"📊 전체 {total_chunks}개 조각 기준:")
    print(f"  - 긍정 (Positive) 🟢: {average_scores['positive']:.4f} ({average_scores['positive']*100:.1f}%)")
    print(f"  - 부정 (Negative) 🔴: {average_scores['negative']:.4f} ({average_scores['negative']*100:.1f}%)")
    print(f"  - 중립 (Neutral)  ⚪️: {average_scores['neutral']:.4f} ({average_scores['neutral']*100:.1f}%)")

    # ===================================================================
    # ⭐ 감성 신호 부각을 위한 추가 분석
    # ===================================================================

    print(f"\n--- 감성 신호 부각 분석 결과 ---")

    # --- 방법 1: 중립을 제외한 감성 양극성(Polarity) 점수 계산 ---
    pos_score = average_scores['positive']
    neg_score = average_scores['negative']
    polarity_score = pos_score / (pos_score + neg_score) if (pos_score + neg_score) > 0 else 0
    print(f"📈 감성 양극성(Polarity) 점수: {polarity_score:.4f}")
    print(f"   (해석: 중립을 제외한 감성 중 긍정이 차지하는 비율이 {polarity_score*100:.1f}%임을 의미)")

    # --- 방법 2: 감성 조각 필터링 후 평균 재계산 ---
    sentimental_scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
    sentimental_chunk_count = 0
    for chunk_result in results:
        # 각 조각에서 가장 점수가 높은 레이블을 찾음
        dominant_sentiment = max(chunk_result, key=lambda x: x['score'])
        
        # 지배적인 감성이 '중립'이 아닌 경우에만 집계
        if dominant_sentiment['label'].lower() != 'neutral':
            sentimental_chunk_count += 1
            for score_info in chunk_result:
                label = score_info['label'].lower()
                sentimental_scores[label] += score_info['score']

    if sentimental_chunk_count > 0:
        filtered_average_scores = {
            label: score / sentimental_chunk_count for label, score in sentimental_scores.items()
        }
        print(f"\n📊 중립 제외 {sentimental_chunk_count}개 조각 기준 평균 감성:")
        print(f"  - 긍정 (Positive) 🟢: {filtered_average_scores['positive']:.4f} ({filtered_average_scores['positive']*100:.1f}%)")
        print(f"  - 부정 (Negative) 🔴: {filtered_average_scores['negative']:.4f} ({filtered_average_scores['negative']*100:.1f}%)")
        print(f"  - 중립 (Neutral)  ⚪️: {filtered_average_scores['neutral']:.4f} ({filtered_average_scores['neutral']*100:.1f}%)")
    else:
        print("\n📊 중립을 제외한 감성적인 조각을 찾을 수 없습니다.")
        filtered_average_scores = {'positive': None, 'negative': None, 'neutral': None}

    # 결과를 리스트에 추가
    analysis_results.append({
        'ticker': ticker.upper(),
        'positive': average_scores['positive'],
        'neutral': average_scores['neutral'], 
        'negative': average_scores['negative'],
        'polarity_score': polarity_score,
        'filtered_positive': filtered_average_scores['positive'],
        'filtered_neutral': filtered_average_scores['neutral'],
        'filtered_negative': filtered_average_scores['negative']
    })

print("\n" + "="*60)
print("🎯 모든 티커 분석 완료!")
print("="*60)

# ===================================================================
# CSV 파일로 결과 저장
# ===================================================================
print("\n📊 분석 결과를 CSV 파일로 저장 중...")

# DataFrame 생성
df = pd.DataFrame(analysis_results)

# 컬럼명을 사용자가 요청한 형태로 변경
df.columns = [
    'Ticker',
    'Positive',
    'Neutral', 
    'Negative',
    'Polarity_Score',
    'Filtered_Positive',
    'Filtered_Neutral',
    'Filtered_Negative'
]

# CSV 파일명 생성 (현재 날짜 포함)
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"finbert_sentiment_analysis_{timestamp}.csv"

# CSV 파일 저장
df.to_csv(csv_filename, index=False, float_format='%.4f')

print(f"✅ 분석 결과가 '{csv_filename}' 파일로 저장되었습니다.")
print("\n📋 결과 요약:")
print(df.to_string(index=False, float_format='%.4f'))

print("\n📖 컬럼 설명:")
print("  - Ticker: 종목 티커")
print("  - Positive: 긍정 감성 평균 점수")
print("  - Neutral: 중립 감성 평균 점수") 
print("  - Negative: 부정 감성 평균 점수")
print("  - Polarity_Score: 감성 양극성 점수 (긍정/(긍정+부정))")
print("  - Filtered_Positive: 중립 제외한 긍정 감성 점수")
print("  - Filtered_Neutral: 중립 제외한 중립 감성 점수")
print("  - Filtered_Negative: 중립 제외한 부정 감성 점수")

print("----------------------------------------------------------")

# ===================================================================
# ⭐ 신호 부각을 위한 추가 분석 (이 부분을 추가)
# ===================================================================

print(f"\n--- 감성 신호 부각 분석 결과 ---")

# --- 방법 1: 중립을 제외한 감성 양극성(Polarity) 점수 계산 ---
pos_score = average_scores['positive']
neg_score = average_scores['negative']
polarity_score = pos_score / (pos_score + neg_score) if (pos_score + neg_score) > 0 else 0
print(f"📈 감성 양극성(Polarity) 점수: {polarity_score:.4f}")
print(f"   (해석: 중립을 제외한 감성 중 긍정이 차지하는 비율이 {polarity_score*100:.1f}%임을 의미)")

# --- 방법 2: 감성 조각 필터링 후 평균 재계산 ---
sentimental_scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
sentimental_chunk_count = 0
for chunk_result in results:
    # 각 조각에서 가장 점수가 높은 레이블을 찾음
    dominant_sentiment = max(chunk_result, key=lambda x: x['score'])
    
    # 지배적인 감성이 '중립'이 아닌 경우에만 집계
    if dominant_sentiment['label'].lower() != 'neutral':
        sentimental_chunk_count += 1
        for score_info in chunk_result:
            label = score_info['label'].lower()
            sentimental_scores[label] += score_info['score']

if sentimental_chunk_count > 0:
    filtered_average_scores = {
        label: score / sentimental_chunk_count for label, score in sentimental_scores.items()
    }
    print(f"\n📊 중립 제외 {sentimental_chunk_count}개 조각 기준 평균 감성:")
    print(f"  - 긍정 (Positive) 🟢: {filtered_average_scores['positive']:.4f} ({filtered_average_scores['positive']*100:.1f}%)")
    print(f"  - 부정 (Negative) 🔴: {filtered_average_scores['negative']:.4f} ({filtered_average_scores['negative']*100:.1f}%)")
    print(f"  - 중립 (Neutral)  ⚪️: {filtered_average_scores['neutral']:.4f} ({filtered_average_scores['neutral']*100:.1f}%)")
else:
    print("\n📊 중립을 제외한 감성적인 조각을 찾을 수 없습니다.")

print("----------------------------------------------------------")

