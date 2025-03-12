
ROUGE-Recall :참조 요약본을 구성하는 단어들 중 모델 요약본의 단어들과 얼마나 많이 겹치는지 계산한 점수입니다.
ROUGE-Precision: 모델 요약본을 구성하는 단어들 중 참조 요약본의 단어들과 얼마나 많이 겹치는지 계산한 점수입니다.
ROUGE-N과 ROUGE-L은 비교하는 단어의 단위 개수를 어떻게 정할지에 따라 구분됩니다.
ROUGE-N은 unigram, bigram, trigram 등 문장 간 중복되는 n-gram을 비교하는 지표입니다.
ROUGE-1는 모델 요약본과 참조 요약본 간에 겹치는 unigram의 수를 비교합니다.
ROUGE-2는 모델 요약본과 참조 요약본 간에 겹치는 bigram의 수를 비교합니다.
ROUGE-L: LCS 기법을 이용해 최장 길이로 매칭되는 문자열을 측정합니다. n-gram에서 n을 고정하지 않고, 단어의 등장 순서가 동일한 빈도수를 모두 세기 때문에 보다 유연한 성능 비교가 가능합니다.

# ROUGE란?

- ROUGE (Recall-Oriented Understudy for Gisting Evaluation)는 자연어 처리(NLP)에서 텍스트 요약 성능을 평가하는 대표적인 지표
- 사람이 작성한 참조 요약본(reference summary) 과 모델이 생성한 모델 요약본(candidate summary) 사이의 겹치는 정도를 계산

# ROUGE-Recall & ROUGE-Precision

- ROUGE-Recall: 참조 요약본의 단어 중 모델 요약본에서 겹치는 단어의 비율을 측정
  - $\frac{\text{겹치는 단어 수}}{\text{참조 요약본의 단어 수}}$
- ROUGE-Precision: 모델 요약본의 단어 중 참조 요약본과 겹치는 단어의 비율을 측정
  - $\frac{\text{겹치는 단어 수}}{\text{모델 요약본의 단어 수}}$

# ROUGE-N

| 지표 | 설명 |
|------|------|
| **ROUGE-1** | unigram(1-gram) 단위로 비교 (단어 단위 겹침) |
| **ROUGE-2** | bigram(2-gram) 단위로 비교 (연속된 두 단어 겹침) |
| **ROUGE-3** | trigram(3-gram) 단위로 비교 |

- n-gram을 활용한 평가 방식으로, n개의 연속된 단어를 비교하여 유사도를 측정합니다.
- 계산 방식
  - ROUGE-1: 참조 요약본과 모델 요약본에서 **겹치는 단어 수를 비교**
  - ROUGE-2: 참조 요약본과 모델 요약본에서 **겹치는 연속된 두 단어 수를 비교**
- 예시
  - 참조 요약본: "기계 학습 모델은 데이터를 학습한다"
  - 모델 요약본: "기계 학습은 데이터를 분석한다"
    - **ROUGE-1 (unigram)**: "기계", "학습", "데이터" → 3개 겹침
    - **ROUGE-2 (bigram)**: "기계 학습", "데이터 학습" → 2개 겹침

# ROUGE-L (Longest Common Subsequence)

- ROUGE-L은 **LCS (Longest Common Subsequence, 최장 공통 부분 문자열) 기법**을 활용
- **연속되지 않아도 순서를 유지하는 단어들의 유사도를 측정**
- 특징
  - n-gram에서 n을 고정하지 않고, **단어의 등장 순서를 고려**
  - 보다 **유연한 성능 비교 가능** (예: 문장 내 단어 순서가 바뀌어도 유사성이 반영됨)
- 계산식:
  - LCS 길이를 기반으로 ROUGE-Recall과 ROUGE-Precision을 계산
- 예시:
  - 원본 문장: "`The` `cat` sat `on` the `mat`"
  - 요약 문장: "`The` `cat` is `on` `mat`"
    - LCS 찾기
      - LCS: "The cat on mat" (공통으로 등장하는 부분, 순서 유지)
      - LCS 길이 = 4
- ROUGE-L 특징 및 장점
  - 순서 정보 반영: 단어 매칭 기반 평가 방식(ROUGE-N)과 달리, LCS를 사용하여 문장 순서도 반영
  - 문장 길이 영향을 줄임: n-gram 기반 ROUGE-N과 비교해 단어 길이에 덜 민감함
  - 간단한 계산 방식: BLEU, METEOR 같은 지표보다 계산이 직관적
- 한계점
  - 문맥을 완전히 반영하지 못함 → 단순한 LCS 기반 평가라 의미적 유사성 측정 불가능
  - 동의어나 문장 구조 변경에 취약 → 의미는 같지만 어순이 다르면 낮은 점수 가능
