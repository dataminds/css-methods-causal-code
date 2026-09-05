# 사회과학 통계와 인과추론 · 실습 데이터와 코드

책 「사회과학 통계와 인과추론: 계산은 AI가, 판단은 당신이」(계산사회과학 방법 총서 기초 「첫걸음」 추론 축)의 공식 실습 저장소입니다.

- 책 소개: https://grow.minds.kr/textbooks/css-methods/causal/
  (본문은 출판 준비 기간 동안 열람 계정을 받은 분께만 열립니다. 이 저장소의 데이터·코드는 누구나 씁니다.)
- 풀이와 채점 기준: https://grow.minds.kr/textbooks/css-methods/causal/solutions/
- 자매서 「계산으로 보는 사회」(모형으로 세계를 짓는 쪽): https://grow.minds.kr/textbooks/css-methods/vol0/
- 지은이: 안도현 (Ahn, Dohyun)

⭐ 이 책은 **선수 지식을 전제하지 않습니다.** 통계를 한 번도 안 배웠어도 처음부터 읽을 수 있습니다.

## 이 저장소가 담은 것

실습 전체가 하나의 모의 세계 "여정과 의미"에서 돕니다. 같은 연구질문(자기 삶의 이야기를 영웅의 여정으로 보는 것이 삶의 의미를 높이는가)을 실험·요인·설문·패널·반복 횡단·시계열·내용분석으로 다시 묻는 일곱 벌의 데이터가 들어 있고, 전부 씨앗 기반 생성이라 정답을 알고 채점까지 할 수 있습니다.

> 본 자료는 Rogers 등(2023, Journal of Personality and Social Psychology, https://doi.org/10.1037/pspa0000341)의 공개 자료 구조와 논문에 보고된 통계를 참조하여 씨앗 기반으로 생성한 모의 자료이며, 실제 참여자의 응답이 아닙니다.

## 실행: 두 갈래

### 갈래 1 · 설치 없이, 브라우저에서 (권장 시작점)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dataminds/css-methods-causal-code/blob/main/colab_quickstart.ipynb)

위 배지를 누르면 Google Colab(무료 ; 구글 계정 필요)에서 시작 노트북이 열립니다. 셀을 위에서부터 실행(▶ 또는 Shift+Enter)하면 설치 없이 일곱 벌 데이터를 불러와 책의 첫 수치들을 직접 재현할 수 있습니다.

**장별 실습 노트북**은 `notebooks/` 폴더에 있습니다. 본문 21장 가운데 코드가 있는 18장(ch02~ch06·ch08·ch09·ch11~ch21) + 기초 두 단위(b1·b2) + 시나리오 여섯(s1~s6) + 코드 읽기 = **27편**입니다. 각 장의 핵심 수치를 코랩에서 직접 재현·검증합니다.

⚠ **1·7·10장에는 노트북이 없습니다.** 코드가 없는 장이라 그렇습니다. 2부(인과의 논리)는 평균과 나눠 보기만으로 가고, 그것이 이 책의 설계입니다.

코랩에서 열려면 아래 주소의 `{장}`을 바꾸세요:

```
https://colab.research.google.com/github/dataminds/css-methods-causal-code/blob/main/notebooks/ch07.ipynb
```

**코드 읽기 노트북**(`notebooks/code_reading.ipynb`)은 이 책의 목적 — AI가 준 코드를 읽고 판단하기 — 을 직접 훈련합니다. AI 코드의 조용한 오류 여섯 가지(엉뚱한 집단·엉뚱한 열·표본 결측·경계 조건·부호 방향·통계 기본값)를 버그판·정답판 나란히 돌려 사냥합니다. 전부 crash 없이 그럴듯한 다른 숫자를 내는 오류라, 출력만으로는 안 걸리고 코드를 읽어야 잡힙니다.

[![코드 읽기 노트북 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dataminds/css-methods-causal-code/blob/main/notebooks/code_reading.ipynb)

### 갈래 2 · 내 컴퓨터에 설치 (10분)

1. 파이썬 3.11 이상 설치 (python.org ; "Add Python to PATH" 체크)
2. 이 저장소 내려받기: 우측 상단 Code → Download ZIP (또는 `git clone`)
3. 터미널에서 압축 푼 폴더로 이동 후:

```bash
pip install -r requirements.txt
python -c "import pandas, statsmodels; print('준비 끝')"
```

편집기는 아무거나 되지만 골라야 한다면 VS Code(무료)를 권합니다. 막히면 오류 메시지를 통째로 생성 AI에게 붙여넣고 물어보세요. 이 책의 표준 작업 절차입니다(1장).

## 구성

| 파일/폴더 | 내용 | 책 연결 |
|---|---|---|
| `notebooks/` | 실습 노트북 27편(장별 18 + 기초 b1·b2 + 시나리오 6 + 코드 읽기) : 코랩에서 바로 열어 재현·검증·버그 사냥 | 각 장 실습 · 2장 §2.4 |
| `make_notebooks.py` | 노트북 생성기(`--verify`로 전 노트북 코드 실행 점검) | 배포 게이트 |
| `data/` | 모의 세계 일곱 벌 CSV + 데이터 사전 README | 2장·전 시나리오 |
| `make_data_family.py` | 데이터 생성기 (자체 검증 배터리 내장). 씨앗을 바꾸면 "다른 세계"가 무한히 나온다 | 6장 표본 요동·S1·S2 검정력 |
| `make_book_figures.py` | 책의 그림 전부를 만드는 스크립트 (`figures/`에 저장) | 부록 C |
| `check_book_claims.py` | 책 본문의 모든 수치를 데이터·코드로 재현·대조하는 배터리 | 부록 B 검증 로그 |
| `check_anonymity.py` | 심사에 낼 저장소에서 신원이 새는 곳을 훑는 점검기 | 21장 §21.2 |
| `journey_stories-예문20.md` | 내용분석 코딩 연습용 자작 예문 20편 + 정답표 | 시나리오 S6 |
| `worksheets/` | 검증 로그 양식 · 아홉 단계 체크리스트 (논문 쓸 때 들고 가는 낱장) | 부록 B·D |

## 두 가지 규율

**재현성 = 씨앗.** 모든 확률적 결과는 씨앗(seed)을 명시합니다. 이 책의 기본 씨앗은 73, 교차 확인은 37입니다(서로 뒤집은 소수). 책의 수치를 재현하려면 반드시 씨앗 73, 표본 요동을 체험하려면 아무 씨앗이나.

**검증 = 코드로.** 이 책은 손계산을 검증 수단으로 요구하지 않습니다. 계산은 기계의 일이고, 연구자의 일은 어느 계산을 시켜 어떻게 교차시킬지입니다. 검증 3종(독립 재계산·극단값·재실행)은 전부 코드로 하며, 양식은 `worksheets/검증로그-양식.md`에 있습니다.

## 본문 수치를 스스로 검증하기

책이 인용하는 모든 수치는 이 저장소의 데이터·코드에서 재현됩니다. 직접 확인하려면:

```bash
python check_book_claims.py            # 전체(수 분)
python check_book_claims.py --fast     # 무거운 시뮬 생략
```

"본문-실물 대조 전 항목 일치"가 나오면 책의 수치와 이 저장소의 실측이 하나라는 뜻입니다. 웹판은 배포 때마다 이 배터리를 통과해야만 갱신됩니다.

## 심사에 낼 익명판 만들기 (21장 §21.2)

이 저장소를 자기 연구의 본으로 쓰는 분을 위해 점검기를 함께 둡니다.

```bash
python check_anonymity.py . --name "홍길동" "Hong, Gildong" --org 제주대
```

커밋 이력의 저자 이름, 노트북 출력에 굳은 절대 경로, 라이선스의 저작권자처럼 **본문이 아니라 곁으로 새는 곳**을 훑습니다. 이 저장소를 그대로 훑으면 수십 건이 걸리는데, 그중 커밋 이력 3건은 **파일을 다 고쳐도 안 없어집니다.** 익명판은 이력을 옮기지 말고 새로 만드세요(파일만 복사해 `git init` 후 커밋 하나).

⛔ 통과가 익명을 증명하지는 않습니다. 아는 무늬만 찾습니다. 걸린 것이 있으면 확실히 익명이 아니라는 쪽만 확실합니다.

## 라이선스

- 코드·데이터: 이 저장소의  파일(MIT)이 정합니다.
- 책 본문: 저작권 보유 (무단 복제·배포 금지)

질문·오류 제보: help@minds.kr
