# 데이터 패밀리: "여정과 의미" 모의 세계

본 자료는 Rogers 등(2023, Journal of Personality and Social Psychology, https://doi.org/10.1037/pspa0000341)의 공개 자료 구조와 논문에 보고된 통계를 참조하여 씨앗 기반으로 생성한 모의 자료이며, 실제 참여자의 응답이 아니다.

생성 = `make_data_family.py` (씨앗 73 정본 ; 37 교차 검증). 변수 사전과 심는 구조의 정전 =
교재 데이터 설계 문서. 씨앗을 바꿔 실행하면 "다른 세계"를 얼마든지 만들 수 있다.

| 파일 | 시나리오 | 행 |
|---|---|---|
| journey_exp.csv | S1 실험(2집단·기저 포함) | 380 |
| journey_fac.csv | S2 요인(2×2)·사전-사후 | 448 |
| journey_svy.csv | S3 설문(조절·과통제 재료 포함) | 590 |
| journey_panel.csv | S4a 패널(3웨이브 long·이탈 포함) | 1,270 |
| journey_cohort.csv | S4b 반복 횡단(2015/2020/2025) | 1,800 |
| journey_ts.csv | S5 단절 시계열(주간 집계) | 104 |
| journey_coding.csv | S6 내용분석 코딩표(코더 2인) | 200 |
