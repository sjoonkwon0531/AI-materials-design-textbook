# AI 기반 첨단소재 탐색·설계·발견·최적화: 코드와 그림

권석준(성균관대학교)의 교과서 「AI 기반 첨단소재 탐색·설계·발견·최적화」의 동반 저장소입니다. 본문에 실린 Hands-on Lab의 Python 코드, 같은 Lab을 옮긴 Julia 코드, 그림과 그림 생성 스크립트, 연습문제 해답 검증 코드를 장별로 모았습니다. 원고는 이 저장소에 포함되지 않습니다.

Companion repository for the Korean textbook *AI-Driven Discovery, Design, and Optimization of Advanced Materials* by Seok Joon Kwon (Sungkyunkwan University). It contains the Python and Julia code for every hands-on lab, the figures and the scripts that draw them, and the code that checks the end-of-chapter exercise solutions. The manuscript itself is not included.

## 장 구성

| 폴더 | 장 | Python Lab | Julia | 그림 | 해답 코드 |
|---|---|---:|---:|---:|---:|
| `ch01` | 1. 수치해석 입문 | 70 | 5 | 3 | 59 |
| `ch02` | 2. 미분방정식의 수치해법 | 10 | 16 | 18 | 50 |
| `ch03` | 3. 계산소재과학 개관 | 1 | 4 | 8 | 4 |
| `ch04` | 4. 양자·원자 스케일: DFT와 분자동역학 | 7 | 7 | 15 | 24 |
| `ch05` | 5. 메조스케일과 연속체 방법론 | 29 | 18 | 17 | 44 |
| `ch06` | 6. 고전적 머신러닝과 소재 기술자 | 27 | 26 | 25 | 32 |
| `ch07` | 7. 딥러닝과 Geometric Prior | 29 | 29 | 26 | 23 |
| `ch08` | 8. 강화학습과 모델 강건성 | 25 | 21 | 16 | 27 |
| `ch09` | 9. PINN, Neural Operator, Prior-Fitted PDE 풀이 | 38 | 28 | 23 | 24 |
| `ch10` | 10. MLIP: DFT를 O(N)으로 | 30 | 15 | 15 | 36 |

각 장 폴더의 구성은 다음과 같습니다.

```
chNN/
  labs/            본문 Hands-on Lab의 Python 코드와 실행 출력(*.out)
  julia/           같은 Lab의 Julia 대응 코드
  figures/         본문 그림(PNG)과, Ch.5 이후는 그림 생성 스크립트(make_*.py)
  figure_scripts/  Ch.1–4의 그림 생성 스크립트
  solutions/       연습문제 해답 검증 코드와 출력
  data/            Lab 입력 데이터(작은 텍스트, 표 형식 파일만)
common/bookstyle.py  모든 그림이 공유하는 스타일(Arial Narrow, 색, 패널 라벨)
```

## 실행 방법

스크립트는 **해당 장 폴더에서** 실행합니다. 다른 장의 파일은 `../chNN/...` 상대 경로로 참조합니다.

```bash
pip install -r requirements.txt
cd ch07
python labs/71_backprop.py        # 예: 역전파를 손으로 구현해 autograd와 대조하는 Lab
```

`*.out` 파일은 집필 당시 실제 실행한 출력입니다. 본문의 수치는 이 출력에서 왔으므로, 직접 실행한 결과와 비교할 수 있습니다. 난수 시드는 고정되어 있지만 BLAS, GPU, 라이브러리 판에 따라 마지막 자리가 조금 다를 수 있습니다.

Julia 코드는 Julia 1.10 이상에서 확인했습니다. 쓰는 패키지는 주로 `LinearAlgebra`, `Statistics`, `Random`, `Printf`, `FFTW`, `ForwardDiff`, `Zygote`, `Lux`/`Flux`, `NPZ`, `CSV`, `DataFrames`, `OrdinaryDiffEq` 입니다.

## 저장소에 넣지 않은 것

학습된 가중치(`*.pt`)와 중간 데이터(`*.npz`, `*.npy`)는 용량 때문에 넣지 않았습니다. 모두 Lab 스크립트로 다시 만들 수 있습니다.

- **Ch.7:** 에탄올 xTB 궤적은 `ch07/data/make_xtb_ethanol.py`로 만듭니다(`tblite` 필요). GFN1 라벨은 `ch07/data/74_make_gfn1_labels.py`로 만듭니다.
- **Ch.8:** `data/81_*` 결과 파일은 `labs/81_*` 스크립트를 실행하면 생깁니다.
- **Ch.9:** 학습과 시험 데이터는 `labs/93_data.py`, `94_data.py`, `95_ns_data.py`, `96_ch_data.py`로 만듭니다(합계 약 350 MB).
- **Ch.10:** 데이터는 `labs/102_data.py`, `104_data.py`, `105_ft_data.py`로 만듭니다. MACE-MP-0 small 모델은 [ACEsuit/mace-mp releases](https://github.com/ACEsuit/mace-mp/releases)에서 내려받아 `ch10/data/`에 둡니다.

그림을 다시 그리면 `figures/`의 PNG를 덮어씁니다. 원본과 같은 모양을 얻으려면 Arial Narrow 글꼴이 설치되어 있어야 합니다.

## 라이선스

- **코드** (`*.py`, `*.jl`, `*.sh`): MIT License (`LICENSE`)
- **그림:** CC BY 4.0 (`LICENSE-figures.md`)
- 포함된 외부 데이터(예: Matbench `expt_gap`)는 원래 라이선스를 따릅니다.

## 인용

```
권석준, 「AI 기반 첨단소재 탐색·설계·발견·최적화」 동반 코드 저장소, 2026.
```
