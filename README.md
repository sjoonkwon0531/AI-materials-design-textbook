# AI 기반 첨단소재 탐색/설계/발견/최적화 교과서의 재현 코드와 그림 생성 코드

권석준(성균관대학교) 교수가 작업 중인 교과서 'AI 기반 첨단소재 탐색/설계/발견/최적화'(출판사 사회평론)의 코드 및 데이터 저장소입니다. 본문에 실린 Hands-on Lab의 Python 코드, 같은 Lab을 포팅한 Julia 코드, 그림과 그림 생성 스크립트, 연습문제 해답 검증 코드를 장별로 저장한 저장소입니다. 교과서 텍스트는 이 저장소에 포함되지 않았습니다.

Companion repository for the Korean textbook *AI-Driven Discovery, Design, and Optimization of Advanced Materials* by Seok Joon Kwon (Sungkyunkwan University). It contains the Python and Julia code for every hands-on lab, the figures and the scripts that draw them, and the code that checks the end-of-chapter exercise solutions. The manuscript itself is not included.

## 장 구성

| 폴더 | 장 | Python Lab | Julia | 그림 | 해답 코드 |
|---|---|---:|---:|---:|---:|
| `ch01` | 1. 수치해석 입문 | 23 | 5 | 3 | 59 |
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

### 환경

검증에 쓴 환경은 Python 3.11과 Julia 1.13.4입니다(자세한 결과는 `VERIFICATION.md`).

Python 환경은 두 개입니다. Ch.10의 MACE-MP-0 모델이 요구하는 `e3nn` 판(0.4.4)과 Ch.7 Lab이 쓰는 `e3nn` 판(0.6.0)이 서로 다르기 때문입니다.

| 환경 | 설치 | 정확한 판 | 쓰는 곳 |
|---|---|---|---|
| A | `pip install -r requirements.txt` | `lock/env-a.txt` | 아래 B를 뺀 모든 스크립트, Ch.10의 MACE와 CHGNet |
| B | `pip install -r requirements-b.txt` | `lock/env-b.txt` | `e3nn`을 직접 쓰는 Ch.7 스크립트(목록은 `requirements-b.txt`), Ch.10의 SevenNet과 MatterSim |

```bash
python3.11 -m venv venv-a && venv-a/bin/pip install -r requirements.txt
python3.11 -m venv venv-b && venv-b/bin/pip install -r requirements-b.txt
# 검증 때와 같은 판으로 맞추려면: venv-a/bin/pip install -r lock/env-a.txt
```

Ch.10에서 네 MLIP를 비교하는 스크립트는 두 환경을 오가며 실행합니다. 각 환경의 Python 경로를 환경 변수로 알려 줍니다.

```bash
export MLIP_PY_A=$PWD/venv-a/bin/python
export MLIP_PY_B=$PWD/venv-b/bin/python
```

### 실행 위치

스크립트는 **그 스크립트가 있는 폴더에서** 실행합니다. 다른 장의 파일은 `../../chNN/...`처럼 상대 경로로 참조하므로, 다른 폴더에서 실행하면 파일을 찾지 못합니다.

```bash
cd ch07/labs
python 71_backprop.py             # 예: 역전파를 손으로 구현해 autograd와 대조하는 Lab
cd ../../ch01/labs/blocks
python run_all.py                 # Ch.1의 코드 블록 23개를 차례로 실행
```

Ch.9의 평가 스크립트(`93_rollout_eval.py`, `95_compare.py`, `96_fno_ch_eval.py` 등)와 일부 해답 코드는 학습 스크립트가 만든 모델과 데이터 파일을 읽습니다. 데이터 생성, 학습, 평가 순서로 실행하십시오. 명령줄 인자가 필요한 해답 코드는 파일 첫머리의 설명에 인자가 적혀 있고, 저장된 `*.out` 파일 이름이 실제 쓴 인자를 가리킵니다(예: `s92_6_b30_per.out` ← `python s92_6.py b 30 0 600 2400 periodic`).

`*.out` 파일은 집필 당시 실제 실행한 출력입니다. 본문의 수치는 이 출력에서 왔으므로, 직접 실행한 결과와 비교할 수 있습니다. 난수 시드는 고정되어 있지만 BLAS, GPU, 라이브러리 판에 따라 마지막 자리가 조금 다를 수 있습니다. `ch04/labs/*.out`은 검증 실행에서 새로 만든 출력입니다.

### Julia

Julia 코드는 Julia 1.13.4에서 확인했습니다. 장마다 `chNN/julia/Project.toml`과 `Manifest.toml`이 있으므로, 처음 한 번 패키지를 받은 뒤 스크립트 폴더에서 실행합니다.

```bash
cd ch06/julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. lab_6_1.jl
```

Ch.7과 Ch.8의 Julia Lab은 Python Lab과 똑같은 표본을 쓰도록 학습·시험 인덱스를 파일에서 읽습니다(NumPy와 Julia의 난수열이 다르기 때문). 이 파일은 저장소의 `ch07/data/splits/`에 들어 있습니다. Julia가 읽는 데이터 표(`ch07/data/magpie_expt_gap.csv`, `67_features_clean.csv`, `xrd_peaks.csv`, `xrd_labels.csv`)도 함께 들어 있습니다. Python Lab은 같은 데이터를 Parquet과 NPZ로 읽습니다. 인덱스와 데이터 표는 모두 `ch07/data/export_splits.py`로 다시 만들 수 있습니다.

Python 모델을 부르는 Julia Lab(Ch.7의 `lab_7_22.jl`, Ch.10)은 PythonCall로 위의 Python 환경을 씁니다. conda를 따로 설치하지 않도록 다음처럼 실행합니다.

```bash
cd ch10/julia
JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=$MLIP_PY_A julia --project=. lab_10_2.jl
```

`lab_10_5.jl`은 MLIP 네 개를 모델마다 따로 실행한 뒤 결과를 합칩니다. `mace`와 `chgnet`은 `MLIP_PY_A`로, `sevennet`과 `mattersim`은 `MLIP_PY_B`로 실행하고, 마지막에 `julia --project=. lab_10_5.jl merge`를 실행합니다.

Optim.jl은 1.x로 고정했습니다(`[compat]`). 2.x에서는 Ch.9 Lab이 쓰는 `only_fg!`가 없어졌습니다.

다음 파일은 다른 Lab이 `include`하는 도우미 파일이라 단독으로 실행하지 않습니다. `ch06/julia/dscribe_bridge.jl`, `ch07/julia/ch7_common.jl`, `equiv_layers.jl`, `flux_tools.jl`, `mlip_common.jl`, `models77.jl`, `so3.jl`, `ch08/julia/bandgap83.jl`, `ch8_common.jl`, `film_env.jl`, `furnace.jl`, `sam.jl`, `ch09/julia/ch9_common.jl`, `fno.jl`, `pinn_common.jl`, `ch10/julia/ch10_common.jl`, `ch10_common2.jl`.

## 저장소에 넣지 않은 것

학습된 가중치(`*.pt`)와 중간 데이터(`*.npz`, `*.npy`)는 용량 때문에 넣지 않았습니다. 모두 Lab 스크립트로 다시 만들 수 있습니다.

- **Ch.7:** 에탄올 xTB 궤적은 `ch07/data/make_xtb_ethanol.py`로 만듭니다(`tblite` 필요). GFN1 라벨은 `ch07/data/74_make_gfn1_labels.py`로 만듭니다.
- **Ch.8:** `data/81_*` 결과 파일은 `labs/81_*` 스크립트를 실행하면 생깁니다.
- **Ch.9:** 학습과 시험 데이터는 `labs/93_data.py`, `94_data.py`, `95_ns_data.py`, `96_ch_data.py`로 만듭니다(합계 약 350 MB).
- **Ch.3:** `ch3_lab.py`, `sol_ch3_6C_network.py`, `lab_3_2.jl`, `lab_3_optimade.jl`은 Materials Project API(키 필요)나 OPTIMADE 서버에 접속합니다. 인터넷 연결이 있어야 실행됩니다.
- **Ch.10:** 데이터는 `labs/102_data.py`, `104_data.py`, `105_ft_data.py`로 만듭니다. MACE-MP-0 small 모델은 [ACEsuit/mace-mp releases](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model)에서 내려받아 `ch10/data/mace_mp0_small.model`로 저장합니다.

그림을 다시 그리면 `figures/`의 PNG를 덮어씁니다. 원본과 같은 모양을 얻으려면 Arial Narrow 글꼴이 설치되어 있어야 합니다.

## 라이선스

- **코드** (`*.py`, `*.jl`, `*.sh`): MIT License (`LICENSE`)
- **그림:** CC BY 4.0 (`LICENSE-figures.md`)
- 포함된 외부 데이터(예: Matbench `expt_gap`)는 원래 라이선스를 따릅니다.

## 인용

```
권석준, 'AI 기반 첨단소재 탐색/설계/발견/최적화' (출판사: 사회평론) 동반 코드 저장소, 2026.
```
