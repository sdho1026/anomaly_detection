# 병렬 CNN 패치 임베딩 + Causal Adapter

핵심 모델은 CAUSAL_ADAPTER입니다.

- 전체 50,001-point 전압을 20배 FIR resampling하여 2,501 points로 입력합니다.
- patch/stride=32, CNN kernel 3·5 병렬 분기, hidden=64, causal Transformer 2층입니다.
- 사전학습은 shifted causal context + patch diffusion + self-only decoder입니다.
- Adapter는 사전학습에 넣지 않고 분류 파인튜닝에서 각 encoder 층 뒤에 추가합니다.
- Adapter: 64→8→64 residual MLP. Backbone을 고정하고 Adapter와 head만 학습합니다.
- 분류 head: 8개 시간 구간 평균 + global max pooling.
- 기존 비교군 BI_FULL, BI_ADAPTER도 보존했습니다.

## 공개용 변경
데이터 파일명, 파형 해시, 기존 고정 fold 목록은 공개하지 않습니다.
prepare_local_metadata.py가 사용자가 넣은 ZIP에서 이 파일들을 로컬로 생성합니다.
새로 생성한 fold는 과거 실험의 저장된 fold와 동일하다고 보장하지 않습니다.
따라서 과거 AUROC를 그대로 재현한 배포판으로 주장하지 않습니다.

## 실행
python prepare_local_metadata.py --data /path/to/data.zip
python checks.py
python cv.py train --data /path/to/data.zip --out /path/to/new_run --device cuda
python cv.py test --data /path/to/data.zip --out /path/to/new_run --device cuda

원본의 model.py, data.py, run.py, cv.py, checks.py 및 config.json은 보존했습니다.
공식 TimeDART의 완전한 재현이 아닌 NDT 적용 변형입니다.
