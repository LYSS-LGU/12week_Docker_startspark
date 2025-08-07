#!/bin/bash
# run_analysis.sh

# 스크립트 실행 중 오류가 발생하면 즉시 중단
set -e

echo "--- 1. 데이터 분석 스크립트 실행 시작 ---"
spark-submit pet_supplies_analysis.py

echo "\n--- 2. 예측 모델 스크립트 실행 시작 ---"
spark-submit pet_supplies_prediction_model.py

echo "\n--- 모든 분석 완료 ---"