# Devin Proto Mabataki

アニメキャラクターの目パチ・口パクアニメーション生成エンジン

## 概要

1枚の画像から目パチ・口パクアニメーションを生成・管理するプロトタイプエンジンです。
TalkingHeadsとControlNetを使用して、自然なアニメーション生成を実現します。

## 特徴

- 目パチアニメーション生成
  - 自然な瞬きのタイミング制御
  - 透過処理対応
- 口パクアニメーション生成
  - TalkingHeadsによる自然な口の動き
  - 透過処理対応

## セットアップ

```bash
# 環境構築
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 実行
python src/app.py
```

## 使用方法

1. Webブラウザで開く
2. 元となる画像をアップロード
3. 「アニメーション生成」ボタンをクリック
4. 生成された目パチ・口パクアニメーションを確認

## 技術構成

- Gradio: UIインターフェース
- ControlNet: 画像生成・変換
- TalkingHeads: 口パクアニメーション生成
- MediaPipe: 顔特徴点検出
