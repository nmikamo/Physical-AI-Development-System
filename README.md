# Physical AI Development System

2足歩行の人型ロボットを対象とした、モーションキャプチャーによる模倣学習・強化学習（**DAPG / RLPD / AWAC**）、**MuJoCo** 動作確認（MP4動画＋Three.js 3D骨格表示）、**alphaXiv** 論文スクレイピング＆テキストマイニング、**Amazon Bedrock** 要約・AI開発相談を統合した Web システムです。

## アーキテクチャ（AWS 大阪リージョン `ap-northeast-3`）
- **EC2 (`Physical-AI-Development-System`)**: `c6i.xlarge` / Ubuntu 24.04 LTS / FastAPI + Nginx + Linux Cron
- **S3 (`physical-ai-development-system-bucket`)**: バージョニング有効・Intelligent-Tiering・10年保管ライフサイクル
  - `papers/`: 収集済み論文メタデータ・要約
  - `text_mining/`: 月次テキストマイニング集計・2D密度マップデータ
  - `mocap/`: モーションキャプチャー入力データ（CSV / NPZ / BVH）
  - `report/`: 学習レポート（Train/Val Loss・タスク成功率）・MuJoCo評価MP4動画・3Dフレームデータ
- **Amazon Bedrock**: 論文日本語要約・Antigravity同等のAI学習相談および評価再指示アシスタント
- **IAM**: EC2用ロール（S3バケット＋Bedrock＋SSM）および既存ユーザー権限（`Developer` / `WORK`）

## 設計方針（関数型プログラミングベース＆機能集約）
- `domain/`: 不変データモデル（`frozen=True` dataclass）および純粋関数によるテキストマイニング・RL損失計算・診断ロジック
- `adapters/`: S3・Bedrock・alphaXivスクレイピング（`time.sleep >= 2.0s` レートリミット遵守）・MuJoCo実行のI/O境界分離
- `workflows/`: 月次論文パイプラインおよび模倣/強化学習ワークフロー
- `web/`: FastAPI + Plotly.js + Three.js による高速ダッシュボード（初回表示3秒以内）

## テスト実行
```bash
pytest -v
```
