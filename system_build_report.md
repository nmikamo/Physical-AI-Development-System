# Physical AI Development System 構築・検証レポート

## 1. 目的・前提
2足歩行の人型ロボットを対象に、モーションキャプチャーデータを用いた模倣学習および強化学習（**DAPG / RLPD / AWAC**）を行い、**MuJoCo** での動作確認・学習曲線・評価レポートを Web アプリで可視化するシステムを AWS（大阪リージョン `ap-northeast-3`）上に構築した。

---

## 2. 構築済み AWS インフラ構成（大阪リージョン `ap-northeast-3`）

| リソース種別 | リソース名 / ID | 設定・仕様 |
| :--- | :--- | :--- |
| **VPC（既存）** | `vpc-0a53a29051426716a` | CIDR `172.31.0.0/16` |
| **Publicサブネット（既存）** | `subnet-08e935c1f3bde332f` | AZ `ap-northeast-3a` / `172.31.64.0/20` |
| **IGW・ルートテーブル（既存）** | `igw-0827e3b89801c3a61` / `rtb-01c1d55327e584ddc` | `0.0.0.0/0 -> IGW` 設定済み |
| **EC2** | `Physical-AI-Development-System` (`i-03b69f02834e3c1fa`) | `c6i.xlarge` (4vCPU/8GB) / Ubuntu 24.04 LTS / EBS 50GB gp3 / Public IP `13.208.128.87` |
| **セキュリティグループ** | `Physical-AI-Development-System-sg` (`sg-0926fe082c2281a4b`) | TCP `22, 80, 443, 8000, 8501` (`0.0.0.0/0`) |
| **S3 バケット** | `physical-ai-development-system-bucket` | バージョニング有効、Intelligent-Tiering、10年（3650日）保持ポリシー、`papers/`・`text_mining/`・`mocap/`・`report/` |
| **IAM（EC2ロール）** | `Physical-AI-Development-System-ec2-role` | `physical-ai-development-system-bucket` への S3 読み書き権限 + Bedrock + SSM |
| **IAM（既存ユーザー）** | `Developer` / `WORK` | 開発技術者（フル権限）/ 閲覧者（`AmazonEC2ReadOnlyAccess`）を継続利用 |

---

## 3. アプリケーションアーキテクチャ（関数型プログラミングベース）

- **[models.py](src/physical_ai_system/domain/models.py)**: `frozen=True` の不変データクラスでドメインモデルを定義。
- **[text_mining.py](src/physical_ai_system/domain/text_mining.py)**: TF-IDF → TruncatedSVD 2D射影 → KMeans クラスタリング → 2D KDE 等高線密度グリッド・月次集計を純粋関数として実装。
- **[rl_algorithms.py](src/physical_ai_system/domain/rl_algorithms.py)**: PyTorch による **DAPG / RLPD / AWAC** 損失計算・ジャーク平滑度・過学習診断を純粋関数として実装。
- **[advisor.py](src/physical_ai_system/domain/advisor.py)**: Antigravity 同等のアルゴリズム推薦・評価フィードバックからのハイパーパラメータ自動調整ロジック。
- **[alphaxiv_scraper.py](src/physical_ai_system/adapters/alphaxiv_scraper.py)**: `time.sleep(2.0)` 以上のウェイトを厳守した `alphaxiv.org` 論文収集アダプタ（`cron` により毎月1日 03:00 に自動実行）。
- **[mujoco_engine.py](src/physical_ai_system/adapters/mujoco_engine.py)**: 2足歩行ヒューマノイドモデル、CSV/NPZ/BVH パーサー、OSMesa オフスクリーン MP4 動画生成、Three.js 用 3D 骨格フレーム出力。
- **[app.py](src/physical_ai_system/web/app.py)**: FastAPI + Plotly.js + Three.js による3タブ構成 Web アプリ（Nginx リバースプロキシ経由でポート80公開）。

---

## 4. 検証結果と考察

1. **自動テスト (`pytest -v`)**:
   - 全 9 件（API・MuJoCo 学習＆MP4/3D出力・DAPG/RLPD/AWAC 損失＆過学習診断・テキストマイニング）が **6.70秒で全合格**。
2. **初回表示応答時間（非機能要件「3秒以内」）**:
   - 外部クライアントからの `http://13.208.128.87/` 応答時間: **220 ms**、`/api/health`: **428 ms** を記録し、要件を大幅に満たした。
3. **初回実データ生成＆S3永続化**:
   - `alphaxiv.org` / arXiv 連携により 14 件の最新論文を取得・テキストマイニングし、`papers/` および `text_mining/` へ保存済み。
   - `DAPG` 初期学習（15 エポック）を実行し、`report/run_initial_dapg_sample/` 配下に `evaluation.mp4`（103 KB）・`frames_3d.json`（195 KB）・`report.json` を保存済み。
4. **Amazon Bedrock に関する確認事項**:
   - Bedrock API への IAM 疎通は正常だが、初回実行時に `Model use case details have not been submitted for this account`（Anthropic 社の利用申請フォーム未提出）が確認された。現在はローカルの決定論的要約・アドバイザーへ自動フォールバックして全機能が正常動作する。

---

## 5. 今後のアクション / 推奨事項

1. **Amazon Bedrock の Anthropic 申請（任意・30秒）**:
   - AWS コンソール（大阪リージョン `ap-northeast-3`）の **Amazon Bedrock → プレイグラウンド（またはモデルカタログ）** から Claude Sonnet / Haiku を1回選択し、Anthropic 利用用途フォーム（Use case details）を送信すると、約15分後に Web アプリ内の論文要約と AI 相談が自動的に Claude へ切り替わります。
2. **Web アプリへのアクセス**:
   - `http://13.208.128.87/` にアクセスして、テキストマイニングマップのクリック遷移・AI相談・MuJoCo動画/3D骨格表示をご確認ください。
   - ※EC2 は動的パブリック IP のため、インスタンスを停止・再起動した場合は AWS コンソールまたは `aws ec2 describe-instances` で新しい IP をご確認ください。
