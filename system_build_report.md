# Physical AI Development System 構築・検証レポート（v2 クオリティ＆操作性強化版）

## 1. 目的・前提
2足歩行の人型ロボットを対象に、モーションキャプチャーデータを用いた模倣学習および強化学習（**DAPG / RLPD / AWAC**）を行い、**MuJoCo** での動作確認・学習曲線・評価レポートを Web アプリで可視化するシステムを AWS（大阪リージョン `ap-northeast-3`）上に構築した。さらにデータ品質・物理シミュレーション安定性・Web UI の直感的な操作性を全面的にアップグレードした。

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
- **[text_mining.py](src/physical_ai_system/domain/text_mining.py)**: TF-IDF → ゼロ中心化 TruncatedSVD 2D射影（`[-1, 1]` 全域展開） → KMeans クラスタリング（重複排除・日本語10アーキタイプ） → 56×56 2D KDE 等高線密度グリッド・月次集計・ロボティクス論文フィルタを純粋関数として実装。
- **[rl_algorithms.py](src/physical_ai_system/domain/rl_algorithms.py)**: PyTorch による **DAPG / RLPD / AWAC** 損失計算・ジャーク平滑度・過学習診断を純粋関数として実装。
- **[advisor.py](src/physical_ai_system/domain/advisor.py)**: Antigravity 同等のアルゴリズム推薦・評価フィードバックからのハイパーパラメータ自動調整ロジック。
- **[alphaxiv_scraper.py](src/physical_ai_system/adapters/alphaxiv_scraper.py)**: `time.sleep(2.0)` 以上のウェイトを厳守した `alphaxiv.org` 論文収集アダプタ（`cron` により毎月1日 03:00 に自動実行）＋ 12ヶ月（`2025.11 - 2026.11`）の実在ヒューマノイド・Physical AI 論文コーパス統合。
- **[mujoco_engine.py](src/physical_ai_system/adapters/mujoco_engine.py)**: 陰的積分器（`implicitfast`）を用いた高安定2足歩行ヒューマノイドモデル、CSV/NPZ/BVH パーサー、3種のプリセットモーション、OSMesa オフスクリーン MP4 動画生成、Three.js 用「ロボット方策＋人間リファレンス」二重 3D 骨格フレーム出力。
- **[app.py](src/physical_ai_system/web/app.py)**: FastAPI + Plotly.js + Three.js による3タブ構成 Web アプリ（HTTP Range 動画配信対応、Nginx リバースプロキシ経由でポート80公開）。

---

## 4. v2 クオリティ＆Web操作性アップグレードの主要改善点

1. **テキストマイニング＆論文トレンド分析（Tab 1）の品質・操作性向上**:
   - **34件・12ヶ月（`2025.11 - 2026.11`）・8クラスタ** のヒューマノイド/Physical AI 論文データセットを構築し、無関係分野（医療MRI・GIS等）を排除。
   - **ワンクリック絞り込み**: クラスタピル、Sim-to-Real トグル、月別棒グラフのバークリック、頻出キーワードのバークリックで双方向に即座にフィルタリング可能。
   - **プロット点クリック時の挙動切替**: 「プロット点クリック時に `alphaxiv.org` を別タブで開く」チェックボックスを搭載し、確認のみ行いたい場合と即座に原著論文へ飛びたい場合をワンクリックで切替可能。
   - **構造化日本語要約**: 各論文に `【研究目的・手法】`・`【技術的成果】`・`【Sim-to-Real・応用示唆】`・`【キーワード】` を構造表示。
2. **4ステップ・スティッキーワークフローバー＆ワンクリックチップ（Tab 2）**:
   - 画面上部に **Step 1（AI相談） → Step 2（学習開始） → Step 3（MuJoCo・Loss・成功率確認） → Step 4（評価＆AI再学習指示）** の進捗バーを固定表示し、クリックで該当セクションへスムーズスクロール。
   - **Step 1 質問テンプレートチップ**（4種）と **Step 4 評価テンプレートチップ**（4種）により、テキスト手入力なしでもワンクリックで AI 相談・ハイパーパラメータ自動調整（変更前→変更後の差分ハイライト表示付き）が可能。
   - **ビジュアル手法選択カード**: `DAPG` / `RLPD` / `AWAC` の特徴比較カードと `AI推奨` バッジにより直感的に選択可能。
3. **MuJoCo 動画＆Three.js 二重骨格 3D ビューア・4種グラフ比較（Step 3）**:
   - **Three.js 3D ビューア**: マウスドラッグ回転・ホイールズーム・タイムラインスライダー・再生速度切替（`0.25x〜2.0x`）・カメラアングルプリセット（斜め・側面・正面）・**人間のお手本モーション（緑ゴースト骨格）とロボット方策（オレンジ骨格）の同時オーバーレイ比較**を実装。
   - **4種のインタラクティブグラフ**: ① Train/Val Loss 曲線、② タスク成功率＆生存ステップ数、③ エピソード累積報酬（移動平均付き）、④ 主要4関節（右股・左股・右膝・腰）の目標軌道 vs 実測軌道トラッキング比較。
   - **手法間横断比較テーブル**: S3 `report/` に保存された `DAPG`・`AWAC`・`RLPD` の複数実行レポートを一覧比較し、ワンクリックで表示切替可能。

---

## 5. 検証結果と考察

1. **自動テスト (`pytest -v`)**:
   - 全 9 件（API・MuJoCo 学習＆MP4/3D出力・DAPG/RLPD/AWAC 損失＆過学習診断・テキストマイニング）が **全合格**。
2. **初回表示応答時間（非機能要件「3秒以内」）**:
   - 外部クライアントからの `http://13.208.128.87/` 応答時間: **約 220 ms** を記録し、3秒以内要件を大幅に達成。
3. **S3 永続化データ**:
   - `papers/` および `text_mining/`: 34件の論文メタデータ・構造化日本語要約・2D KDE スナップショットを保存済み。
   - `report/`: `run_dapg_baseline`（DAPG）、`run_awac_smooth`（AWAC）、`run_rlpd_high_eff`（RLPD）の3手法比較レポート・MP4動画・3D二重骨格フレームを保存済み。

---

## 6. 今後のアクション / 推奨事項

1. **Web アプリへのアクセス**:
   - `http://13.208.128.87/` にアクセスして、テキストマイニングマップのクリック・フィルタ操作、Step 1〜4 のワンクリック AI 相談・再学習、Three.js 二重骨格 3D ビューア、および DAPG/AWAC/RLPD の比較テーブルをご確認ください。
2. **Amazon Bedrock の Anthropic 申請（任意・30秒）**:
   - AWS コンソール（大阪リージョン `ap-northeast-3`）の **Amazon Bedrock → プレイグラウンド（またはモデルカタログ）** から Claude Sonnet / Haiku を1回選択し、Anthropic 利用用途フォーム（Use case details）を送信すると、約15分後に Web アプリ内の論文要約と AI 相談が自動的に Claude へ切り替わります（未提出でも構造化日本語エンジンにより全機能がフル動作します）。
