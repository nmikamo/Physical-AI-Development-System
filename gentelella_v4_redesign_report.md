# Gentelella 2026 v4 ベース Web UI 全面刷新レポート

## 1. 目的・前提
- **対象システム**: Physical AI Development System (`src/physical_ai_system/web/`)
- **参照テンプレート**: [Dashboard | Gentelella 2026 v4](https://preview.colorlib.com/theme/gentelella/production/index.html)
- **実装方針**:
  1. まず Gentelella 2026 v4 のデザインシステム（`gentelella-v4.css`）およびレイアウト構造をベースにしたデザインテンプレート（モック）を構築し、既存バックエンドAPIとの統合は段階的に実施する。
  2. 総合 `Dashboard (Overview)` を新設し、サイドバーから `Dashboard` / `Papers & Mining` / `AI Advisor & Training` / `Evaluation & 3D` / `System & AWS` の5画面を切り替え可能にする。
  3. グラフ・チャート可視化を **ECharts 6** へ移行し、3D骨格表示（**Three.js**）は維持する。

## 2. 主な変更内容
- **[gentelella-v4.css](file:///c:/Users/mikamo/Desktop/MEI%20&%20AME/PhysicalAIDevelopmentSystem/src/physical_ai_system/web/static/css/gentelella-v4.css)** (新規追加)
  - Gentelella 2026 v4 のトークン駆動CSS（Light / Dark テーマ、レスポンシブサイドバー・64pxレールモード、Statカード、テーブル、ステッパー、トースト、コマンドパレット）を導入。
- **[style.css](file:///c:/Users/mikamo/Desktop/MEI%20&%20AME/PhysicalAIDevelopmentSystem/src/physical_ai_system/web/static/css/style.css)** (全面改訂)
  - ページセクション切替、フォームグリッド、DAPG/RLPD/AWAC アルゴリズム選択カード、AIチャットログ、Three.js 3Dステージ、AWSインフラKey-Valueグリッド等の拡張スタイルを Gentelella v4 変数で統一。
- **[index.html](file:///c:/Users/mikamo/Desktop/MEI%20&%20AME/PhysicalAIDevelopmentSystem/src/physical_ai_system/web/templates/index.html)** (全面改訂)
  - Gentelella v4 シェル（アコーディオン付き左サイドバー、検索/コマンドパレット付きトップバー、Light/Darkテーマ切替、パンくずリスト）を構築。
  - 5つのページビュー（`#page-dashboard`, `#page-papers`, `#page-train`, `#page-eval`, `#page-system`）および `Ctrl+K` / `⌘K` コマンドパレットを実装。
  - チャートライブラリを `echarts@6` CDN へ切り替え。
- **[app.js](file:///c:/Users/mikamo/Desktop/MEI%20&%20AME/PhysicalAIDevelopmentSystem/src/physical_ai_system/web/static/js/app.js)** (全面改訂)
  - 関数型設計に基づき、イミュータブルなモックデータ定義・純粋関数による ECharts 6 オプションビルダー・テーマ連動再描画・Three.js 3Dヒューマノイド歩行モックアニメーションを実装。

## 3. 検証・現状の制約
- ローカル環境（Windows）には Python / pytest が未導入のため、`pytest -v` によるテスト実行は未実施（`index.html` の `<title>` に `Physical AI Development System` を保持し、`tests/test_api.py` のアサーション条件とは整合）。
- バックエンドAPI（`/api/health`）が応答する場合はトップバーのステータスバッジが `API Connected (Mock UI)` に切り替わり、静的プレビュー時は `Template Preview` として動作する。

## 4. 今後のアクション（段階的統合フェーズ）
1. **Papers & Mining 統合**: `/api/dashboard-init`・`/api/papers/scrape` と接続し、ECharts 6 散布図・年度別トレンド・論文テーブルを実データ化。
2. **Training & Advisor 統合**: `/api/advisor/ask`・`/api/train/start`・ジョブ進捗ポーリングを接続。
3. **Evaluation & 3D 統合**: `/api/jobs/{job_id}/video`（MP4再生）および実フレームJSONでの Three.js 骨格再生・実損失カーブ描画を接続。

## 5. 全画面バックエンド API 統合完了 (2026-11-14)
- **Dashboard**: `/api/dashboard-init` から取得した論文総数・Sim2Real件数・最新成功率・保存済みRun一覧・アルゴリズム比率を実データ描画。Run行クリックで `Evaluation & 3D` へ遷移。
- **Papers & Mining**: `/api/papers/scrape` + `/api/jobs/{job_id}` ポーリング、クラスタ/年度/Sim2Real/キーワード検索フィルタ、ECharts 6 2D散布図クリックでの日本語要約・alphaXivリンク表示を統合。
- **AI Advisor & Training**: `/api/advisor/ask`（推奨設定のワンクリック適用）、`/api/mocap/upload`、`/api/robot-xml/upload`、`/api/training/start` + ジョブ進捗バーを統合。
- **Evaluation & 3D**: `/api/runs/{run_id}/report`, `/api/runs/{run_id}/video` (MP4再生), `/api/runs/{run_id}/frames` (Three.js Policy/Reference骨格比較再生), `/api/advisor/retrain` (AI調整案確認・自動再学習) を統合。
