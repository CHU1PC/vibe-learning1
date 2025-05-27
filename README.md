# VibeCoding テンプレートリポジトリ

## 概要
このリポジトリは、VibeCodingプロジェクトのテンプレートとして使用されます。以下のルールに従って開発を進めてください。

## 使い方

「use this template」ボタンを使用し、自身のリポジトリを作成してください。
以後の作業はすべて自身のリポジトリで実施してください。

## 開発ルール
1. **仕様書作成フェーズ**
   - プロジェクト開始時に、`prd.md` をGPTsを使ってつくったものに置き換えてください
   - prdをもとに、画面設計、api設計、DB設計、技術選定を行います。
     - それぞれ、Agentに「api設計仕様書を `api.md` に記述してください」など指示すればベースが完成します。
   - UIデザインは上記仕様書を使用し、以下のツールなどを活用して作ってください。（以下以外でももちろんOK）
     - v0
     - Lovable
     - Create
     - Stitch
     - Claude / Gemini / ChatGPT　（汎用系）
   - 完成したUIラフは Html to design なども適宜つかって、Figmaにいれておくと便利です
     - 実装時にFigma MCPを使うととっても速いです。

2. **開発フロー**
   - `vibecoding-plan.md`をAgentに作成させてください。
   - 計画書と仕様書に基づいてコードを実装します。計画書のステップごとに進めると良いです。
   - 実装後、コードレビューを行い、必要に応じて修正を加えます。

3. **Git管理**
   - まめに変更を記録しましょう。Agentに「pushしておいて」と依頼するだけでもOKです
   - コミットメッセージもAgentに依頼して作成してもらいましょう
   - 必要に応じてブランチを作成し、プルリクエストを通じて変更をマージしてください。

4. **最後に**
   - Vibe Codingが終わったら学びカリキュラムを生成してください。
   - 学びカリキュラムに沿って学習を進めてみましょう！

## ファイル構成
- `docs/details/api.md` : API設計書
- `docs/details/database.md` : DB設計書
- `docs/details/technology-selection.md` : 技術選定書
- `docs/prd/prd.md` : PRD仕様書
- `docs/vibecoding-plan/vibecoding-plan.md` : VibeCoding計画書
- `learning-path/learning-path.md` : 学びカリキュラム
- `src/` : ソースコード (実装部分)

## 注意事項
- 不明点があれば、運営やメンターに相談してください。