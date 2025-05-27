# Database設計

## テーブル一覧


> (例)
>
> ### users テーブル
> | カラム名   | データ型   | 制約         |
> |------------|------------|--------------|
> | id         | UUID       | PRIMARY KEY |
> | name       | VARCHAR(50)| NOT NULL    |
> | email      | VARCHAR(100)| UNIQUE, NOT NULL |
> | password   | VARCHAR(255)| NOT NULL    |
> | created_at | TIMESTAMP  | NOT NULL, DEFAULT now() |
> | updated_at | TIMESTAMP  | NOT NULL, DEFAULT now() |


## インデックス

> (例)
> 
> ### users テーブル
> - email に UNIQUE インデックス

## リレーションシップ

> (例)
>
> ### users テーブル
> - users.id → habits.user_id (1:N)


## データベース設計の説明

> (例)
>
> ### habits テーブル
> - `description`: ユーザーが入力した習慣の自然言語での説明
> - `action`: AIが抽出した実際の行動（例: 「英語を勉強する」）
