# API設計

## エンドポイント一覧

> (例)
>
> ### ユーザー関連
> - `POST /users` - ユーザー登録
> - `GET /users/{id}` - ユーザー情報取得
> - `PUT /users/{id}` - ユーザー情報更新
> - `DELETE /users/{id}` - ユーザー削除


## リクエストとレスポンス例

### POST /users

> (例)
> 
> **リクエスト**
> ```json
> {
>   "name": "string",
>   "email": "string",
>   "password": "string"
> }
> ```
> 
> **レスポンス**
> ```json
> {
>   "id": "string",
>   "name": "string",
>   "email": "string"
> }
> ```
