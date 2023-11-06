import os
import requests
import json  # jsonモジュールをインポートする
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# OpenAIのチャットモデルをセットアップする
def setup_chat_model():
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("環境変数にOPENAI_API_KEYが設定されていません。")
    model_kwargs = {
        'api_key': openai_api_key,
    }
    return ChatOpenAI(model_kwargs=model_kwargs, model_name="gpt-4-0613", max_tokens=32768)


# Kibelaでの検索を行う
def search_kibela(query, token, team):
    endpoint = f'https://{team}.kibe.la/api/v1'
    graphql_query = """
    query Search($query: String!) {
      search(query: $query, first: 10) {
        edges {
          node {
            title
            document {
              ... on Note {
                content
              }
            }
          }
        }
      }
    }
    """

    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    # GraphQLのインジェクション攻撃を防ぐために、変数を使う
    response = requests.post(endpoint, headers=headers, json={'query': graphql_query, 'variables': {'query': query}})
    return response

def split_into_chunks(text, max_length):
    """
    与えられたテキストを最大長以下のチャンクに分割する
    """
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def summarize_chunk(chat_model, chunk, max_tokens=1000):
    print("経過：kibela要約中…")
    prompt = f"以下のテキストを要約してください:\n\n{chunk}"
    #print(prompt)
    response = chat_model.predict(text=prompt, max_tokens=max_tokens)
    #print(response)
    print("経過：要約の一部が完了")
    # レスポンスがバイナリ文字列の場合、デコードする
    if isinstance(response, bytes):
        response = response.decode('utf-8')
    # レスポンスが文字列の場合、そのまま使用する
    if isinstance(response, str):
        return response.strip()
    # それ以外の場合、エラーを返す
    raise ValueError("Invalid response type received from predict function.")


def summarize_results(chat_model, search_results):
    # 検索結果をJSONオブジェクトとしてパースする
    search_results_obj = json.loads(search_results)
    # 検索結果からテキストを抽出する
    texts = [edge['node']['document']['content'] for edge in search_results_obj['data']['search']['edges']]
    # テキストを連結する
    full_text = ' '.join(texts)
    # テキストをチャンクに分割する
    chunks = split_into_chunks(full_text, 3500)  # トークンではなく文字数に基づいた仮の分割

    summaries = []
    for chunk in chunks:
        # チャンクごとに要約を行う
        summary = summarize_chunk(chat_model, chunk, 5000)
        summaries.append(summary)
        if len(summaries) == 1:
            break  # 最大5回のリクエストに制限する

    # 全チャンクの要約を結合する
    combined_summary = ' '.join(summaries)
    # 結合した要約をさらに要約する
    final_summary = summarize_chunk(chat_model, combined_summary, 5000)
    return final_summary

# メイン関数
def main():
    chat_model = setup_chat_model()

    search_query = input("検索クエリを入力してください: ")
    token = os.getenv('KIBELA_API_TOKEN')
    team = os.getenv('KIBELA_TEAM')

    if not token or not team:
        print("KIBELA_API_TOKEN または KIBELA_TEAM 環境変数が設定されていません。")
        return

    search_response = search_kibela(search_query, token, team)

    if search_response.ok:
        search_results = search_response.json()
        summary = summarize_results(chat_model, json.dumps(search_results, ensure_ascii=False))
        print("要約結果:")
        print(summary)
    else:
        print(f'レスポンスステータスコード: {search_response.status_code}')
        print(f'レスポンス内容: {search_response.text}')

# スクリプトとして実行されたときにのみmain関数を実行する
if __name__ == "__main__":
    main()
