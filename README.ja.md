# Qwen3.8-27B QSA for llama.cpp

[English](README.md) | [日本語](README.ja.md)

このforkはQwen3.8-27B向けにQSA、マルチモーダルindexer、MTP、稼働中のprefixキャッシュ再利用を追加したものです。対象はQwen3.8-27Bであり、Qwen3.8-FlashNextではありません。

基盤は[pentaCoxian/Qwen-V100-llama.cpp](https://github.com/pentaCoxian/Qwen-V100-llama.cpp)のcommit `6209237df53569f07429f3f79fc5354de5fe704f`です。

[GGUFと実行環境](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF) | [BF16ソースと元モデルの復元](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-BF16)

## 実装

16個の全注意層では4トークンのblockを対象に、固定K=2,048と現在のblockを選択します。残る48層のGated DeltaNet構造は保持しています。KVはすべて保持するため、QSAはKVキャッシュを圧縮する方式ではありません。backbone、vision encoder、projector、MTPの重みは、量子化前の時点では変更していません。

テキスト・画像・動画にはM-RoPEに対応したnative QSAを使用します。MTPは最大2個のdraft tokenを生成し、lag/asyncは無効です。1つの稼働中slotで、本体KV、pooled indexer key、MTP carryを再利用します。このキャッシュはディスクには保存されず、サーバー再起動後には引き継がれません。

## ビルドと実行

ビルド、変換、実行手順は[ランタイムガイド](qsa27b/RUNTIME.md)を参照してください。収録した実行環境はCUDA 12.9とsm70の32GB GPU 2台を使用します。それ以外のハードウェアでは、ソースからのビルドとメモリ容量に合わせた設定が必要です。

成果物名、checkpoint path、branch名に残る`mm2`は、互換性と再現性のために保持した過去の実験識別子です。QSA方式の名前ではありません。

## 測定結果と限界

現行QSAだけを測定したテキストベンチでは、decode速度は0kの追加コンテキストで48.94 token/s、261,120入力トークンで41.07 token/sとなり、低下率は16.08%でした。コード2課題と散文2課題を混ぜ、各課題を各コンテキスト深度で3回ずつ測定しています。0kの測定点にも490〜563 tokenのタスクpromptが含まれます。

通常attentionとの同条件対照はまだ完了していません。マルチモーダル確認は64kまでの限定的なprobeにとどまり、広範な品質同等性を示すものではありません。

## ライセンスと上流

[元のforkのREADME](README.upstream.md)も参照できます。MITおよびthird-party noticesを保持しています。
