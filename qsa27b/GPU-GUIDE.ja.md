# GPU別のセットアップと起動

[English](GPU-GUIDE.md) | [日本語README](../README.ja.md)

[QSA対応エンジン](https://github.com/kuraneko1/Qwen3.8-27B-QSA-llama.cpp)と[QSA GGUF](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF)を使用します。通常のllama.cppには、このQSA経路は実装されていません。

## 動作を確認した構成

以下は単一GPUで起動し、短いテキスト応答を確認した構成です。VRAM使用量はGPU全体の観測値で、F16 mmprojと最大2トークンのMTPを含みます。ドライバー、ほかのプロセス、実行設定などで変わります。

| GPU | モデル | コンテキスト容量 | 本体KV | 観測VRAM使用量 |
| --- | --- | ---: | --- | ---: |
| RTX 3090 24 GB | UD-Q5_K_M | 65,536 | Q8_0 | 23,779 / 24,576 MiB |
| Tesla V100 32 GB | UD-Q6_K | 65,536 | Q8_0 | 25,944 / 32,768 MiB |
| Tesla V100 32 GB | UD-Q4_K_M | 262,144 | Q8_0 | 約28,102 / 32,768 MiB |

Q5とQ6の構成では、入力埋め込みを含むモデルテンソルをGPUに配置しました。Q4は入力埋め込みをCPUに置いていました。コンテキスト容量は起動時に確保した枠で、起動確認時の入力長ではありません。Q5/Q6は短い応答、Q4は最大16,064トークンの入力で確認しました。一般的な品質・速度のベンチマークではありません。画像・動画の処理系は読み込んでいますが、Q5/Q6の起動確認では画像・動画の回答品質を評価していません。

## GPUに合わせてビルド

Linux、CMake、Ninja、C++コンパイラー、CUDA toolkitが必要です。表はビルド対象の例であり、各GPUでモデルの動作を確認したという意味ではありません。

| GPU | `QSA_CUDA_ARCH` |
| --- | --- |
| Tesla V100 | `70` |
| RTX 3090 / 3090 Ti | `86` |
| RTX 4090 | `89` |

```sh
git clone https://github.com/kuraneko1/Qwen3.8-27B-QSA-llama.cpp.git
cd Qwen3.8-27B-QSA-llama.cpp
# 例: RTX 3090。CUDA_HOMEはtoolkitのインストール先に合わせます。
QSA_CUDA_ARCH=86 CUDA_HOME=/usr/local/cuda-12.9 bash qsa27b/build.sh
```

既定のサーバーは `build-qsa/bin/llama-server` に作成されます。配布済みruntime archiveはLinux x86_64・NVIDIA sm70向けです。ほかのGPUアーキテクチャーではソースからビルドしてください。新規ビルドと配布済みバイナリがbyte単位で一致する保証はありません。

## モデルをダウンロード

Hugging Face CLIをインストールした環境で、Q5と共通の画像・動画用projector、3種類のUnsloth版チェックサムを取得する例です。モデル名をQ4またはQ6に変更して選べます。projectorは言語モデルのGGUFとは別ファイルです。

```sh
hf download higa2515/Qwen3.8-27B-QSA-GGUF \
  Qwen3.8-27B-QSA-Unsloth-UD-Q5_K_M.gguf \
  mmproj-Qwen3.8-27B-QSA-F16.gguf unsloth-qsa/SHA256SUMS \
  --local-dir models
(cd models && grep -F 'Qwen3.8-27B-QSA-Unsloth-UD-Q5_K_M.gguf' unsloth-qsa/SHA256SUMS | sha256sum -c -)
```

## 単一GPUで起動

65,536コンテキスト、本体KV Q8_0、MTP draft 2の確認済み構成を基にした例です。GPU番号は `nvidia-smi` で確認してください。GPUを1枚だけ可視にした場合、サーバー内での名前は `CUDA0` になります。Q4やQ6を使う場合はモデルのパスを変更します。

```sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export QSA_MODEL="$PWD/models/Qwen3.8-27B-QSA-Unsloth-UD-Q5_K_M.gguf"
export QSA_MMPROJ="$PWD/models/mmproj-Qwen3.8-27B-QSA-F16.gguf"
export LLAMA_QSA_RESERVE_BLK=1 LLAMA_QSA_BLK_REBUILD=1
export LLAMA_QSA27B_MASKED_PREFILL_MAX=8192 LLAMA_QSA_RAW_RING=8192
export LLAMA_MTP_KV_ONLY=1 LLAMA_MTP_PAD=4 LLAMA_MTP_N_UBATCH=64
export LLAMA_MTP_LAG=0 LLAMA_MTP_ASYNC=0 LLAMA_SPEC_PREFILL_TAIL=2048
export LLAMA_REUSE_OUT_ORDER=1 LLAMA_KV_PAD_GEO=2 LLAMA_CKPT_TAIL_UBATCH=1
export GGML_FT_GRAPH_CACHE=0 LLAMA_GPU_ARGMAX=1 LLAMA_SAMPLE_TOPK=1
export GGML_CUDA_TOPK_DET=1 CUDA_SCALE_LAUNCH_QUEUES=4x

./build-qsa/bin/llama-server \
  -m "$QSA_MODEL" --mmproj "$QSA_MMPROJ" \
  --device CUDA0 --device-draft CUDA0 --mmproj-device CUDA0 \
  -ngl all -ngld all -sm none --fit off \
  --override-tensor token_embd.weight=CUDA0 \
  -fa on -np 1 -t 4 -tb 4 -td 4 -tbd 4 \
  -c 65536 -b 2048 -ub 256 \
  -ctk q8_0 -ctv q8_0 -ctkd f16 -ctvd f16 \
  --cache-ram 0 --ctx-checkpoints 8 --checkpoint-min-step 1024 --no-context-shift \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0 \
  --spec-draft-sampling probabilistic --jinja --host 127.0.0.1 --port 8091
```

入力埋め込みの明示指定は、既定でCPUに置くエンジンでもGPUへ配置できます。その分VRAMを使うため、容量が足りなければ `-c` を小さくするか、`--override-tensor token_embd.weight=CPU` に変更して入力埋め込みをCPUに置いてください。どのモデルも収まることを保証するものではありません。CPUオフロード時の速度は実測してください。

## GPUを2枚使う

例えばV100とRTX 3090の両方に対応させるビルドでは、アーキテクチャーをまとめて指定します。

```sh
QSA_CUDA_ARCH='70;86' CUDA_HOME=/usr/local/cuda-12.9 bash qsa27b/build.sh
```

単一GPUの起動例を変更する場合、例えば `export CUDA_VISIBLE_DEVICES=0,1` で2枚を可視にします。`--device CUDA0` を `--device CUDA0,CUDA1` に、`-sm none` を `-sm layer -ts 1,1` に変更します。例ではMTP draftとmmprojは `CUDA0` のままなので、そのGPUの空き容量も確認してください。`-ts` は可視GPUの順番に対応するレイヤー配置比率です。GPU容量が異なる場合は調整してください。すべてのGPUの組み合わせ・モデルで検証済みの設定ではありません。

## 起動後の確認

```sh
curl -fsS http://127.0.0.1:8091/health
curl -fsS http://127.0.0.1:8091/v1/models
```

Web UIは `http://127.0.0.1:8091/` です。例はAPIキーなしでlocalhostにのみ待ち受けます。外部から使う場合は、待受先を変更する前に適切なアクセス制御を設定してください。
