# WebGPU画像テクスチャの形式選択

画像の格納型・encoding・Float/RGB用途を保ち、GPU転送時のRGBA32Float固定を解消する。
CPUの画像読み取り・MIPMapの契約は維持する。

## 形式と責務

| 投影後の格納型 | Float用途 | RGB用途 |
| --- | --- | --- |
| linear U8 | R8Unorm | Rgba8Unorm |
| sRGB U8 | 線形F16/F32へ変換 | Rgba8UnormSrgb |
| linear F16 | R16Float | Rgba16Float |
| linear F32 | R32Float | Rgba32Float |

ImageDecoderはU8のencoding、EXRのhalf/floatを保持する。EXRの選択チャンネルに
型の混在がある場合はF32とする。追加のEXR reader APIを使い、既存CPU readerの
戻り値は変えない。

ImageCompilerが用途別のチャンネル選択、RGB平均、格納型選択を担当する。
F32→F16は既存ImageOptimizationPolicyに従い、範囲・誤差条件を満たさない画像は
F32のまま残す。任意gammaやsRGBのscalar値は線形F16/F32へ変換する。
アルファにもv4のImageと同じencodingを適用する。sRGBアルファをR8として直接使わない。

ミップの計算は線形値で行い、各levelを元のencodingと格納型へ符号化する。
丸め前の計算値から次のミップを作る。既存のNPOT処理とフィルター形状は変更しない。
v4のpower-of-twoリサイズを含むミップ生成全体の一致を主張する変更ではない。

WebGPU uploaderは全levelを検証して形式を一度選び、F16をF32へ戻さず転送する。
RGBだけA=1を追加し、Floatは1chのまま格納する。WGSLはFloat画像のRをvec3へ複製して
からscale/invertを適用する。既存のfilterable float texture配列を使い、ABIは変更しない。

画像光源の重要度分布、power、illuminance、portal変換は線形F32で計算する。
転送にはsourceの格納型、portal変換結果には既存のF16/F32 policyを使う。
同じsourceの通常画像光源は格納データとGPU画像indexを共有し、sampling recordは個別に保つ。

v4対応の確認先はローカルpbrt-v4の`textures.cpp`のGPU画像factoryと
`createSingleChannelTextureArray`、`util/image.cpp`の`GeneratePyramid`、
`GetChannels`、`CopyRectIn`、`util/image.h`の`SetChannel`。

## RTX 3060での検証

NVIDIA GeForce RTX 3060、driver 610.57.04、Vulkanで確認した。

- 7形式を1つのbinding arrayに混在させ、textureLoad、filtering、mip参照、
  productionのblack-wrap helperをreadbackで検証した。
- 7形式を使う112×32、4 spp、volpath/maxdepth 1のシーンを描画した。
  同じグラフで画像の葉だけを定数へ置き換えた参照との最大絶対RGB差は
  `0.000028789043`、相対L1差は`0.00001639941184`（0.001640%）。
- 2048×2048・全12ミップの画像7枚について、Vulkan allocatorのlive allocationは
  RGBA32Float固定で626,359,808 bytes、7形式を1枚ずつ使う場合は218,118,368 bytes。
  この組み合わせでは65.177%減少した。常駐VRAM全体や描画速度の測定値ではない。

```sh
cargo test --offline --test gpu_image_formats
cargo test --offline --test gpu_image_formats \
  mixed_formats_sample_and_load_through_one_binding_array -- --ignored --nocapture
cargo test --offline --test gpu_image_formats \
  seven_image_formats_match_constant_texture_leaves -- --ignored --nocapture
cargo test --offline --test gpu_image_formats \
  selected_formats_reduce_native_texture_allocations -- --ignored --nocapture
```

GPU依存テストは通常実行ではignoreする。描画用のsource・scene・EXR・logを残す場合は
`PBRT_IMAGE_FORMAT_RENDER_DIR`に出力先を指定する。初期検証のllvmpipeでも数値readbackは通過した。

CPU/Flat IRのfocused testでは型保持、sRGBミップ、alpha、HDRのF32維持、payload/row layout、
共有画像光源と個別sampling recordを確認する。合成したWGSLの検証も行う。
