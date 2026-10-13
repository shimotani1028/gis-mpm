# PR1: 数値計算の現状と再現可能な検証

対象基準: `cb5233a8e420256e02a8e060b0ed162d81601c95` / 実施日: 2026-10-13。
**既存モデルは格子単位の実験モデル。SI 水理計算、実災害予測、避難判断、土木設計には未検証。**
今回の PR は監査・自動テスト・独立した診断基盤であり、既存不具合の修正 PR ではない。

## 1. 変更ファイル

| ファイル | 役割 |
|---|---|
| `scripts/validation/reference.mjs` | 独立した CPU 数式参照。実 GPU ソルバーの代替ではない |
| `tests/numerics.test.mjs` | カーネル/APIC/単位/粘性/固定小数点/重力の15自動テスト |
| `scripts/validation/cpu-diagnostics.mjs` | 同一物理ブロックの P2G 解像度診断、自由落下の時間刻み診断 |
| `src/validation/gpu.ts`, `validation/index.html` | 実 WGSL 4本を読み込む独立した GPU 診断画面 |
| `docs/validation/numerical-audit.md` | 数式とソース行、SI 変換、検証範囲、次の修正設計 |
| `docs/validation/boundary-and-roadmap.md` | PLATEAU 5形状・座標/標高・LOD・性能・共有 solver 設計 |
| `docs/validation/results/*.json` | 実測 GPU / CPU 診断の生データ |
| `.github/workflows/validation.yml`, `package.json` | CPU テスト・診断・型検査/ビルドの CI と実行コマンド |
| `README.md`, `@tasks/todo.md`, 本文書 | 導線、計画、結果・制限・再現手順 |

本体の `src/config.ts`, `src/mpm/**`, `src/import/**`, `src/main.ts`、描画・地形は変更なし。
依存パッケージ追加なし。診断 entry point は本体に import されず、通常 build にも入らない。

## 2. 変更前に存在した問題

1. 表示変換 `x=origin+diag(L/N)q` と、応力・粘性・斜面接触の単位が整合していない。
2. y 境界でカーネルが領域外へ切断される。一粒子 y=0.02 なら格子転送質量は 0.5198 倍。
3. `atomicAdd(i32)` の合計は個別 clamp では保護できない。質量が負になる場合もある。
4. 質量≤0のセルに運動量が残ると G2P が速度として読む可能性がある。
5. 密度再構成/EOS、静水圧、時間・空間収束、建物衝突の検証証拠がなかった。
6. 通常経路の建物屋根は初期配置専用。側面衝突は一般の3D固体を表さない。

詳細な式と該当行は [数値監査](numerical-audit.md)、建物・CRS/標高の制約は
[境界監査](boundary-and-roadmap.md) に記載。

## 3. 改善した内容

- 数式の解析的性質を検査する15テストを CI で再実行できるようにした。
- 実 WGSL をコンパイル・実行し、粒子と格子を別々に読み戻す14ケースを作った。
- 粒子数一定だけでは保存性合格にしない。内部移動流体の grid mass と momentum、
  自由落下の離散解析値、摩擦なし壁の法線/接線速度を許容誤差付きで検査する。
- 既知不具合を再現する検査を明示した。これらの成功は「不具合ありを確認」の意味。
- 実行条件、GPU情報、device limits、uniform、shader SHA256、非有限成分、重心、
  最大速度、エネルギー、粒子総質量、初期/最終stepのgrid質量、時間と明示バッファ量を保存する。
- 描画を含まない計算診断として CPU 完了待ちと GPU timestamp を分けた。

## 4. 計算結果の比較

### CPU 数式参照

`npm test`: **15/15成功**。倍精度の独立数式であり、GPU/流体の正解判定ではない。
同じ0.25 m立方体・1000 kg/m³を16/32/64格子で P2G すると、各総質量15.625 kg、
内部密度1000 kg/m³、倍精度参照で相対誤差0。これは転送一回の診断であり流体収束ではない。
重力だけの1秒積分の位置誤差は dt=0.04/0.02/0.01/0.005 s で
0.196133/0.0980665/0.04903325/0.024516625 m（一次収束）。
生データ: [cpu-diagnostics.json](results/cpu-diagnostics.json)。

### 実機 GPU（既存 shader）

環境: macOS、Codex 内蔵 Chromium 155、adapter vendor=`apple`, architecture=`metal-3`。
機種名はブラウザから非公開。timestamp-query 利用可。
device maxBufferSize=256 MiB、maxStorageBufferBindingSize=128 MiB。
生データ: [gpu-apple-metal3.json](results/gpu-apple-metal3.json)。日時・全設定・hash は JSON に記録。

| ケース | 比較と観測 | 判定範囲 |
|---|---|---|
| 自由落下 32³, 1粒子, dt=.05, 20step | 離散解析値との差: 速度2.5223e-5、位置9.8114e-6 | 許容2e-4以内。単位はgrid/model time |
| 閉領域内移動 32³, 512粒子, 20step | 最終scatter質量相対誤差−3.3892e-7、運動量相対誤差3.1728e-5 | 許容質量1e-4・運動量1e-3以内。壁未接触の短時間試験 |
| 同上 4096粒子 | 質量−2.4314e-6、運動量2.2113e-4 | 同上。量子化誤差が粒子数に依存 |
| 底面 y=.02, 1粒子 | grid mass=0.5197995、解析0.5198との差5e-7 | **48.02005%の転送欠損を再現。未修正** |
| 同一セル中心2粒子, 各mass=300 | particle mass=600、grid mass=170.5032624、負質量セル1個 | **整数overflowを実機再現。未修正** |
| 摩擦なし壁、初速度(-1,0,.25), 20step | 最終x=1.00100255、法線速度絶対値3.875e-6、接線速度誤差1.8895e-5 | inset境界への非侵入と投影のみ確認。反発/摩擦/斜壁は未検証 |

全14ケースで非有限成分は0、active数不変、設定した検査は通過。ただし、これを
「発散なし・安定性保証」と解釈しない。短時間かつ既知不具合再現を含む。

### ダムブレイク・静止開始水柱の感度

平坦床、摩擦/反発0、K=3、rho0=4、mu=.01、g=-.12、比較終了モデル時刻2。
ダムは幅N/4の立方体、水柱は全水平断面の高さN/4の層。水柱は一様密度・速度0からの
緩和試験で、静水圧平衡を初期化していない。32³は4096粒子、64³は32768粒子。

| ケース | N | dt | 最大速度 | 重心y | 最終scatter質量の相対誤差 |
|---|---:|---:|---:|---:|---:|
| ダム | 32 | .05 | .250907 | 3.817131 | −1.92597% |
| ダム | 32 | .025 | .252759 | 3.824997 | −1.90906% |
| ダム | 64 | .05 | .250906 | 7.796163 | −0.96467% |
| ダム | 64 | .025 | .252682 | 7.801746 | −0.95576% |
| 水柱 | 32 | .05 | .853729 | 4.098564 | −2.01177% |
| 水柱 | 32 | .025 | .806018 | 4.104970 | −1.97866% |
| 水柱 | 64 | .05 | 1.093178 | 7.969495 | −1.02028% |
| 水柱 | 64 | .025 | 1.102704 | 7.973957 | −1.00491% |

**未検証:** 公開 dam-break データ、検証済み solver、静水圧鉛直分布との定量一致。
格子N変更で同一SI問題を維持していないため、上表を空間収束と呼ばない。
finalGridMass は最終 G2P **直前**の scatter で、final粒子位置に再scatterした量ではない。
初期転送から床で欠損しており、単なる浮動小数点誤差ではない。

## 5. 性能測定結果

変更前後の高速化比較は対象外（本体コード未変更）。以下は計測基盤の一回の試行値。
CPU 側 command encoding は測定外、submit→GPU完了待ちを含み、readbackは測定外。
同期やOSスケジューリングを含むため、純粋GPU処理時間・本体性能とは呼ばない。

| ケース (dt=.05) | 粒子数 | step数 | submit+完了待ち | 同時間によるstep/s | 明示buffer合計 |
|---|---:|---:|---:|---:|---:|
| 閉領域32³ | 512 | 20 | 12.2 ms | 1639.3 | 1,131,344 B |
| 閉領域32³ | 4096 | 20 | 19.1 ms | 1047.1 | 1,704,784 B |
| ダム32³ | 4096 | 40 | 31.5 ms | 1269.8 | 1,704,784 B |
| ダム64³ | 32768 | 40 | 50.3 ms | 795.2 | 13,632,336 B |

最後のstepの timestamp（ダム64³ dt=.05）はP2G mass 0.393216 ms、応力0.458752 ms、
grid 0.065536 ms、G2P 0.131072 ms。量子化単位約.065536 ms、スケジューリングの
ばらつきが大きく、中央値/p95や高速化率の根拠にはしない。0 msは無負荷を意味しない。
境界はgrid/G2Pに内包。各stageを別passにしているため本体の同一pass方式と時間も異なる。
1step warmup後の測定であり、長いwarmup・複数試行・50k/200k/500k粒子は後続課題。
参考: [GPUWeb timestamp API](https://gpuweb.github.io/types/interfaces/GPUComputePassTimestampWrites.html)。

buffer合計にはreadbackを含み、driver/pipeline/texture/地形は含まない。
`kineticEnergy` は粒子並進分のみ（APIC affine分を含まない）、densityMin/Max はrho/rho0。
GPU総メモリ実使用量、境界単独、水面/地形の個別GPU時間、統制されたFPSは**未測定**。

## 6. 既存デモの回帰確認

- 変更前/変更後とも `npm run build` 成功。主bundleは同じ `index-C327ZNu9.js`、
  workerも同じ出力hash。本体ソース差分なし。既存の大bundle警告は残る。
- 実機ブラウザで GSI地形61 tiles・active142999/上限500000を読み込み、地形と水面を表示。
- Startで時刻進行、Stopで18.60 model timeに停止、Resetで0.00へ復帰を確認。
- この操作中、ブラウザ取得console errorは0件。これは長時間安定性・全操作の保証ではない。
- PLATEAUタブのサンプル読込は外部 terrarium タイルが HTTP 400 を返し完了しなかった。
  変更前commitを別ポートで実行して同種の400を再現。変更後は `15/29028/13012.png`、
  変更前は `15/29028/13011.png` の取得で失敗。両方 `?geoid=none`。
  [操作ログ](results/demo-smoke.json)と[読込失敗画面](results/plateau-load-error.png)を保存。
  外部配信側・URL仕様のどちらが原因かは未特定で、本PRでは修正していない。
- PLATEAU読込完了・実データ衝突精度、形状5種類、標高基準は未検証。
- [GSIリセット後画面](results/gsi-demo-reset.png)、[GPU診断完了画面](results/gpu-validation.png)を保存。

## 7. 残存問題・新たな制限

単位、床support、overflow、屋根/開口/高架の問題は未修正。
次はPLATEAU改善より先に**既存実験モデルを維持する独立SI検証モード**を作ることを推奨。
長時間安定性、物理条件を揃えたh/dt収束、静水圧・公開ベンチマークを受入条件にする。
PR2衝突/PR3高速化/PR4格子可変・solver共有は別PRで実施する。

診断画面の制限: 開発サーバー専用、WebGPU必須、通常描画なし、地形/建物なし、
格子32/64の合成ケースのみ。128や異方格子の本体対応は実装していない。
device上限超過はskip、コンパイル/検査失敗は成功扱いにしない。
固定小数点の全寄与・飽和回数や全stepの中間値はGPU監視していない。
既知不具合テストは将来修正PRで「再現」から「解消」へ期待値を更新する必要がある。

## 8. 再現手順

```sh
npm ci
npm test
npm run --silent benchmark:cpu > cpu-diagnostics.json
npm run build
npm run dev
```

1. WebGPU対応ブラウザで `http://127.0.0.1:5173/validation/` を開く。
2. 「標準検証を実行」を押し、14ケースが完了するまで待つ。
3. 結果JSONを保存。`checks[].pass`, `errors`, `expectedKnownDefect`, `physicalValidation`
   を合わせて確認する。`diagnostic-completed` は物理精度合格の意味ではない。
4. 保存JSONのshader hash・uniform/parameters・GPUを本PRの生データと比較。
   別GPUで許容差を変える場合は理由を記録し、失敗を隠さない。
5. `/` でGSI地形表示・Start/Stop/Reset・水面を確認する。

CIはCPUテストとbuildのみ。GPUは実行環境を明記した手動診断であり、CI成功だけで
GPU・PLATEAU・物理妥当性を検証済みとはしない。
