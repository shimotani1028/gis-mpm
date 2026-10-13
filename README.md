# GIS-MPM

このプロジェクトは、matsuoka-601 さんの webgpu-ocean の実装と解説記事を参考に、地形上で流体計算を行えるようにすることを目的としています。
WebGPU を使ったブラウザ上の 3D 地形ビューア兼 MPM シミュレーション実験アプリです。
国土地理院の標高タイルや PLATEAU データを読み込み、地形・建物・粒子を 3D 表示します。

## 必要環境

- Chrome / Edge 113 以降
- WebGPU が有効な環境

## 数値検証（実験モデル）

現行計算は格子単位の実験モデルです。異方的なメートル表示を含む SI 整合性、
床付近の転送欠損、固定小数点のオーバーフローなどを監査しています。
[検証結果・制限・再現手順](docs/validation/README.md)を参照してください。

```sh
npm ci
npm test
npm run benchmark:cpu
npm run dev
```

開発サーバーの `/validation/` から、実 WGSL の独立 GPU 診断を実行できます。
通常のデモ・ビルドには組み込んでいません。テストの成功は実現象の正確さや
災害予測への適用を保証しません。

## データ出典

このアプリは、画面右下に使用中データの出典を表示します。

- 国土地理院の地理院タイル（標高タイル）を加工して作成
  - <https://maps.gsi.go.jp/development/ichiran.html>
- 国土交通省 Project PLATEAU の PLATEAU-Terrain / CityGML を加工して作成
  - <https://www.mlit.go.jp/plateau/site-policy/>

## 参考資料

- [webgpu-ocean](https://github.com/matsuoka-601/webgpu-ocean/tree/main)
- [WebGPU で MLS-MPM を実装し，ブラウザ上で 10 万粒子規模のリアルタイムシミュレーションを実現する](https://zenn.dev/sparkle/articles/3eb1225f891a87)

## 免責

このソフトウェアは実験用です。表示やシミュレーション結果の正確性は保証しません。
災害予測、避難判断、土木設計、行政判断など、安全に関わる意思決定には使用しないでください。

## ライセンス

ソースコードは MIT License です。詳細は [LICENSE](LICENSE) を参照してください。
外部データは、それぞれの提供元の利用条件に従ってください。
