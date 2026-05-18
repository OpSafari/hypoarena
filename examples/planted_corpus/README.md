# planted_corpus — 生成植入语料并校验 grounding

完全离线、确定性，不联网、不调用真实模型。运行（仓库根目录）：

```bash
.venv/bin/python examples/planted_corpus/verify_grounding.py
```

脚本做三件事：

1. 用 `synthetic.build_bundle` 生成带植入因果链的合成语料与 gold claim；
2. 用 `grounding.GroundingVerifier` 校验整张假设图，打印 grounding 摘要；
3. 把一条 gold claim 的引用改指向语料中**不存在**的文档，演示校验器会把它标成
   `fabricated`——即编造引用一定被拦下。

实际输出（seed=7，合成数据，由本脚本真实运行得到）：

```text
planted-corpus grounding check (synthetic, offline, seed=7)
  documents=22  gold claims=9
  claims graded=9
  grounded=9  weak=0  ungrounded=0  fabricated=0
  grounded_rate=1.000  mean_score=1.000
  a gold claim is grounded as: grounded
  the same claim citing an absent document: fabricated
note: synthetic corpus with planted ground truth; no real literature and no model benchmark.
```

这些数字只描述 grounding 机制在这份**合成**语料上的表现：gold claim 按构造引用真实
span，因此全部 grounded；被篡改引用的 claim 被判为 fabricated。它们不是任何真实模型的
基准，也不构成对科学真伪的判断。
