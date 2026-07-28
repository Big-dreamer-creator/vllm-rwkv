# rwkv-skills prompt-profile context sweep

这组结果复刻了 `rwkv-skills` knowledge runner 的四个 job：

- `normal_direct`：normal profile，无 CoT
- `normal_cot`：normal profile，官方 G1h `User✿...Bot✿<think` CoT
- `naive_direct`：naive profile，无 CoT
- `naive_cot`：naive profile，`User: ... Assistant: <think` CoT

每个格式使用相同的 early/late secret-token 任务，输入目标长度为
1K、2K、4K、6K、8K、10K、12K、16K、20K；生成参数为
`temperature=0`、`top_p=1`、`seed=20260728`、`max_tokens=128`。

## 服务对照

- baseline：`context-window-strategy=none`，GPU2/GPU3，端口 19348/19349
- sliding：`context-window=8192`、`context-window-strategy=sliding_window`，GPU2/GPU3，端口 19344/19345

滑窗结果在所有 raw prompt 超过 8192 的请求中都报告 `actual_prompt_tokens=8192`。
baseline 使用当前 `ctx10240` 权重和 128 个输出 token，因此从约 10K 输入开始会因总长度超过
10240 被 vLLM 拒绝；这属于当前权重/服务上限，不是滑窗精度结果。

## 结果文件

- `1p5b_baseline.json` / `1p5b_sliding.json`
- `7p2b_baseline.json` / `7p2b_sliding.json`

`correct` 只表示生成文本是否包含预置 secret token，不等价于正式 benchmark accuracy。
重复请求探针显示 7.2B 在 `temperature=0` 和相同 seed 下仍可能出现不同的 CoT 文本，
所以单次 completion 的文本差异不能直接归因于 context-window 插件；正式精度比较应使用多次重复、
相同请求配对，并同时比较 token/logprob 或最终判定。
