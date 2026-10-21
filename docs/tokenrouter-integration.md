# TokenRouter 接入方案：接插件还是接 DSH 内核

> **状态：未落地（调研记录）。** 结论是「接插件，不接内核」，分级路径见下；但 TokenRouter 硬依赖 NVIDIA GPU（见附录），本机（Apple M1）跑不起来，用户已决定不实施。本文档保留为调研记录——若将来有 GPU 机器，按「七、落地顺序」执行即可。

## 结论先说

**接插件，不接内核。** 但不是"再写一个 provider 记录"那么轻——TokenRouter 值得一次中等规模的改造，因为它同时是"一条聚合车道"和"一个需要托管的 Python 服务"。推荐路径分三级，按投入从轻到重：

| 级别 | 做法 | 代码量 | 得到的能力 |
|---|---|---|---|
| **P0（立即可做）** | 注册表加一条 local provider | ~1 条 JSON + 4 处测试常量 | 手动填 `http://localhost:30000/v1` 与模型 id 后可用；可进 combo 降级链；设置页零改动 |
| **P1（推荐落地）** | P0 + 托管子进程 + 配置面 | 新增 `src/tokenrouter/`，约 300 行 | 一键启停、健康状态、模型 id 从 YAML 自动推导、错误可读 |
| **P2（重投入）** | 插件做成 TokenRouter 的控制面 | 600+ 行 + 前端 | 在 DSH 里选算法、写 YAML、看路由决策；与上游强耦合，不推荐 |

核心判断依据：TokenRouter 对外的协议形状（一个 OpenAI 兼容 base URL、model 字段随便传、路由 100% 由 YAML 决定）与插件聚合车道已有的"local provider"完全同形，而内核侧没有任何 TokenRouter 概念，也没有托管 Python 服务的既有机制。

---

## 一、TokenRouter 是什么（源码实证）

NeurIPS 2026 接收的 token 级模型路由推理框架（arXiv 2610.12242，仓库 `thu-nics/TokenRouter`）。它把多个不同规模 LLM 装进同一份显存，在推理过程中**逐 token 决定当前请求该由哪个模型继续**，从而用平均成本远低于大模型的代价拿到接近大模型的质量。

### 对外契约（这是接入方案的唯一事实基础）

- 一个进程 = 一份 YAML = 一个 OpenAI 兼容 HTTP 服务（`uvicorn.run(app, port=entry_model.port)`）。
- 官方示例统一是 `OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")` + `model="default"`。
- **请求体里的 `model` 字段对路由零影响**：`tokenrouter/entrypoints/openai/serving_chat.py:27` 只做 `setattr(adapted_request, 'target_model', request.model)` 把字符串透传给 tokenizer manager；scheduler 侧（`managers/schedulers/base_scheduler.py`）完全不读 `target_model`。唯一严格校验点是 Python 引擎 `engine.py:169`（拒绝非 entry model），HTTP 层不校验。
- `GET /v1/models` 只返回**一个** id（`_global_state.tokenizer_manager.served_model_name`，未设 `served_model_name` 时落到 `model_path`）。不暴露各 subserver 名。
- 只有 entry model 的 scheduler 会吐流式输出（`base_scheduler.py:1021-1024`、`:2212`），所以 HTTP 端口必须属 entry model。
- YAML 的 `model:` 每项建 `ServerArgs(server_name=model_name, all_server_names=list(config["model"].keys()))`；`system.entry_model` 指定入口；路由由每项的 `scheduler_class` 决定。
- `config/supported_server_args.yaml` 是白名单，**未知键被静默丢弃**（`server_args_field_names` 过滤）——写配置文档只能列白名单键。
- 扩展接口：新算法 = 每个模型一个 scheduler 子类 + `managers/schedulers/__init__.py` 的 `SCHEDULER_REGISTRY` 加一行 + YAML 引用；算法实现 `route(batch, result)`，返回模型名列表或 shape `(batch.batch_size(),)` 的 tensor。

### 对插件接入的含义

1. **它就是一个 local provider**：协议、鉴权（无）、端口形态与 ollama/lmstudio/llamacpp 完全一致。
2. **model 字段无所谓**：插件聚合车道本来就会把 `body.model` 覆写成 catalog id（`src/aggregate.js:516`），两者不冲突。
3. **`/v1/models` 不能用作模型清单来源**：只回一个 id。模型 id 必须由用户按 YAML 的 entry model 名填写，或从 YAML 推导。
4. **reasoning 需要 YAML 侧配 `reasoning_parser`**：`serving_chat.py:56` 的门是 `reasoning_parser and request.separate_reasoning`，而插件三条 wire 都不发 `separate_reasoning`（`src/adapter.js:684`）→ 不配就是思考与正文混排；配了则插件两条解码路径（`src/stream.js:261`、`src/aggregate.js:609`）都能正确分流。
5. **tool_calls 门是 `tool_choice != "none" and request.tools`**：插件不发 `tool_choice`（缺省非 none），所以带 tools 的调用能正常解析。

---

## 二、为什么接插件而不是接内核

### 内核侧的证据

- 内核 LLM 服务契约（`class LlmRuntime`，`lib/main.js:5964-6600`）提供 `registerAdapter` / `registerConfigurableProviders` / `registerModelDiscovery` / `prepareCall` / `adapterStream`。内核本身**没有任何 TokenRouter 概念**，也没有托管外部 Python 服务的机制。
- 内核里唯一同构参照是 `@deepseek-ai/dsh-llm-pi-ai`（`lib/index.js:2860-2975`）：一个通用 OpenAI 兼容多 provider 路由插件，模式是「provider 目录 + 每条 route 一个 adapter + discovery」。接内核 = 再写一个 pi-ai 形态的插件，还要自带 scheduler 选择、GPU 检测、模型对配置 UI。
- 代价对比：内核方案是**从零建一条车道**（注册、目录、发现、UI、错误分类全部重写），插件方案是**往已有车道加第四条 local provider**（零新增架构）。

### 插件侧的独特增益（内核方案拿不到）

1. **combo 降级链**（`src/combos.js`）：TokenRouter 一行天然可与免费车道组合成 `fallback` 链——TokenRouter 不可用时自动落到匿名免费模型。内核没有这个概念。
2. **托管子进程的现成范式**（`src/headroom/index.js:76 startHeadroom` + `index.js:938 syncHeadroomOnce` + `index.js:998 scheduleHeadroomRestart`，15s→600s 指数退避、稳定 2 分钟重置、`timer.unref()`）：这正是托管 `python -m tokenrouter.launch_server` 需要的全部机制，改路径即可复用。
3. **设置页零前端改动**：`client.js:5255-5267` 已把 `category:'local'` 渲染为「本机」、`auth.kind:'none'` 渲染为「无需 Key」；`client.js:5426-5446` 的 provider 编辑器已有 enabled/key/baseUrl/models/test/discover/remove 全套。
4. **转发面白送**：`src/forward.js` 自带 OpenAI 兼容监听与 Anthropic 原生 `/v1/messages`，本机 TokenRouter 可被外部工具直接当 OpenAI 兼容端点用。
5. **不改内核 = 不破坏既有生态**：内核 adapter 契约正在被 9 个 `@deepseek-ai/dsh-llm-*` 插件消费，动它影响面不可控。

---

## 三、P0：注册表加一条 local provider（最小可用）

### 3.1 为什么不是"第四条"而是"第 76 条"

`src/providers/registry.json` 已有 75 家，其中 local 三条 = llamacpp / lmstudio / ollama（idx 70-72），**且 `tokenrouter`（idx 46）已被商业网关占用**（`https://api.tokenrouter.com/v1`，apikey，21 个模型）。新 id 必须另起，推荐 `tokenrouter-local`。

### 3.2 新增记录（照抄 local 三条的形状）

```json
{
  "id": "tokenrouter-local",
  "name": "TokenRouter (本地)",
  "category": "local",
  "auth": { "kind": "none", "env": null, "url": null },
  "baseUrl": "http://localhost:30000/v1",
  "format": "openai-chat",
  "models": [],
  "notes": "thu-nics/TokenRouter 本地服务：一个入口模型，按 YAML 的 scheduler 在多个模型间做 token 级路由。模型 id 由用户在设置里按 YAML 的 entry model 名填写。",
  "website": "https://github.com/thu-nics/TokenRouter",
  "local": true,
  "source": "fcc",
  "baseUrlOverrideEnv": "TOKENROUTER_BASE_URL"
}
```

三点说明：

- **不声明 `modelsUrl`**。`/v1/models` 只回一个 id（且落到 `model_path`），declare 了反而让用户点「发现」时得到一个误导性的单行清单。`src/aggregate.js:375` 对 `modelsUrl === null` 直接返回 null，`index.js` 的 `discover` 回 `{models:null, note:'this provider declares no model listing'}`——语义正确。
- **不声明 `headers`**。TokenRouter 的 `api_key` 是 YAML 可选项，且 `defaultHeaders` 只对 `auth.kind==='apiKey'` 加 Authorization（`src/providers/index.js:145`）。单机服务默认不设 key；若用户设了，见 §5.2 的限制说明。
- `models: []` 是刻意的：与三条 local 一致，靠用户在设置里填 id 或 P1 的自动推导。

### 3.3 必须同步的测试常量（否则 `npm run test:contributor` 红）

`scripts/registry-test.mjs`：

- `:52` `EXPECTED_PROVIDERS = 75` → `76`
- `:53` `EXPECTED_BY_CATEGORY.local = 3` → `4`
- `:148` `allowedEmpty` 加 `'tokenrouter-local'`（注释「three local servers」→「four local servers」）
- `:316-321` `listProviders({authKind:'none'})` 断言数组 `['llamacpp','lmstudio','ollama']` → 按字典序插入（`tokenrouter-local` > `ollama`，排尾）
- `:250-254` 可加一条 `chatEndpoint('tokenrouter-local') === 'http://localhost:30000/v1/chat/completions'`

同时修两处历史漂移（顺手项，先修后加）：`src/aggregate.js:7` 与 `client.js:5318` 注释写「76 providers」，README.md:67 / 313 / 357 写「75 家」——统一改为 76。

### 3.4 排序

`:136` 断言 CATEGORY_VALUES 序 + 同类 id 字典序。`tokenrouter-local` 属 local 段且字典序大于 `ollama`，所以**插在 ollama 之后、github_copilot（oauth 段）之前**。

### 3.5 用户使用方式（P0 完成后的实际体验）

1. 在 Linux + 多卡机器上 `python -m tokenrouter.launch_server --config_path config/r2r/Qwen3-0.6B_Qwen3-32B.yaml`（或先用 `config/me/` 三模型随机基线，不需要 router 权重）。
2. DSH 设置 → 聚合渠道 → 找到「TokenRouter (本地)」→ 启用 → baseUrl 填 `http://localhost:30000/v1` → models 填 entry model 名（如 `Qwen/Qwen3-0.6B`，或任何你想在 picker 里看到的名字）。
3. 该行进入 catalog，id 形如 `tokenrouter-local/Qwen/Qwen3-0.6B`，channel `aggregate`，availability 恒 `available`（aggregate 不探活，`src/core/runtime.js:272`：一次探活 = 一次真实 GPU 生成）。
4. 可把它加进任意 combo 的 `fallback` 链。

### 3.6 P0 的坑（必须写进文档）

- **`baseUrl` 覆盖不会改写 `modelsUrl`**：`src/aggregate.js:127` 只在 declared 以 registryBase 开头时才改写。TokenRouter 不声明 `modelsUrl`，所以无影响——但这解释了为什么不能顺手 declare。
- **`/v1/models` 只回一个 id**，所以 discover 不可用作清单来源（见上）。
- **egress outlet 会把 localhost 请求送去代理**：`src/egress.js:104 egressFetch` 有 relay 时把所有 http(s) 目标（含 `http://localhost:30000/v1`）改写进 `x-ofm-egress-target` 走 mihomo。用户若同时开了代理 outlet，TokenRouter 请求会被绕出国。方案：P0 文档里写明「TokenRouter 与代理 outlet 互斥」；P1 可在 `src/egress.js` 加回环目标旁路（`src/trust.js:23` 已有 `LOOPBACK_NAMES` 可复用）。

---

## 四、P1：托管子进程 + 配置面（推荐落地）

P1 不改变 P0 的注册表形态，而是在插件里新增一个 TokenRouter 管理模块，把"用户自己开终端跑 python"变成"DSH 里点一下"。

### 4.1 新模块 `src/tokenrouter/`（新增约 300 行）

| 文件 | 职责 |
|---|---|
| `src/tokenrouter/manager.js` | 生命周期：spawn / 状态 / 退出 / 重启退避。完全照抄 `src/headroom/index.js:76 startHeadroom` 的骨架 |
| `src/tokenrouter/config.js` | 读写用户目录下的 YAML（`${dataDir}/tokenrouter/config.yaml`），提供模板 |
| `src/tokenrouter/probe.js` | 轮询 `/get_model_info`（`tr_http_server.py:325`）判断就绪；**不用 `/v1/chat/completions` 探活**（那是一次真实生成） |

### 4.2 设置项（`src/store.js`）

```js
tokenrouter: {
  enabled: false,
  pythonPath: '',        // 空 = PATH 上的 python3
  configPath: '',        // 空 = ${dataDir}/tokenrouter/config.yaml
  port: 30000,
  autoStart: true,
  state: 'stopped',      // 由 manager 回写：stopped|starting|ready|error
  detail: '',            // 最后一条可读错误
  pid: 0,
}
```

沿用 headroom 的设置面形态：`index.js:1260-1301 tokenSaver.get/set` + `client.js:5753-5800` 的「一个开关 + 一个状态行 + 一个路径输入」。

### 4.3 关键实现点

1. **进程发现与启动**：`findHeadroomBinary` 同款三段（显式路径 → PATH → `python3 -m`）→ 校验 `python -m tokenrouter.launch_server --help` 能跑（否则给出 `pip install -e .` 提示）→ spawn `python -m tokenrouter.launch_server --config_path <yaml>`。
2. **端口占用处理**：`portAnswers` 已占用则 `joinHandle` 不抢（headroom 原样语义）——用户自己在终端起过时插件只接管状态展示。
3. **就绪判定**：轮询 `GET http://127.0.0.1:<port>/get_model_info`，直到 200 或超时；**绝不 POST /v1/chat/completions**。超时上限参考 `READY_TIMEOUT_MS = 15_000` 放大到 120s（sglang 多模型 warmup 很慢，`_execute_server_warmup` 有 120×1s 轮询）。
4. **重启退避**：`scheduleHeadroomRestart` 的阶梯（15s → 600s，稳定 `HEADROOM_STABLE_MS=120_000` 后重置）+ `timer.unref?.()` + disposer 清 timer。**1 分钟内同一指纹失败不重试**（`index.js:960`）。
5. **模型 id 自动推导**：从 YAML 读 `system.entry_model` 与其 `model_path`，回填 `aggregate.providers['tokenrouter-local'].models`（`index.js:1154` 的 save 面已有 models 写入 + `syncAggregateCatalog()`）。这样 picker 里出现的名字与 YAML 一致，不需要用户猜。
6. **路由算法提示**：从 YAML 各 `model[*].scheduler_class` 汇总一行可读文本（如 `r2r_slm + r2r_llm → R2R`），显示在状态行。只读展示，不生成 YAML——生成 YAML 属于 P2。
7. **egress 旁路**：`src/egress.js:104 egressFetch` 加一条——目标 host ∈ `LOOPBACK_NAMES` 且无 relay 时本来就走直连；有 relay 时对回环目标跳过改写（复用 `src/trust.js:23`）。这是 P1 唯一需要改既有模块的地方，约 5 行。

### 4.4 P1 的边界

- **不生成 YAML**：算法选择、模型对、router 权重路径交给用户；插件只提供一份最小模板（entry model + 一个 scheduler + `mem_fraction_static`）。
- **不管多机**：`system.peer_endpoints` 是集群部署形态，插件只服务单机单进程。
- **不在插件内实现路由**：路由 100% 在 TokenRouter 侧，插件不参与、也不该看到 `route()` 的决策。

---

## 五、P2：把插件做成 TokenRouter 控制面（不推荐）

如果未来要做，形态是：DSH 设置页里选算法（R2R / CITER / Co-LLM / R-Stitch / GlimpRouter / ME）、选模型对、填 router 权重路径（HuggingFace `nics-efc/*`）、生成 YAML、启停服务、并在 catalog 行上展示「本会话有多少 token 走了小模型」。

**不推荐的理由**：

1. TokenRouter 的 `SCHEDULER_REGISTRY`（13 个 scheduler）与 `supported_server_args.yaml` 白名单是上游活跃演进面，插件镜像它 = 长期维护负担。
2. 每模型的 `scheduler_kwargs` 形状各不相同（r2r 要 `model_path` + `override_init_args`，me 要 `routing_probabilities`），通用 UI 要么过窄要么过宽。
3. 插件的价值主张是「免费/自带 Key 的模型路由」，不是「GPU 推理服务管理」；混入 P2 会让插件定位模糊。
4. 真正需要调参的用户会直接编辑 YAML——P1 的 `configPath` 设置项已经支持指向用户自己的文件。

---

## 六、两条路线的完整对比

| 维度 | 接插件（P0/P1） | 接 DSH 内核 |
|---|---|---|
| 新增架构 | 0（复用聚合车道 + headroom 托管范式） | 1 条全新车道（目录/注册/发现/UI/错误分类） |
| 代码量 | P0 ~20 行 / P1 ~300 行 | 600+ 行 |
| combo 降级链 | 有（TokenRouter 挂了自动落免费模型） | 无 |
| 设置页/前端改动 | 0（local 分类已渲染） | 需要新建 provider 面板 |
| 托管 Python 服务 | 有现成范式（headroom） | 无先例，需新造 |
| 对内核风险 | 零 | 动 `LlmRuntime` 契约，影响 9 个 `dsh-llm-*` 插件 |
| 多算法配置 UI | P2 才做（不推荐） | 同左 |
| 维护耦合 | 低（TokenRouter 对外只有一个 OpenAI 端点） | 高（要跟着上游 scheduler 演进） |

---

## 七、落地顺序

> 本节为**条件性方案**：仅在将来获得带 NVIDIA GPU 的 Linux 机器时才执行。当前不实施。

0. **前置门槛（不可跳过）**：确认目标机器有 NVIDIA GPU 且已装 CUDA + `pip install -e .`。TokenRouter 硬依赖 CUDA（见附录），没有 GPU 的机器上 P1 的托管子进程只会启动失败——插件侧必须把这类失败渲染成可读错误，而不是留下一个崩掉的子进程。
1. **P0**（半天）：registry 加一条 + 测试常量同步 + 修 5 处「75/76」漂移 + README 聚合小节补一段 + 在有 NVIDIA GPU 的机器上按 §3.5 手动验证一次真实 turn。
2. **P1**（2-3 天）：`src/tokenrouter/` 三文件 + store 设置项 + settings API + 状态行 UI + egress 回环旁路 + 一个 `scripts/tokenrouter-test.mjs`（用本地 stub 模拟 `/get_model_info` 与 SSE，不需要 GPU）。
3. **P2**：仅当出现明确的「用户在 DSH 里选算法」需求再做。

## 验证结果

- **未实施**（TokenRouter 硬依赖 NVIDIA GPU，本机 Apple M1 跑不起来，用户决定不落地）。
- 协议与接线结论全部来自源码取证；**推理侧实跑须在有 NVIDIA GPU 的 Linux 机器上按 §3.5 完成**。
- 插件侧接线（registry → aggregate catalog → combo → adapter stream）在本机可用 stub upstream 完整验证，模板见 `scripts/aggregate-wiring-test.mjs`（boot 时写 `aggregate.providers.<id>`，起 stubUpstream，读 `/api/our-free-model/summary` 的 catalog，再跑真实 turn）。

---

## 附录：TokenRouter 必须 NVIDIA GPU 吗？

**是，硬依赖。** 源码实证（本机 M1 上 grep 全量落盘副本得出）：

1. **启动链无设备分支**：`engine.py:440-490` 的 `_launch_subprocesses` 按 `base_gpu_id + i * gpu_id_step` 直接算出整数 `gpu_id` 传给每个 scheduler 子进程，全程没有 `device == 'cpu'` 之类的判定；`tr_engine.py:618` 用 `torch.cuda.mem_get_info(i)` 读每张卡的显存来算 KV cache 预算，这一行在无 CUDA 环境直接抛异常。
2. **上游就是 sglang**：`TokenRoutingEngine` 继承 `sglang.srt.entrypoints.EngineBase`，scheduler 子类继承 `sglang.srt.managers.scheduler.Scheduler`。sglang 的 CUDA 依赖不是可选项。
3. **多模型共享同一张卡靠 CUDA MPS**：README `:124-135` 明确「If multiple models share the same physical GPU, CUDA MPS is recommended for reproducing overlapping serving behavior」，并给出 `nvidia-cuda-mps-control -d` 启动命令。这是 NVIDIA 独有的机制。
4. **张量并行走 NCCL**：`tr_tokenrouter_server_args.py:40-50` 自动分配 `nccl_port`（NCCL 是 NVIDIA 集合通信库）；YAML 白名单里 `attention_backend` / `decode_attention_backend` / `prefill_attention_backend` 全是 CUDA kernel 选项。

**对方案的三个直接后果：**

- **P0/P1 的托管子进程只能在有 NVIDIA GPU 的机器上真正产出推理**。插件本身（TypeScript 侧）跨平台无影响，`src/tokenrouter/manager.js` 的启动失败路径要能给出可读错误（如「未检测到 CUDA / torch.cuda 不可用」），而不是让用户看到一个崩掉的子进程。
- **`config/me/` 三模型基线示例要求三张卡**（`base_gpu_id 0/1/2`），`config/r2r/` 的 0.6B+32B 示例是 32B 用 `tp_size: 2`。想单卡跑，需要自己写一份小模型的 YAML（两个 0.6B/1.7B + 同一个 `base_gpu_id` + `mem_fraction_static` 调小 + 开 CUDA MPS）。
- **本机（M1）永远无法冒烟**，所以 `scripts/tokenrouter-test.mjs` 必须用 stub 覆盖：伪造 `/get_model_info` 与 SSE 流来验证生命周期、就绪判定、模型 id 推导和 egress 旁路，GPU 路径交给 CI 或用户的 Linux 机器。
