# 在 OSWorld 镜像上试用 envd 与 NetworkPolicy

这个示例给两个 OSWorld 基础镜像附加 envd，构建成镜像后推送，并在创建沙箱实例时
附加出口网络策略，最后验证策略确实生效。用 `--flavor` 选择基础镜像：

| flavor | 基础镜像 |
| --- | --- |
| `osworld1` | `osworld1-base`，OSWorld 桌面任务与 Benchmark |
| `osworld2` | `osworld2-base`，含需要 Docker 的任务 |

## 为什么需要 envd

NetworkPolicy 只管理沙箱到外部的出口连接。HTTPS 的 L7 匹配依赖 MITM，平台要通过
沙箱内**以 root 运行、监听 49983** 的 envd 注入 CA 证书。因此 Tool 必须：

- 在 `Ports` 中声明 49983，并以 `/health` 作为就绪探针；
- 让 envd 以 root 启动，同时保留 `/sbin/init`（systemd）作为 PID 1，桌面与
  OSWorld API 才能正常工作；
- 在 `StartSandboxInstance` 时传入 `NetworkPolicy`，运行中可用
  `UpdateSandboxInstance` 替换策略（更新后调用 Describe 回读）。

## 准备

- Python 3.11+ 与 `uv`，具有 AGS 权限的腾讯云凭证。
- Docker 或 Podman，并且对两个基础镜像所在的源仓库有推送权限。
- NetworkPolicy 需要先开通并使用支持该能力的运行环境。

```bash
make setup
# 编辑 .env：填写腾讯云凭证、AGS_REGION，以及 CUSTOM_IMAGE_OSWORLD1/2
```

## 构建并推送两个带 envd 的镜像

[Dockerfile.envd](Dockerfile.envd) 从固定 digest 的 envd 镜像复制 `/usr/bin/envd`，
并用 [envd.service](envd.service) 让 systemd 以 root 启动 49983；`/sbin/init`、
桌面用户和 OSWorld 服务保持不变。构建时会执行 `envd -version` 校验版本。

派生镜像推回**源仓库**（`ccr.ccs.tencentyun.com/ags-image/osworld{1,2}-base`），
并使用专用 tag，例如 `envd-np-v1`：

- 基础层已在同一 repository，推送时只需上传包含 envd 的小层，不必重传 13–20 GiB；
- 绝不覆盖基础镜像的 tag；要更新镜像就换一个新 tag（`envd-np-v2`），便于回退；
- 脚本会拒绝把 `CUSTOM_IMAGE_*` 设成与基础镜像相同的 tag。

```bash
make build build2    # 基于两个基础镜像执行 Dockerfile.envd
make push push2      # 推送前先登录源仓库
```

没有 Docker 时可设置 `CONTAINER_TOOL=podman`。

## 运行与验证

```bash
make run    # OSWorld1：建 Tool、起实例、附加策略、验证、输出 noVNC 链接
make run2   # OSWorld2：同样流程，独立状态
```

`make run` 会检查：envd 以 root 运行、`/health` 可用、平台 CA 是否注入、
策略允许的域名可访问、被拒绝的域名返回 403 或连接被阻断。报告写入
`.state/<region>/<flavor>/policy-report.json`。终端输出的 noVNC 链接含访问 token，
请勿公开分享。

## 修改策略与运行中更新

[network-policy.json](network-policy.json) 默认拒绝所有出口流量，只放行 GitHub 的
HTTPS 地址，并显式拒绝 `api.openai.com` 与 `example.com`。`Rule` 按数组顺序匹配，
第一条命中的规则生效；省略 `DefaultDecision` 时默认为 deny。

```bash
make verify          # 对当前实例重新验证
make verify2
make update          # UpdateSandboxInstance 运行中替换策略，回读后再验证
make update2
# 也可用 --policy-file 指定其它策略：
uv run python scripts/demo.py update --flavor osworld1 --policy-file my-policy.json
```

验证脚本会从策略中自动挑选一个具体的 allow 域名和 deny 域名做探测，
所以自定义策略时请保留一个非通配的 allow `Hosts` 项。

## 状态与清理

`.state/<region>/<flavor>/` 保存 Tool/实例 ID 和验证报告，两个 flavor 相互独立。

```bash
make clean   # 停止两个实例并删除本示例创建的 Tool
make check   # 单元测试
```

实例的默认超时是一小时，`make clean` 也会清理两个 flavor 的状态。

## 已知边界

- 报告里的 `ca_extra` 表示 envd 是否落地 `e2b-ca.crt` 持久化副本；本示例验证的是
  Host/端口级放行与拒绝。若要用 HTTPS 的 Path、Header、Body 匹配，需另行确认
  MITM 与客户端对代理 CA 的信任。
- 首次冷启动时桌面接口可能晚于 envd `/health` 数分钟才可用，脚本会等待最多 360 秒；
  本机 DNS 解析数据面域名异常时也会表现为等待超时，重跑 `make run` 即可复用已创建的实例。
- `ROLE_ARN` 只在私有仓库拉取镜像时需要；源仓库的公开镜像不需要。
