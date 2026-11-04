# 二期模板快照与 AgentCBS 数据盘（Go）

[English](./README.md)

使用同一份二期模板启动沙箱，每次启动时指定 AgentCBS 数据盘。示例由 Go
完成建盘、挂载、写文件、停止实例、在新实例中重挂同一块盘并校验 SHA-256。
可以创建新盘，也可以使用已有数据卷。

默认使用我们的 code 镜像：

```text
ccr.ccs.tencentyun.com/ags-image/sandbox-code:latest
```

控制面使用腾讯云 SDK 的 `CommonRequest` 原样发送 `AgentCbs`、实例级
`StorageMounts` 等新字段，签名仍由 SDK 处理；数据面使用 AGS Go SDK。

## 前置条件

- Go 1.22+、Make。
- 当前账号的腾讯云凭证，并有目标地域的 AGS 权限。
- 账号和工具已开通二期模板自动制作与使用、AgentCBS 系统盘和 AgentCBS 数据卷。
- 已有二期模板就绪的自定义 ToolId；或具备创建工具并触发自动制作的权限。
- 如需创建新工具，准备自己的镜像拉取 RoleArn。

默认地域是南京 `ap-nanjing`，可在配置中修改。能力开通、模板就绪需要在平台确认；
请求中的偏好参数不会代替开通，也不保证每次命中模板。工具 `ACTIVE` 不等于模板 `READY`。

## 快速开始

```bash
git clone --branch feat/template-v2-agentcbs-go --single-branch \
  https://github.com/lirong-lirong/ags-cookbook.git
cd ags-cookbook/examples/template-v2-agentcbs-go
make setup
cp .env.example .env
```

在 `.env` 填写自己的凭证和已就绪的 `AGS_TOOL_ID`，执行：

```bash
make run
```

程序自动读取 `.env`，终端中已导出的环境变量优先。复用已有工具时不需要 `AGS_ROLE_ARN`。

### 没有工具时先创建

在 `.env` 中填写自己的 `AGS_ROLE_ARN`：

```bash
make create-tool
```

该命令使用默认 code 镜像、2 vCPU / 2 GiB 内存和 20 GiB 系统盘创建自定义工具，
轮询到 `ACTIVE` 后发起一次准备启动，再停止准备实例。账号和工具必须事先启用自动制作。

把输出的 `AGS_TOOL_ID` 保存到 `.env`，等待平台确认二期模板就绪后再执行 `make run`。
准备启动可能走冷启动，不作为二期模板恢复验收。

### 复用或更换数据盘

`AGS_VOLUME_ID` 留空时创建新 20 GiB AgentCBS 数据卷。程序输出卷 ID 后，
将其保存到 `.env`，下次运行即可复用，避免重复创建。

将 `.env` 中的 `AGS_VOLUME_ID` 改为另一块可用数据卷，再执行 `make run`，
即可保持同一工具、同一模板配置，改用另一块盘。每次只增加一个随机命名的测试文件，
不会格式化或清空已有数据卷。

## 执行过程与预期结果

1. 创建或选择数据卷，等待 `ACTIVE` 且未被挂载。
2. 启动实例 A，通过 `StorageMounts[].Volume.VolumeId` 挂载到 `/mnt/data`。
3. 检查真实挂载，写入本次运行独有的文件。
4. 停止 A，等待数据卷释放 RWO 占用。
5. 启动实例 B，挂回同一块盘，校验文件 SHA-256。
6. 停止 B，保留数据卷供后续复用。

```text
MOUNT=/dev/... ext4
WRITE_OK
MOUNT=/dev/... ext4
PERSISTENCE_OK
TEMPLATE_RESTORE_MARKER_MATCH
SUCCESS: two sandboxes used the selected data volume; checksum matched
```

本例创建的工具在 `/init` 前将随机标记写入 `/dev/shm/cookbook-boot-id`。
两个新实例读到相同标记，可辅助验证模板恢复。标记不同时，即使数据校验通过也会返回错误。
已有工具若没有此标记，只报告数据持久化结果，模板命中需在平台另行确认。

## 环境变量

| 变量 | 用途 | 默认值 |
| --- | --- | --- |
| `TENCENTCLOUD_SECRET_ID` | 腾讯云 SecretId | 必填 |
| `TENCENTCLOUD_SECRET_KEY` | 腾讯云 SecretKey | 必填 |
| `TENCENTCLOUD_TOKEN` | 临时凭证的会话 Token | 可选 |
| `TENCENTCLOUD_REGION` | 工具、实例与数据卷所在地域 | `ap-nanjing` |
| `AGS_TOOL_ID` | 二期模板已就绪的自定义工具 | `make run` 必填 |
| `AGS_VOLUME_ID` | 已有 AgentCBS RWO 数据卷 | 空则新建数据卷 |
| `AGS_ROLE_ARN` | 自己的镜像拉取角色 | 创建工具时必填 |
| `AGS_IMAGE` | 新建工具使用的镜像 | 上述 code 镜像 |
| `AGS_TOOL_NAME` | 新建工具的名称 | 自动生成唯一名称 |

启动请求偏好二期模板，并请求 AgentCBS 系统盘。本例关闭 monitor；复用已有工具时，
模板制作配置需与此一致。镜像、命令、资源规格和 monitor 的变化可能影响模板匹配。

二期恢复沿用内存状态，不重新执行启动命令。本例启动命令不访问外部数据盘，
依赖数据盘的业务工作在 `StartSandboxInstance` 成功返回后执行。挂载由平台完成，
客户端不额外调用挂盘接口。

## 数据保留和清理

函数正常返回或因错误返回时，会尝试停止本次创建的实例；实例还配置了 10 分钟生存时间。
工具和数据卷默认保留，供后续复用。

需要删除测试资源时，先核对 `.env` 中的 ID，再执行：

```bash
make delete-volume
make delete-tool
```

**DeleteVolume 会删除该卷上的全部数据。** 删除工具与删除数据卷是独立操作，
业务数据卷应按业务保留策略管理。

## 常见问题

- **能力或模板未就绪**：先确认该账号、地域和工具已开通相应能力，且二期模板已就绪。
- **数据盘被占用**：RWO 数据盘同一时刻只供一个实例使用；停止占用实例后，等待 `MountedInstanceId` 清空。
- **数据卷类型不符**：传入 AGS 管理的 `AgentCbs` RWO `vol-*`，不传 COS/CFS 卷或 CVM 云盘 ID。
- **启动标记不同**：本轮未证明两次复用了同一份内存快照，需确认模板就绪和配置一致。
- **创建请求超时**：保留输出的 `ClientToken` 和错误中的 `RequestId`，先核对结果再发起新的创建请求；程序不自动重放启动请求。

单个实例使用一块可写整盘，不设置 SubPath。并发实例使用独立数据卷。
更换数据卷作用于下一次实例启动，不在运行中的实例上替换盘。

## 本地验证

```bash
make check
```

原始流程已在南京使用默认 code 镜像验证新建数据卷、指定已有卷、ext4 实际挂载、
跨实例 SHA-256 校验和模板标记一致性。本地检查覆盖请求字段透传、存储类型校验、
环境变量优先级、vet 和编译。
