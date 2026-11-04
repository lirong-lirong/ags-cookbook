# Template snapshot v2 with AgentCBS data volumes (Go)

[中文说明](./README_zh.md)

Start a template-v2 sandbox with a selected AgentCBS data volume, write a file,
stop the sandbox, and attach the same volume to a new sandbox to verify its
SHA-256 checksum. The example can create a volume or reuse an existing one.

The default image is:

```text
ccr.ccs.tencentyun.com/ags-image/sandbox-code:latest
```

The Go program handles the control-plane requests and guest commands. It uses
the Tencent Cloud SDK's `CommonRequest` to preserve new fields such as
`AgentCbs` and instance-level `StorageMounts`; SDK signing is unchanged.

## Prerequisites

- Go 1.22+ and Make.
- Tencent Cloud credentials with AGS access in the target region.
- Template snapshot v2, AgentCBS system disks, and AgentCBS data volumes enabled
  for your account and tool in that region.
- A custom tool with a READY v2 template, or permission to create a tool and
  request automatic template preparation.
- For tool creation, your own image-pull role ARN.

This example defaults to `ap-nanjing`. Feature enablement and template readiness
must be confirmed for your account; the metadata preference does not enable
these features or guarantee a template hit. A tool becoming `ACTIVE` does not
mean its v2 template is `READY`.

## Quick start

```bash
git clone --branch feat/template-v2-agentcbs-go --single-branch \
  https://github.com/lirong-lirong/ags-cookbook.git
cd ags-cookbook/examples/template-v2-agentcbs-go
make setup
cp .env.example .env
```

Edit `.env` with your credentials and a prepared `AGS_TOOL_ID`, then run:

```bash
make run
```

The program reads `.env` automatically. Exported environment variables take
precedence. With an existing tool, `AGS_ROLE_ARN` is not required.

### Create a tool if needed

Set your `AGS_ROLE_ARN` in `.env`, then:

```bash
make create-tool
```

This creates a custom tool using the default code image, 2 vCPU, 2 GiB memory,
and a 20 GiB system disk. It waits for the tool to become `ACTIVE`, sends one
preparation start request, and stops that instance. Automatic template creation
must already be enabled by the platform for this account/tool.

Save the printed `AGS_TOOL_ID` in `.env`. After the platform confirms its v2
template is ready, run `make run`. Preparation can cold-start; it is not a
template-restore verification.

### Select or reuse a data volume

Leave `AGS_VOLUME_ID` empty to create a new 20 GiB AgentCBS volume. The program
prints its ID; save it in `.env` to reuse it on subsequent runs.

To select a different data volume, set its `AGS_VOLUME_ID` in `.env` and run
`make run` again. Keep the same tool and template configuration. Each run adds
one uniquely named test file; it does not format or clear the volume.

## Expected result

Each run:

1. Creates or selects a volume and waits until it is `ACTIVE` and unmounted.
2. Starts sandbox A with `StorageMounts[].Volume.VolumeId` at `/mnt/data`.
3. Checks the actual mount and writes a uniquely named file.
4. Stops A and waits for the volume's RWO ownership to clear.
5. Starts sandbox B with the same volume and checks the file's SHA-256 checksum.
6. Stops B and retains the volume.

```text
MOUNT=/dev/... ext4
WRITE_OK
MOUNT=/dev/... ext4
PERSISTENCE_OK
TEMPLATE_RESTORE_MARKER_MATCH
SUCCESS: two sandboxes used the selected data volume; checksum matched
```

Tools created by this example write a random marker to
`/dev/shm/cookbook-boot-id` before `/init`. Matching markers across two new
instances support template-restore verification. Different markers make the
demo return an error even if persistence passed. For an existing tool without
this marker, only data persistence is reported; confirm template usage with the
platform separately.

## Configuration

| Variable | Meaning | Default |
| --- | --- | --- |
| `TENCENTCLOUD_SECRET_ID` | Tencent Cloud SecretId | Required |
| `TENCENTCLOUD_SECRET_KEY` | Tencent Cloud SecretKey | Required |
| `TENCENTCLOUD_TOKEN` | Session token for temporary credentials | Optional |
| `TENCENTCLOUD_REGION` | Region for the tool, instances, and volume | `ap-nanjing` |
| `AGS_TOOL_ID` | Custom tool with a ready v2 template | Required for `make run` |
| `AGS_VOLUME_ID` | Existing AgentCBS RWO data volume | Empty creates a new volume |
| `AGS_ROLE_ARN` | Your image-pull role | Required for tool creation |
| `AGS_IMAGE` | Image used when creating a tool | Default code image above |
| `AGS_TOOL_NAME` | Name used when creating a tool | Generated unique name |

The startup request prefers v2 and requests an AgentCBS system disk. Monitor is
disabled for this example; a reused tool must have a template prepared with the
same configuration. Image, command, resource, and monitor changes can affect
template matching.

V2 restore resumes memory and does not rerun the startup command. The example
keeps disk access out of the tool startup command and runs data-dependent work
only after `StartSandboxInstance` succeeds. The platform handles attachment;
the client does not call a separate attach API.

## Cleanup and data retention

Normal returns attempt to stop each instance created by the program, including
returns caused by errors. Instances also have a 10-minute lifetime. The tool
and data volume are retained for reuse.

To delete test resources, check the IDs in `.env` and run:

```bash
make delete-volume
make delete-tool
```

**Deleting a volume removes all data on that volume.** Deleting a tool and
deleting a volume are separate operations. Retain business volumes according
to your application's policy.

## Troubleshooting

- **Feature or template unavailable:** confirm access and template readiness
  for this account, region, and tool before running the demo.
- **Volume busy:** a `ReadWriteOnce` volume can be used by one instance at a
  time. Stop its current owner and wait for `MountedInstanceId` to clear.
- **Wrong volume backend:** pass an AGS-managed `AgentCbs` RWO `vol-*` volume,
  not a COS/CFS volume or a CVM disk ID.
- **Boot markers differ:** the two starts did not demonstrate reuse of the same
  memory snapshot. Check template readiness and matching configuration.
- **Request timeout:** retain the printed `ClientToken` and error `RequestId`,
  then reconcile the operation before sending a new create request. The demo
  does not automatically replay starts.

Each instance uses one whole-disk, writable volume without `SubPath`. Give
concurrent instances separate volumes. Selecting a new volume applies to the
next instance start, not to an already running sandbox.

## Local validation

```bash
make check
```

The underlying flow was validated in Nanjing using the default code image:
new-volume creation, reuse of an existing volume, actual ext4 mounts, checksum
persistence, and matching template boot markers. Local checks cover request
field preservation, backend validation, environment precedence, vet, and build.
