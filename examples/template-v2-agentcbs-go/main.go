package main

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/joho/godotenv"

	sandboxcode "github.com/TencentCloudAgentRuntime/ags-go-sdk/sandbox/code"
	"github.com/TencentCloudAgentRuntime/ags-go-sdk/tool/command"
	ags "github.com/tencentcloud/tencentcloud-sdk-go/tencentcloud/ags/v20250920"
	"github.com/tencentcloud/tencentcloud-sdk-go/tencentcloud/common"
	tchttp "github.com/tencentcloud/tencentcloud-sdk-go/tencentcloud/common/http"
	"github.com/tencentcloud/tencentcloud-sdk-go/tencentcloud/common/profile"
)

const defaultImage = "ccr.ccs.tencentyun.com/ags-image/sandbox-code:latest"

type object = map[string]interface{}
type resource struct {
	ToolId, InstanceId, VolumeId, Status, StatusReason, MountedInstanceId string
	StorageType, AccessMode                                               string
}
type reply struct {
	RequestId, ToolId         string
	Instance, Volume          resource
	SandboxToolSet, VolumeSet []resource
	Error                     *struct{ Code, Message string }
}
type demoClient struct {
	api *ags.Client
	run string
}

func env(name, fallback string) string {
	if s := os.Getenv(name); s != "" {
		return s
	}
	return fallback
}
func quote(s string) string { return "'" + strings.ReplaceAll(s, "'", "'\"'\"'") + "'" }

// CommonRequest preserves new API fields while the SDK handles TC3 signing.
func (d *demoClient) call(action string, body object) (reply, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	req := tchttp.NewCommonRequest("ags", "2025-09-20", action)
	if err := req.SetActionParameters(body); err != nil {
		return reply{}, err
	}
	req.SetContext(ctx)
	resp := tchttp.NewCommonResponse()
	begin := time.Now()
	if err := d.api.Send(req, resp); err != nil {
		return reply{}, fmt.Errorf("%s: %w", action, err)
	}
	var envelope struct{ Response reply }
	if err := json.Unmarshal(resp.GetBody(), &envelope); err != nil {
		return reply{}, err
	}
	r := envelope.Response
	fmt.Printf("%s RequestId=%s elapsed=%s\n", action, r.RequestId, time.Since(begin).Round(time.Millisecond))
	if r.Error != nil {
		return r, fmt.Errorf("%s: %s (RequestId=%s)", r.Error.Code, r.Error.Message, r.RequestId)
	}
	return r, nil
}

func (d *demoClient) start(tool, volume, label string) (string, error) {
	body := object{
		"ToolId": tool, "Timeout": "10m", "ClientToken": d.run + "-" + label,
		"Metadata": []object{
			{"Name": "x-ags-agent-cbs-system-disk", "Value": "true"},
			{"Name": "x-ags-prefer-launch-snapshot-v2", "Value": "true"},
			{"Name": "x-ags-monitor", "Value": "off"},
		},
	}
	if volume != "" {
		body["StorageMounts"] = []object{{
			"Name": "data", "MountPath": "/mnt/data",
			"Volume": object{"VolumeId": volume},
		}}
	}
	fmt.Printf("Start client_token=%s\n", body["ClientToken"])
	r, err := d.call("StartSandboxInstance", body)
	if err != nil {
		return "", err
	}
	if r.Instance.InstanceId == "" || r.Instance.Status != "RUNNING" {
		return r.Instance.InstanceId, fmt.Errorf("start returned status=%s; instance=%s", r.Instance.Status, r.Instance.InstanceId)
	}
	fmt.Println("InstanceId=" + r.Instance.InstanceId)
	return r.Instance.InstanceId, nil
}

func (d *demoClient) stop(id string) error {
	if id == "" {
		return nil
	}
	_, err := d.call("StopSandboxInstance", object{"InstanceId": id})
	return err
}

func (d *demoClient) exec(id, script string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	sbx, err := sandboxcode.Connect(ctx, id, sandboxcode.WithClient(d.api))
	if err != nil {
		return "", err
	}
	r, err := sbx.Commands.Run(ctx, script, &command.ProcessConfig{User: "root"}, nil)
	if err != nil {
		return "", err
	}
	if r.ExitCode != 0 {
		return "", fmt.Errorf("command exit=%d stderr=%s", r.ExitCode, r.Stderr)
	}
	out := string(r.Stdout)
	fmt.Print(out)
	return out, nil
}

func wait(poll func() (bool, error)) error {
	deadline := time.Now().Add(5 * time.Minute)
	for time.Now().Before(deadline) {
		done, err := poll()
		if err != nil {
			return err
		}
		if done {
			return nil
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("resource did not become ready within five minutes")
}

func (d *demoClient) volumeReady(id string) error {
	return wait(func() (bool, error) {
		r, err := d.call("DescribeVolumeList", object{"VolumeIds": []string{id}, "Limit": 1})
		if err != nil {
			return false, err
		}
		if len(r.VolumeSet) != 1 {
			return false, fmt.Errorf("volume not found: %s", id)
		}
		v := r.VolumeSet[0]
		if v.StorageType != "AgentCbs" || v.AccessMode != "ReadWriteOnce" {
			return false, fmt.Errorf("volume must be AgentCbs/ReadWriteOnce; got %s/%s", v.StorageType, v.AccessMode)
		}
		if v.Status == "FAILED" {
			return false, fmt.Errorf("volume failed: %s", v.StatusReason)
		}
		return v.Status == "ACTIVE" && v.MountedInstanceId == "", nil
	})
}

// Create a custom tool using the default code image. External storage is absent
// from its startup command; each cold boot creates a fresh in-memory marker.
func (d *demoClient) createTool() (err error) {
	role := os.Getenv("AGS_ROLE_ARN")
	if role == "" {
		return fmt.Errorf("set AGS_ROLE_ARN to your image-pull role")
	}
	r, err := d.call("CreateSandboxTool", object{
		"ToolName": env("AGS_TOOL_NAME", "code-cbs-"+d.run), "ToolType": "custom",
		"RoleArn": role, "ClientToken": d.run + "-tool",
		"NetworkConfiguration": object{"NetworkMode": "SANDBOX"},
		"CustomConfiguration": object{
			"Image": env("AGS_IMAGE", defaultImage), "ImageRegistryType": "personal",
			"Command":   []string{"/bin/sh"},
			"Args":      []string{"-c", "cat /proc/sys/kernel/random/uuid > /dev/shm/cookbook-boot-id; exec /init"},
			"Resources": object{"CPU": "2", "Memory": "2Gi", "Storage": "20Gi"},
			"Ports":     []object{{"Name": "envd", "Port": 49983, "Protocol": "TCP"}},
			"Probe":     object{"HttpGet": object{"Path": "/health", "Port": 49983, "Scheme": "HTTP"}, "ReadyTimeoutMs": 30000, "ProbeTimeoutMs": 1000, "ProbePeriodMs": 1000, "SuccessThreshold": 1, "FailureThreshold": 100},
		},
	})
	if err != nil {
		return err
	}
	tool := r.ToolId
	if tool == "" {
		return fmt.Errorf("empty ToolId")
	}
	fmt.Printf("export AGS_TOOL_ID=%s\n", quote(tool))
	if err = wait(func() (bool, error) {
		x, e := d.call("DescribeSandboxToolList", object{"ToolIds": []string{tool}, "Limit": 1})
		if e != nil {
			return false, e
		}
		if len(x.SandboxToolSet) != 1 {
			return false, fmt.Errorf("tool not found")
		}
		t := x.SandboxToolSet[0]
		if t.Status == "FAILED" {
			return false, fmt.Errorf("tool failed: %s", t.StatusReason)
		}
		return t.Status == "ACTIVE", nil
	}); err != nil {
		return err
	}
	// This request triggers automatic template preparation only when enabled
	// for this account/tool by the platform. ACTIVE alone is not template READY.
	id, err := d.start(tool, "", "prepare")
	if id != "" {
		defer func() { err = errors.Join(err, d.stop(id)) }()
	}
	if err != nil {
		return err
	}
	fmt.Println("Tool created and preparation requested. Confirm template v2 is READY before running demo.")
	return nil
}

func (d *demoClient) demo() (err error) {
	tool := os.Getenv("AGS_TOOL_ID")
	if tool == "" {
		return fmt.Errorf("set AGS_TOOL_ID to your READY-template tool")
	}
	volume := os.Getenv("AGS_VOLUME_ID")
	if volume == "" {
		r, e := d.call("CreateVolume", object{
			"VolumeName": "code-data-" + d.run, "ClientToken": d.run + "-volume",
			"StorageType": "AgentCbs", "Storage": object{"AgentCbs": object{"Capacity": "20Gi"}},
			"AccessMode": "ReadWriteOnce",
		})
		if e != nil {
			return e
		}
		volume = r.Volume.VolumeId
		if volume == "" {
			return fmt.Errorf("empty VolumeId")
		}
	}
	fmt.Printf("export AGS_VOLUME_ID=%s\n", quote(volume))
	fmt.Println("The data volume is retained; this run stops only the instances it creates.")
	if err = d.volumeReady(volume); err != nil {
		return err
	}
	current := ""
	defer func() {
		if current != "" {
			err = errors.Join(err, d.stop(current))
		}
	}()
	current, err = d.start(tool, volume, "a")
	if err != nil {
		return err
	}
	file := "/mnt/data/cookbook-" + d.run + ".txt"
	data := "persistent data: " + d.run
	sum := fmt.Sprintf("%x", sha256.Sum256([]byte(data)))
	checkMount := `set -eu; awk '$2=="/mnt/data" {found=1; print "MOUNT=" $1 " " $3} END {exit !found}' /proc/mounts; `
	boot := `if [ -f /dev/shm/cookbook-boot-id ]; then printf 'BOOT='; cat /dev/shm/cookbook-boot-id; fi; `
	first, err := d.exec(current, checkMount+"printf '%s' "+quote(data)+" > "+quote(file)+"; sync; "+boot+"echo WRITE_OK")
	if err != nil {
		return err
	}
	if err = d.stop(current); err != nil {
		return err
	}
	current = ""
	if err = d.volumeReady(volume); err != nil {
		return err
	}
	current, err = d.start(tool, volume, "b")
	if err != nil {
		return err
	}
	second, err := d.exec(current, checkMount+"test \"$(sha256sum "+quote(file)+" | awk '{print $1}')\" = "+quote(sum)+"; "+boot+"echo PERSISTENCE_OK")
	if err != nil {
		return err
	}
	bootID := func(out string) string {
		for _, line := range strings.Split(out, "\n") {
			if strings.HasPrefix(line, "BOOT=") {
				return strings.TrimPrefix(line, "BOOT=")
			}
		}
		return ""
	}
	a, b := bootID(first), bootID(second)
	if a != "" && b != "" {
		if a != b {
			return fmt.Errorf("data persisted, but boot markers differ: snapshot restore was not demonstrated; confirm template readiness and retry")
		}
		fmt.Println("TEMPLATE_RESTORE_MARKER_MATCH")
	} else {
		fmt.Println("Persistence verified. This tool has no boot marker; confirm template restoration with the platform.")
	}
	fmt.Println("SUCCESS: two sandboxes used the selected data volume; checksum matched")
	return nil
}

func loadEnvironment(path string) error {
	err := godotenv.Load(path)
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("load .env: %w", err)
	}
	return nil
}

func run() error {
	if err := loadEnvironment(".env"); err != nil {
		return err
	}
	id, key := os.Getenv("TENCENTCLOUD_SECRET_ID"), os.Getenv("TENCENTCLOUD_SECRET_KEY")
	if id == "" || key == "" {
		return fmt.Errorf("set TENCENTCLOUD_SECRET_ID and TENCENTCLOUD_SECRET_KEY")
	}
	cred := common.NewTokenCredential(id, key, os.Getenv("TENCENTCLOUD_TOKEN"))
	p := profile.NewClientProfile()
	p.HttpProfile.Endpoint = "ags.tencentcloudapi.com"
	p.HttpProfile.ReqTimeout = 90
	client, err := ags.NewClient(cred, env("TENCENTCLOUD_REGION", "ap-nanjing"), p)
	if err != nil {
		return err
	}
	var random [8]byte
	if _, err = rand.Read(random[:]); err != nil {
		return err
	}
	d := demoClient{api: client, run: hex.EncodeToString(random[:])}
	mode := "demo"
	if len(os.Args) > 1 {
		mode = os.Args[1]
	}
	switch mode {
	case "create-tool":
		return d.createTool()
	case "demo":
		return d.demo()
	case "delete-volume":
		v := os.Getenv("AGS_VOLUME_ID")
		if v == "" {
			return fmt.Errorf("set AGS_VOLUME_ID")
		}
		_, err = d.call("DeleteVolume", object{"VolumeId": v})
		return err
	case "delete-tool":
		t := os.Getenv("AGS_TOOL_ID")
		if t == "" {
			return fmt.Errorf("set AGS_TOOL_ID")
		}
		_, err = d.call("DeleteSandboxTool", object{"ToolId": t})
		return err
	default:
		return fmt.Errorf("usage: go run . [create-tool|demo|delete-volume|delete-tool]")
	}
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
