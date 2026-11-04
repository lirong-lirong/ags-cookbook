package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	ags "github.com/tencentcloud/tencentcloud-sdk-go/tencentcloud/ags/v20250920"
	"github.com/tencentcloud/tencentcloud-sdk-go/tencentcloud/common"
	"github.com/tencentcloud/tencentcloud-sdk-go/tencentcloud/common/profile"
)

func testAPI(t *testing.T, handler http.HandlerFunc) *demoClient {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	p := profile.NewClientProfile()
	p.HttpProfile.Scheme = "HTTP"
	p.HttpProfile.Endpoint = strings.TrimPrefix(srv.URL, "http://")
	client, err := ags.NewClient(common.NewCredential("test-id", "test-key"), "ap-nanjing", p)
	if err != nil {
		t.Fatal(err)
	}
	return &demoClient{api: client, run: "test-run"}
}

func TestStartPreservesInstanceVolumeAndTemplatePreference(t *testing.T) {
	var body struct {
		ToolId, ClientToken string
		Metadata            []struct{ Name, Value string }
		StorageMounts       []struct {
			Name, MountPath string
			Volume          struct{ VolumeId string }
		}
	}
	d := testAPI(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-TC-Action") != "StartSandboxInstance" {
			t.Error("unexpected API action")
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"Response":{"Instance":{"InstanceId":"test-instance","Status":"RUNNING"},"RequestId":"test-request"}}`))
	})
	if _, err := d.start("test-tool", "test-volume", "a"); err != nil {
		t.Fatal(err)
	}
	if body.ToolId != "test-tool" || body.ClientToken != "test-run-a" {
		t.Fatal("tool selection or idempotency token was lost")
	}
	if len(body.StorageMounts) != 1 || body.StorageMounts[0].Volume.VolumeId != "test-volume" || body.StorageMounts[0].MountPath != "/mnt/data" {
		t.Fatal("instance-level VolumeId or mount path was lost")
	}
	found := false
	for _, m := range body.Metadata {
		if m.Name == "x-ags-prefer-launch-snapshot-v2" && m.Value == "true" {
			found = true
		}
	}
	if !found {
		t.Fatal("template v2 preference was lost")
	}
}

func TestVolumeReadyRejectsOtherBackends(t *testing.T) {
	d := testAPI(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"Response":{"VolumeSet":[{"VolumeId":"test-volume","Status":"ACTIVE","StorageType":"Cos","AccessMode":"ReadWriteOnce"}],"RequestId":"test-request"}}`))
	})
	if err := d.volumeReady("test-volume"); err == nil || !strings.Contains(err.Error(), "AgentCbs/ReadWriteOnce") {
		t.Fatalf("expected backend validation error; got %v", err)
	}
}

func TestDotenvPreservesExportedCredentials(t *testing.T) {
	t.Setenv("TENCENTCLOUD_SECRET_ID", "exported-test-id")
	f := filepath.Join(t.TempDir(), ".env")
	if err := os.WriteFile(f, []byte("TENCENTCLOUD_SECRET_ID=file-test-id\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := loadEnvironment(f); err != nil {
		t.Fatal(err)
	}
	if os.Getenv("TENCENTCLOUD_SECRET_ID") != "exported-test-id" {
		t.Fatal("dotenv replaced an exported credential")
	}
}
