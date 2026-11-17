package main

import (
	"encoding/json"
	"runtime"
	"testing"
)

func linuxManifest(backend string) Manifest {
	return Manifest{
		Engine: "mlx", Schema: 1, Model: "test", Revision: "sha", Development: true,
		OS: "linux", Arch: "amd64", Backend: backend, Context: 2048, MinRAM: 1,
		Files: []File{{Name: "config.json", Size: 1, SHA256: digest([]byte("x")), URL: "https://example.com/config.json"}},
		Entry: "config.json", RuntimeSHA: digest([]byte("r")),
		RuntimeFiles: []File{{Name: runtimeServer, Size: 1, SHA256: digest([]byte("r")), Executable: true}},
	}
}

func encodeManifest(t *testing.T, m Manifest) []byte {
	t.Helper()
	data, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal manifest: %v", err)
	}
	return data
}

func TestDecodeManifestAcceptsLinuxTargets(t *testing.T) {
	for _, backend := range []string{"cpu", "rocm"} {
		m := linuxManifest(backend)
		if _, err := decodeManifest(encodeManifest(t, m)); err != nil {
			t.Errorf("linux/amd64/%s manifest rejected: %v", backend, err)
		}
	}
}

func TestDecodeManifestRejectsUnknownTargets(t *testing.T) {
	for _, m := range []Manifest{
		func() Manifest { m := linuxManifest("cpu"); m.Arch = "arm64"; return m }(),
		func() Manifest { m := linuxManifest("metal"); return m }(),
		func() Manifest { m := linuxManifest("cpu"); m.OS = "darwin"; m.Arch = "arm64"; return m }(),
	} {
		if _, err := decodeManifest(encodeManifest(t, m)); err == nil {
			t.Errorf("manifest %s/%s/%s should be rejected", m.OS, m.Arch, m.Backend)
		}
	}
}

func TestDecodeManifestKeepsMetalTarget(t *testing.T) {
	m := Manifest{
		Engine: "mlx", Schema: 1, Model: "test", Revision: "sha", Development: true,
		OS: "darwin", Arch: "arm64", Backend: "metal", Context: 2048, MinRAM: 1,
		Files: []File{{Name: "config.json", Size: 1, SHA256: digest([]byte("x")), URL: "https://example.com/config.json"}},
		Entry: "config.json", RuntimeSHA: digest([]byte("r")),
		RuntimeFiles: []File{{Name: runtimeServer, Size: 1, SHA256: digest([]byte("r")), Executable: true}},
	}
	if _, err := decodeManifest(encodeManifest(t, m)); err != nil {
		t.Fatalf("darwin/arm64/metal manifest rejected: %v", err)
	}
}

func TestPreflightAcceptsLinuxManifest(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skip("linux/amd64 only")
	}
	m := linuxManifest("cpu")
	if ram, err := linuxMemoryBytes(); err != nil {
		t.Fatalf("linuxMemoryBytes: %v", err)
	} else if ram < 1<<30 {
		t.Fatalf("implausible memory report: %d", ram)
	}
	if err := preflight(m); err != nil {
		t.Fatalf("preflight rejected a linux/amd64/cpu manifest: %v", err)
	}
}
