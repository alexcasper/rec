// rec-6ek bisect: LD_PRELOAD shim so OLD megakernel builds (whose mid_session_create copies the whole bank from a
// host pointer) load the 21 GiB bank by streaming. The ONE hipMemcpy H2D whose size == $BANK_STREAM_BYTES is served
// from $BANK_STREAM_FILE in 256 MiB chunks (src pointer ignored); every other call passes through untouched.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
typedef hipError_t (*memcpy_t)(void*, const void*, size_t, hipMemcpyKind);
extern "C" hipError_t hipMemcpy(void* dst, const void* src, size_t n, hipMemcpyKind kind) {
    static memcpy_t real = 0;
    if (!real) real = (memcpy_t)dlsym(RTLD_NEXT, "hipMemcpy");
    const char* bs = getenv("BANK_STREAM_BYTES");
    const char* bf = getenv("BANK_STREAM_FILE");
    if (bs && bf && kind == hipMemcpyHostToDevice && n == strtoull(bs, 0, 10)) {
        const size_t CH = (size_t)256 << 20;
        char* buf = (char*)malloc(CH);
        int fd = open(bf, O_RDONLY);
        if (!buf || fd < 0) { fprintf(stderr, "bank_stream_shim: open/malloc failed\n"); return hipErrorInvalidValue; }
        size_t done = 0;
        while (done < n) {
            size_t want = n - done < CH ? n - done : CH, got = 0;
            while (got < want) { ssize_t r = pread(fd, buf + got, want - got, done + got); if (r <= 0) { fprintf(stderr, "bank_stream_shim: short read\n"); return hipErrorInvalidValue; } got += r; }
            hipError_t e = real((char*)dst + done, buf, want, kind);
            if (e) { fprintf(stderr, "bank_stream_shim: hipMemcpy err %d at %zu\n", e, done); return e; }
            posix_fadvise(fd, done, want, POSIX_FADV_DONTNEED);
            done += want;
        }
        close(fd); free(buf);
        fprintf(stderr, "bank_stream_shim: streamed %zu bytes from %s in 256 MiB chunks\n", n, bf);
        return hipSuccess;
    }
    return real(dst, src, n, kind);
}
