/* Preloaded into the FSDS simulator by scripts/sim.sh (LD_PRELOAD), built by `make build`.
 *
 * The simulator's RPC server (rpclib) never sets TCP_NODELAY. A 3.7 MB camera image goes out in several writes, and
 * with Nagle's algorithm the last short segment waits until the bridge has acknowledged the data before it, while
 * Linux delays that acknowledgement by up to 40 ms. In about one image in eight the bridge got its image 40 ms late,
 * and since every round of camera requests waits for the slowest camera, the cameras delivered 12 instead of 14
 * images per second. Setting TCP_NODELAY on every socket the simulator accepts removes the wait; the binary itself
 * can't be rebuilt here.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

static void no_delay(int fd) {
    int one = 1;
    if (fd >= 0) setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

int accept(int fd, struct sockaddr *addr, socklen_t *len) {
    static int (*real_accept)(int, struct sockaddr *, socklen_t *);
    if (!real_accept) real_accept = dlsym(RTLD_NEXT, "accept");
    int client = real_accept(fd, addr, len);
    no_delay(client);
    return client;
}

int accept4(int fd, struct sockaddr *addr, socklen_t *len, int flags) {
    static int (*real_accept4)(int, struct sockaddr *, socklen_t *, int);
    if (!real_accept4) real_accept4 = dlsym(RTLD_NEXT, "accept4");
    int client = real_accept4(fd, addr, len, flags);
    no_delay(client);
    return client;
}
