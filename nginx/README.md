# Load balancer configuration

Two ways to run the same thing.

## Vanilla nginx

`whisper-pool.conf` is a complete server block. Copy it to
`/etc/nginx/conf.d/`, replace `whisper.example.com` and the certificate paths,
then `nginx -t && systemctl reload nginx`.

## Nginx Proxy Manager

NPM is a GUI over nginx, so the same directives apply, split across two places.

**1. The upstream block** goes in `/data/nginx/custom/http_top.conf` inside the
NPM container. It is included at the `http` level, which is where an `upstream`
has to live. Use `npm-http-top.conf` from this directory.

**2. The location block** goes in the proxy host's Advanced tab. Use
`npm-advanced.conf`. Set the proxy host's forward hostname and port to any one
of the replicas: NPM requires values there, but `proxy_pass` in the Advanced tab
overrides them.

Restart the NPM container after editing `http_top.conf`. A reload is not always
enough: nginx keeps old workers alive until their connections drain, and a
worker holding a long-lived upload connection can serve the previous config for
hours. If some requests behave as though the old upstream is still in place,
check for workers stuck in `shutting down`:

    docker top <npm-container> -eo pid,etimes,args | grep nginx
