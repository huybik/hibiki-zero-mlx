## Hibiki-Zero frontend

First, [install `pnpm`](https://pnpm.io/installation) if you don't have it:
```bash
curl -fsSL https://get.pnpm.io/install.sh | sh -
```

If you don't have `node` either, get it using:

```bash
pnpm env use --global lts
```

Install: `pnpm install`

To run in dev mode: `pnpm dev`.
This will start a development server that will auto-reload if you change the source files. 

To run in production mode: First build using `pnpm build`, then start the server with `pnpm start`.

For either of these, you can specify the host using `--host` and the port using `--port`.