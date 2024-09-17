# moshi-client

Frontend for the demo.

## Run the client

- Node is required, I recommend using [NVM](https://github.com/nvm-sh/nvm) to help you manage your node version and make sure you're on the recommended version for this project. If you do so run `nvm use`.
- Generate a public/private key pair, `cert.pem` and `key.pem` in this folder. See instructions after.
- Create an env.local file and add your an entry for `VITE_QUEUE_API_PATH` (default should be `/api`)
- Before running the project for the time or after dependencies update use `npm install`
- To run the project use `npm run dev`
- To build the project use `npm run build`

## Generate a key

As mentioned, you need a `cert.pem` and `key.pem`, you can generate them as
```bash
openssl genrsa -out key.pem 2048
openssl req -new -key key.pem -out cert.csr
openssl x509 -req -days 365 -in cert.csr -signkey key.pem -out cert.pem
```
Just key pressing enter.
