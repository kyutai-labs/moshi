# Protocol

The connection takes place using a websocket. This handles the message lengths
for us. The binary protocol for messages is as follows. The protocol uses little
endian encoding.

Each message starts by a single byte indicating the message type `MT`.
The format for the rest of the message, aka the payload, depends on `MT`.

```
- Handshake MT=0. The payload is made of two fields.
    1. Protocol version (`u32`) - always 0 for now.
    2. Model version (`u32`).
- Audio MT=1. The payload is made of a single field.
  - Binary data for the ogg frames containing opus encoded audio (24kHz, mono).
- Text MT=2. The payload is made of a single field.
  - UTF8 encoded string.
- Control MT=3. The payload is made of a single field. This is not used in full
  streaming mode.
  - One byte B describing the control itself.
    - Start B=0.
    - EndTurn B=1.
    - Pause B=2.
    - Restart B=3.
- MetaData MT=4. The payload is made of a single field.
  - UTF8 encoded string with json data.
- Error MT=5. The payload is made of a single field.
  - UTF8 encoded string containing the error description.
- Ping MT=6. No payload, this message type is currently unused.
```
Messages with an unknow message types should be discarded.
