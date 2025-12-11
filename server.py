import torch
import numpy as np
import sphn
import asyncio
import aiohttp
import time
from aiohttp import web

class ServerState:
    lock: asyncio.Lock

    def __init__(self, device: str | torch.device, sample_rate: int):
        self.device = device
        self.frame_size = 1920
        self.lock = asyncio.Lock()
        self.sample_rate = sample_rate

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        print("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        print("error", f"unexpected message type {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        print("error", f"unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        print("warning", "empty message")
                        continue
                    kind = message[0]
                    if kind == 1:  # audio
                        payload = message[1:]
                        opus_reader.append_bytes(payload)
                    else:
                        print("warning", f"unknown message kind {kind}")
            finally:
                close = True
                print("info", "connection closed")

        async def opus_loop():
            all_pcm_data = None

            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                while all_pcm_data.shape[-1] >= self.frame_size:
                    be = time.time()
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size:]
                    # chunk = torch.from_numpy(chunk)
                    # chunk = chunk.to(device=self.device)[None, None] # [1, 1, 2880]

                    # tokens = self.inference.generator.streaming_inference_tokenize(chunk).squeeze(1) # [3, 3] [C, group_size]
                    # chunk = self.inference.generator.streaming_inference_detokenize(tokens[1:].unsqueeze(1)).cpu().numpy().reshape(-1)
                    
                    opus_writer.append_pcm(chunk) # Echo back

                    # If there is text
                    # msg = b"\x02" + bytes(_text, encoding="utf8")
                    # await ws.send_bytes(msg)

                    print("info", f"frame handled in {1000 * (time.time() - be):.1f}ms")

        async def send_loop():
            while True:
                if close:
                    # print(f"<{"><".join([str(i) for i in self.user_tokens.T.reshape(-1).tolist()])}>")
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)

        close = False
        async with self.lock:
            opus_writer = sphn.OpusStreamWriter(self.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.sample_rate)
            # Send the handshake.
            await ws.send_bytes(b"\x00")
            await asyncio.gather(opus_loop(), recv_loop(), send_loop())
        return ws


if __name__ == "__main__":
    with torch.inference_mode():
        state = ServerState(device="cuda", sample_rate=24000)
        app = web.Application()
        app.router.add_get("/api/chat", state.handle_chat)
        web.run_app(app, port=8023)