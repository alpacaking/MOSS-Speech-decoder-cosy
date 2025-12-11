from typing import Literal
import gradio as gr
from fastrtc import WebRTC, StreamHandler, AdditionalOutputs
from numpy import ndarray
import sphn
import websockets.sync.client
import numpy as np


rtc_configuration = None

class MoshiHandler(StreamHandler):

    def __init__(self,
                 url: str,
                 expected_layout: Literal['mono', 'stereo'] = "mono",
                 output_sample_rate: int = 24000,
                 output_frame_size: int = 480) -> None:
        self.url = url
        proto, without_proto = self.url.split('://', 1)
        if proto in ['ws', 'http']:
            proto = "ws"
        elif proto in ['wss', 'https']:
            proto = "wss"
        
        self._generator = None
        self.output_chunk_size = 1920
        self.ws = None
        self.ws_url = f"{proto}://{without_proto}/api/chat"
        self.stream_reader = sphn.OpusStreamReader(output_sample_rate)
        self.stream_writer = sphn.OpusStreamWriter(output_sample_rate)
        self.all_output_data = None
        super().__init__(expected_layout, output_sample_rate, output_frame_size,
                         input_sample_rate=24000)
    
    def receive(self, frame: tuple[int, ndarray]) -> None:
        if not self.ws:
            self.ws = websockets.sync.client.connect(self.ws_url)
        _, array = frame
        array = array.squeeze().astype(np.float32) / 32768.0
        self.stream_writer.append_pcm(array)
        bytes = b"\x01" + self.stream_writer.read_bytes()
        self.ws.send(bytes)

    def generator(self):
        for message in self.ws:
            if len(message) == 0:
                yield None
            kind = message[0]
            if kind == 1:
                payload = message[1:]
                self.stream_reader.append_bytes(payload)
                pcm = self.stream_reader.read_pcm()
                if self.all_output_data is None:
                    self.all_output_data = pcm
                else:
                    self.all_output_data = np.concatenate((self.all_output_data, pcm))
                while self.all_output_data.shape[-1] >= self.output_chunk_size:
                    yield (self.output_sample_rate, self.all_output_data[: self.output_chunk_size].reshape(1, -1))
                    self.all_output_data = np.array(self.all_output_data[self.output_chunk_size :])
            elif kind == 2:
                payload = message[1:]
                yield AdditionalOutputs(payload.decode())

    
    def emit(self) -> tuple[int, ndarray] | None:
        if not self.ws:
            return
        if not self._generator:
            self._generator = self.generator()
        try:
            return next(self._generator)
        except StopIteration:
            self.reset()
    
    def reset(self) -> None:
        self._generator = None
        self.all_output_data = None
    
    def copy(self) -> StreamHandler:
        return MoshiHandler(self.url,
                            self.expected_layout,
                            self.output_sample_rate, self.output_frame_size)

    def shutdown(self) -> None:
        if self.ws:
            self.ws.close()



with gr.Blocks() as demo:
        gr.HTML(
        """
        <div style='text-align: center'>
            <h1>
                WebRTC + WebSocket
            </h1>
            <p>
                Each conversation is limited to 90 seconds. Once the time limit is up you can rejoin the conversation.
            </p>
        </div>
        """
        )
        response = gr.State(value="")
        chatbot = gr.Chatbot(type="messages", value=[])
        webrtc = WebRTC(label="Conversation", modality="audio", mode="send-receive", rtc_configuration=rtc_configuration)
        webrtc.stream(MoshiHandler("http://127.0.0.1:8023"),
                      inputs=[webrtc, chatbot], outputs=[webrtc], time_limit=300)
        
        def add_text(chat_history, response):
            if len(chat_history) == 0:
                chat_history.append({"role": "assistant", "content": ""})
            chat_history[-1]["content"] += response
            return chat_history
        
        webrtc.on_additional_outputs(add_text,
                                     inputs=[chatbot], outputs=chatbot,
                                     queue=False,
                                     show_progress="hidden")

demo.launch(server_port=7866, server_name='0.0.0.0',root_path='https://notebook-inspire.sii.edu.cn/')