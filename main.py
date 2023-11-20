import gradio as gr
from typing import Tuple, List, Final
import argparse

import setting
import transcribe
import state_class



with gr.Blocks() as demo:
    state = gr.State(value=state_class.StateClass())
    with gr.Tab("main"):
        with gr.Row():
            with gr.Column():
                mic = gr.Audio(sources="microphone")
                speaking = gr.Checkbox(label="speaking")
                reset_button = gr.Button("reset")
            output = gr.Textbox(value="", label="output")
    with gr.Tab("settings"):
        threshold = gr.Slider(minimum=0.01, maximum=1.0,
                              value=setting.val_threshold, label="threshold")
        volume = gr.Slider(minimum=0.01, maximum=1.0,
                           value=setting.val_volume, label="volume")
        language = gr.Dropdown(
            choices=["None", "ja"], label="language", value=setting.val_language, filterable=False)
        save_button = gr.Button("save")

    reset_button.click(
        fn=state_class.reset,
        inputs=[state],
        outputs=[state, output]
    )

    mic.stream(
        fn=transcribe.transcribe,
        inputs=[state, mic, volume, threshold, language],
        outputs=[state, output, speaking],
        show_progress=False
    )

    mic.stop_recording(
        cancels=[transcribe.transcribe]
    )

    save_button.click(
        setting.save,
        inputs=[threshold, volume, language]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", action="store_true")
    parser.add_argument("--cert", type=str)
    parser.add_argument("--key", type=str)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()
    if args.listen:
        if (args.cert == None) or (args.key == None):
            print("cert:{}, key:{}".format(args.cert, args.key))
            exit()
        demo.launch(server_name="0.0.0.0",
                    ssl_certfile=args.cert, ssl_keyfile=args.key,
                    ssl_verify=False, share=args.share, server_port=args.port)
    else:
        demo.launch(share=args.share, server_port=args.port)
