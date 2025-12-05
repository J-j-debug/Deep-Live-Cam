import os
import gradio as gr
import modules.globals
import modules.core
from modules.utilities import is_image, is_video

def init(start_callback, destroy_callback):
    def process_media(
        source_image,
        target_media,
        keep_fps,
        keep_audio,
        keep_frames,
        many_faces,
        face_enhancer,
        video_encoder,
        video_quality
    ):
        # Update globals
        modules.globals.source_path = source_image
        modules.globals.target_path = target_media
        modules.globals.keep_fps = keep_fps
        modules.globals.keep_audio = keep_audio
        modules.globals.keep_frames = keep_frames
        modules.globals.many_faces = many_faces
        modules.globals.video_encoder = video_encoder
        modules.globals.video_quality = video_quality

        # Handle Face Enhancer
        modules.globals.frame_processors = ['face_swapper']
        if face_enhancer:
            modules.globals.frame_processors.append('face_enhancer')
            modules.globals.fp_ui['face_enhancer'] = True
        else:
            modules.globals.fp_ui['face_enhancer'] = False

        # Prepare output path
        if is_image(target_media):
            output_filename = "output.png"
            modules.globals.output_path = os.path.join(os.path.dirname(target_media), output_filename)
        elif is_video(target_media):
            output_filename = "output.mp4"
            modules.globals.output_path = os.path.join(os.path.dirname(target_media), output_filename)
        else:
            return None

        # Run processing
        # We need to capture the output or ensure the UI updates.
        # Since start() prints to stdout/stderr, we will rely on the return value being the file.

        # NOTE: modules.core.start() is blocking. Gradio handles threading.
        # However, modules.core.start() uses update_status which we bypassed in core.py.
        # We could potentially redirect stdout to capture progress, but for V1 let's just run it.

        try:
            start_callback()
            return modules.globals.output_path
        except Exception as e:
            print(f"Error processing: {e}")
            return None

    with gr.Blocks(title="Deep Live Cam Web UI") as interface:
        gr.Markdown("# Deep Live Cam Web UI")

        with gr.Row():
            with gr.Column():
                source_image = gr.Image(type="filepath", label="Source Face", value=modules.globals.source_path)
                target_media = gr.File(label="Target Image or Video", value=modules.globals.target_path)

                with gr.Accordion("Settings", open=True):
                    keep_fps = gr.Checkbox(label="Keep FPS", value=modules.globals.keep_fps)
                    keep_audio = gr.Checkbox(label="Keep Audio", value=modules.globals.keep_audio)
                    keep_frames = gr.Checkbox(label="Keep Frames", value=modules.globals.keep_frames)
                    many_faces = gr.Checkbox(label="Many Faces", value=modules.globals.many_faces)
                    face_enhancer = gr.Checkbox(label="Face Enhancer", value=False)

                    video_encoder = gr.Dropdown(
                        label="Video Encoder",
                        choices=['libx264', 'libx265', 'libvpx-vp9'],
                        value=modules.globals.video_encoder or 'libx264'
                    )
                    video_quality = gr.Slider(
                        label="Video Quality (0-51)",
                        minimum=0,
                        maximum=51,
                        step=1,
                        value=modules.globals.video_quality or 18
                    )

                start_btn = gr.Button("Start Processing", variant="primary")

            with gr.Column():
                output_media = gr.File(label="Output")

        start_btn.click(
            fn=process_media,
            inputs=[
                source_image,
                target_media,
                keep_fps,
                keep_audio,
                keep_frames,
                many_faces,
                face_enhancer,
                video_encoder,
                video_quality
            ],
            outputs=[output_media]
        )

    auth = None
    if modules.globals.ui_username and modules.globals.ui_password:
        auth = (modules.globals.ui_username, modules.globals.ui_password)

    interface.launch(server_name="0.0.0.0", server_port=modules.globals.ui_port, auth=auth)
