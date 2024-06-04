print("Starting up. Please be patient...")

import argparse
import datetime
import os
import sys
from typing import Optional
import json
import utils

import gradio as gr
import torch
import yaml

from common.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from common.log import logger
from common.tts_model import ModelHolder
from infer import InvalidToneError
from text.japanese import g2kata_tone, kata_tone2phone_tone, text_normalize

is_hf_spaces = os.getenv("SYSTEM") == "spaces"
limit = 150

# Get path settings
with open(os.path.join("configs", "paths.yml"), "r", encoding="utf-8") as f:
    path_config: dict[str, str] = yaml.safe_load(f.read())
    # dataset_root = path_config["dataset_root"]
    assets_root = path_config["assets_root"]

def tts_fn(
    model_name,
    model_path,
    text,
    language,
    reference_audio_path,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    line_split,
    split_interval,
    assist_text,
    assist_text_weight,
    use_assist_text,
    style,
    style_weight,
    kata_tone_json_str,
    use_tone,
    speaker,
):
    print(f"[!] model_name: {model_name} ({type(model_name)})")
    print(f"[!] model_path: {model_path} ({type(model_path)})")
    print(f"[!] text: {text} ({type(text)})")
    print(f"[!] language: {language} ({type(language)})")
    print(f"[!] reference_audio_path: {reference_audio_path} ({type(reference_audio_path)})")
    print(f"[!] sdp_ratio: {sdp_ratio} ({type(sdp_ratio)})")
    print(f"[!] noise_scale: {noise_scale} ({type(noise_scale)})")
    print(f"[!] noise_scale_w: {noise_scale_w} ({type(noise_scale_w)})")
    print(f"[!] length_scale: {length_scale} ({type(length_scale)})")
    print(f"[!] line_split: {line_split} ({type(line_split)})")
    print(f"[!] split_interval: {split_interval} ({type(split_interval)})")
    print(f"[!] assist_text: {assist_text} ({type(assist_text)})")
    print(f"[!] assist_text_weight: {assist_text_weight} ({type(assist_text_weight)})")
    print(f"[!] use_assist_text: {use_assist_text} ({type(use_assist_text)})")
    print(f"[!] style: {style} ({type(style)})")
    print(f"[!] style_weight: {style_weight} ({type(style_weight)})")
    print(f"[!] kata_tone_json_str: {kata_tone_json_str} ({type(kata_tone_json_str)})")
    print(f"[!] use_tone: {use_tone} ({type(use_tone)})")
    print(f"[!] speaker: {speaker} ({type(speaker)})")
    
    if len(text)<2:
        return "Please enter some text.", None, kata_tone_json_str
    
    if is_hf_spaces and len(text) > limit:
        return f"Too long! There is a character limit of {limit} characters.", None, kata_tone_json_str

    if(not model_holder.current_model):
        model_holder.load_model_gr(model_name, model_path)
        logger.info(f"Loaded model '{model_name}'")
    if(model_holder.current_model.model_path != model_path):
        model_holder.load_model_gr(model_name, model_path)
        logger.info(f"Swapped to model '{model_name}'")
    speaker_id = model_holder.current_model.spk2id[speaker]
    start_time = datetime.datetime.now()

    wrong_tone_message = ""
    kata_tone: Optional[list[tuple[str, int]]] = None
    if use_tone and kata_tone_json_str != "":
        if language != "JP":
            #logger.warning("Only Japanese is supported for tone generation.")
            wrong_tone_message = "アクセント指定は現在日本語のみ対応しています。"
        if line_split:
            #logger.warning("Tone generation is not supported for line split.")
            wrong_tone_message = (
                "アクセント指定は改行で分けて生成を使わない場合のみ対応しています。"
            )
        try:
            kata_tone = []
            json_data = json.loads(kata_tone_json_str)
            # tupleを使うように変換
            for kana, tone in json_data:
                assert isinstance(kana, str) and tone in (0, 1), f"{kana}, {tone}"
                kata_tone.append((kana, tone))
        except Exception as e:
            logger.warning(f"Error occurred when parsing kana_tone_json: {e}")
            wrong_tone_message = f"アクセント指定が不正です: {e}"
            kata_tone = None

    # toneは実際に音声合成に代入される際のみnot Noneになる
    tone: Optional[list[int]] = None
    if kata_tone is not None:
        phone_tone = kata_tone2phone_tone(kata_tone)
        tone = [t for _, t in phone_tone]
    
    try:
        sr, audio = model_holder.current_model.infer(
            text=text,
            language=language,
            reference_audio_path=reference_audio_path,
            sdp_ratio=sdp_ratio,
            noise=noise_scale,
            noisew=noise_scale_w,
            length=length_scale,
            line_split=line_split,
            split_interval=split_interval,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            use_assist_text=use_assist_text,
            style=style,
            style_weight=style_weight,
            given_tone=tone,
            sid=speaker_id,
        )
    except InvalidToneError as e:
        logger.error(f"Tone error: {e}")
        return f"Error: アクセント指定が不正です:\n{e}", None, kata_tone_json_str
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return f"Error: {e}", None, kata_tone_json_str

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()

    if tone is None and language == "JP":
        # アクセント指定に使えるようにアクセント情報を返す
        norm_text = text_normalize(text)
        kata_tone = g2kata_tone(norm_text)
        kata_tone_json_str = json.dumps(kata_tone, ensure_ascii=False)
    elif tone is None:
        kata_tone_json_str = ""

    if reference_audio_path:
        style="External Audio"
    logger.info(f"Successful inference, took {duration}s | {speaker} | {language}/{sdp_ratio}/{noise_scale}/{noise_scale_w}/{length_scale}/{style}/{style_weight} | {text}")
    message = f"Success, time: {duration} seconds."
    if wrong_tone_message != "":
        message = wrong_tone_message + "\n" + message
    return message, (sr, audio), kata_tone_json_str

def load_voicedata():
    print("Loading voice data...")
    #voices = []
    envoices = []
    jpvoices = []
    styledict = {}
    with open("voicelist.json", "r", encoding="utf-8") as f:
        voc_info = json.load(f)
    for name, info in voc_info.items():
        if not info['enable']:
            continue
        model_path = info['model_path']
        voice_name = info['title']
        speakerid = info['speakerid']
        datasetauthor = info['datasetauthor']
        image = info['cover']
        if not model_path in styledict.keys():
           conf=f"model_assets/{model_path}/config.json"
           hps = utils.get_hparams_from_file(conf)
           s2id = hps.data.style2id
           styledict[model_path] = s2id.keys()
        print(f"Indexed voice {voice_name}")
        if(info['primarylang']=="JP"):
            jpvoices.append((name, model_path, voice_name, speakerid, datasetauthor, image))
        else:
            envoices.append((name, model_path, voice_name, speakerid, datasetauthor, image))
    return [envoices, jpvoices], styledict
        

initial_text = "Hello there! This is test audio of a new Hololive text to speech tool."

initial_md = """
# Hololive [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)
### Space by [Kit Lemonfoot](https://huggingface.co/Kit-Lemonfoot)/[Noel Shirogane's High Flying Birds](https://www.youtube.com/channel/UCG9A0OJsJTluLOXfMZjJ9xA)
### Based on code originally by [fishaudio](https://github.com/fishaudio) and [litagin02](https://github.com/litagin02)

Do no evil.

**Note:** Most of the models are a *work in progress.* They may not sound fully correct.
"""

style_md = """
- You can control things like voice tone, emotion, and reading style through presets or through voice files.
- Neutral acts as an average across all speakers. Styling options act as an override to Neutral.
- Setting the intensity too high will likely break the output.
- The required intensity will depend based on the speaker and the desired style.
- If you're using preexisting audio data to style the output, try to use a voice that is similar to the desired speaker.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--dir", "-d", type=str, help="Model directory", default=assets_root
    )
    parser.add_argument(
        "--share", action="store_true", help="Share this app publicly", default=False
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default=None,
        help="Server name for Gradio app",
    )
    parser.add_argument(
        "--no-autolaunch",
        action="store_true",
        default=False,
        help="Do not launch app automatically",
    )
    args = parser.parse_args()
    model_dir = args.dir

    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_holder = ModelHolder(model_dir, device)

    languages = ["EN", "JP", "ZH"]
    langnames = ["English", "Japanese"]

    model_names = model_holder.model_names
    if len(model_names) == 0:
        logger.error(f"No models found. Please place the model in {model_dir}.")
        sys.exit(1)
    initial_id = 0
    initial_pth_files = model_holder.model_files_dict[model_names[initial_id]]
    #print(initial_pth_files)

    voicedata, styledict = load_voicedata()

    #Gradio preload
    text_input = gr.TextArea(label="Text", value=initial_text)
    line_split = gr.Checkbox(label="Divide text seperately by line breaks", value=True)
    split_interval = gr.Slider(
        minimum=0.0,
        maximum=2,
        value=0.5,
        step=0.1,
        label="Length of division seperation time (in seconds)",
    )
    language = gr.Dropdown(choices=languages, value="EN", label="Language")
    sdp_ratio = gr.Slider(
        minimum=0, maximum=1, value=0.2, step=0.1, label="SDP Ratio"
    )
    noise_scale = gr.Slider(
        minimum=0.1, maximum=2, value=0.6, step=0.1, label="Noise"
    )
    noise_scale_w = gr.Slider(
        minimum=0.1, maximum=2, value=0.8, step=0.1, label="Noise_W"
    )
    length_scale = gr.Slider(
        minimum=0.1, maximum=2, value=1.0, step=0.1, label="Length"
    )
    use_style_text = gr.Checkbox(label="Use stylization text", value=False)
    style_text = gr.Textbox(
        label="Style text",
        placeholder="Check the \"Use stylization text\" box to use this option!",
        info="The voice will be similar in tone and emotion to the text, however inflection and tempo may be worse as a result.",
        visible=True,
    )
    style_text_weight = gr.Slider(
        minimum=0,
        maximum=1,
        value=0.7,
        step=0.1,
        label="Text stylization strength",
        visible=True,
    )

    with gr.Blocks(theme=gr.themes.Base(primary_hue="emerald", secondary_hue="green"), title="Hololive Style-Bert-VITS2") as app:
        gr.Markdown(initial_md)

        #NOT USED SINCE NONE OF MY MODELS ARE JPEXTRA.
        #ONLY HERE FOR COMPATIBILITY WITH THE EXISTING INFER CODE.
        #DO NOT RENDER OR MAKE VISIBLE
        tone = gr.Textbox(
            label="Accent adjustment (0 for low, 1 for high)",
            info="This can only be used when not seperated by line breaks. It is not universal.",
            visible=False
        )
        use_tone = gr.Checkbox(label="Use accent adjustment", value=False, visible=False)

        #for (name, model_path, voice_name, speakerid, datasetauthor, image) in voicedata:
        for vi in range(len(voicedata)):
            with gr.TabItem(langnames[vi]):
                for (name, model_path, voice_name, speakerid, datasetauthor, image) in voicedata[vi]:
                    with gr.TabItem(name):
                        mn = gr.Textbox(value=model_path, visible=False, interactive=False)
                        mp = gr.Textbox(value=f"model_assets/{model_path}/{model_path}.safetensors", visible=False, interactive=False)
                        spk = gr.Textbox(value=speakerid, visible=False, interactive=False)
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown(f"**{voice_name}**\n\nModel name: {model_path} | Dataset author: {datasetauthor}")
                                gr.Image(f"images/{image}", label=None, show_label=False, width=300, show_download_button=False, container=False, show_share_button=False)
                            with gr.Column():
                                with gr.TabItem("Style using a preset"):
                                    style = gr.Dropdown(
                                        label="Current style (Neutral is an average style)",
                                        choices=styledict[model_path],
                                        value="Neutral",
                                    )
                                with gr.TabItem("Style using existing audio"):
                                    ref_audio_path = gr.Audio(label="Reference Audio", type="filepath")
                                style_weight = gr.Slider(
                                    minimum=0,
                                    maximum=50,
                                    value=5,
                                    step=0.1,
                                    label="Style strength",
                                )
                            with gr.Column():
                                tts_button = gr.Button(
                                    "Synthesize", variant="primary", interactive=True
                                )
                                text_output = gr.Textbox(label="Info")
                                audio_output = gr.Audio(label="Result")

                                tts_button.click(
                                    tts_fn,
                                    inputs=[
                                        mn,
                                        mp,
                                        text_input,
                                        language,
                                        ref_audio_path,
                                        sdp_ratio,
                                        noise_scale,
                                        noise_scale_w,
                                        length_scale,
                                        line_split,
                                        split_interval,
                                        style_text,
                                        style_text_weight,
                                        use_style_text,
                                        style,
                                        style_weight,
                                        tone,
                                        use_tone,
                                        spk,
                                    ],
                                    outputs=[text_output, audio_output, tone],
                                )

        with gr.Row():
            with gr.Column():
                text_input.render()
                line_split.render()
                split_interval.render()
                language.render()
            with gr.Column():
                sdp_ratio.render()
                noise_scale.render()
                noise_scale_w.render()
                length_scale.render()
                use_style_text.render()
                style_text.render()
                style_text_weight.render()

        with gr.Accordion("Styling Guide", open=False):
            gr.Markdown(style_md)

    app.launch(allowed_paths=['/file/images/'], share=True)
