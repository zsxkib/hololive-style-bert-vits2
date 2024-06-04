# Prediction interface for Cog ⚙️
# https://cog.run/python

import os

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from cog import BasePredictor, Path, Input
import sys
import json
import yaml
import time
import utils
import torch
import datetime
import subprocess
import soundfile as sf
from typing import Optional

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
    print(f"[!] model_name: {model_name}")
    print(f"[!] model_path: {model_path}")
    print(f"[!] text: {text}")
    print(f"[!] language: {language}")
    print(f"[!] reference_audio_path: {reference_audio_path}")
    print(f"[!] sdp_ratio: {sdp_ratio}")
    print(f"[!] noise_scale: {noise_scale}")
    print(f"[!] noise_scale_w: {noise_scale_w}")
    print(f"[!] length_scale: {length_scale}")
    print(f"[!] line_split: {line_split}")
    print(f"[!] split_interval: {split_interval}")
    print(f"[!] assist_text: {assist_text}")
    print(f"[!] assist_text_weight: {assist_text_weight}")
    print(f"[!] use_assist_text: {use_assist_text}")
    print(f"[!] style: {style}")
    print(f"[!] style_weight: {style_weight}")
    print(f"[!] kata_tone_json_str: {kata_tone_json_str}")
    print(f"[!] use_tone: {use_tone}")
    print(f"[!] speaker: {speaker}")
    if len(text) < 2:
        return "Please enter some text.", None, kata_tone_json_str

    if not model_holder.current_model:
        model_holder.load_model_gr(model_name, model_path)
        print(f"[!] Loaded model '{model_name}'")
    if model_holder.current_model.model_path != model_path:
        model_holder.load_model_gr(model_name, model_path)
        print(f"[!] Swapped to model '{model_name}'")
    speaker_id = model_holder.current_model.spk2id[speaker]
    start_time = datetime.datetime.now()

    wrong_tone_message = ""
    kata_tone: Optional[list[tuple[str, int]]] = None
    if use_tone and kata_tone_json_str != "":
        if language != "JP":
            # print("[!] Only Japanese is supported for tone generation.")
            wrong_tone_message = "アクセント指定は現在日本語のみ対応しています。"  # Accent specification is currently only supported in Japanese.
        if line_split:
            # print("[!] Tone generation is not supported for line split.")
            wrong_tone_message = "アクセント指定は改行で分けて生成を使わない場合のみ対応しています。"  # Accent specification is only supported when not using line split for generation.
        try:
            kata_tone = []
            json_data = json.loads(kata_tone_json_str)
            # tupleを使うように変換  # Convert to use tuple
            for kana, tone in json_data:
                assert isinstance(kana, str) and tone in (0, 1), f"{kana}, {tone}"
                kata_tone.append((kana, tone))
        except Exception as e:
            print(f"[!] Error occurred when parsing kana_tone_json: {e}")
            wrong_tone_message = (
                f"アクセント指定が不正です: {e}"  # Accent specification is invalid: {e}
            )
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
        print(f"[!] Tone error: {e}")
        return f"Error: アクセント指定が不正です:\n{e}", None, kata_tone_json_str
    except ValueError as e:
        print(f"[!] Value error: {e}")
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
        style = "External Audio"
    print(
        f"[!] Successful inference, took {duration}s | {speaker} | {language}/{sdp_ratio}/{noise_scale}/{noise_scale_w}/{length_scale}/{style}/{style_weight} | {text}"
    )
    message = f"Success, time: {duration} seconds."
    if wrong_tone_message != "":
        message = wrong_tone_message + "\n" + message
    return message, (sr, audio), kata_tone_json_str


def load_voicedata():
    print("[!] Loading voice data...")
    # voices = []
    envoices = []
    jpvoices = []
    styledict = {}
    with open("voicelist.json", "r", encoding="utf-8") as f:
        voc_info = json.load(f)
    for name, info in voc_info.items():
        if not info["enable"]:
            continue
        model_path = info["model_path"]
        voice_name = info["title"]
        speakerid = info["speakerid"]
        datasetauthor = info["datasetauthor"]
        image = info["cover"]
        if not model_path in styledict.keys():
            conf = f"model_assets/{model_path}/config.json"
            hps = utils.get_hparams_from_file(conf)
            s2id = hps.data.style2id
            styledict[model_path] = s2id.keys()
        print(f"[!] Indexed voice {voice_name}")
        if info["primarylang"] == "JP":
            jpvoices.append(
                (name, model_path, voice_name, speakerid, datasetauthor, image)
            )
        else:
            envoices.append(
                (name, model_path, voice_name, speakerid, datasetauthor, image)
            )
    return [envoices, jpvoices], styledict


def download_weights(url: str, dest: str) -> None:
    # NOTE WHEN YOU EXTRACT SPECIFY THE PARENT FOLDER
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        global model_holder, g2kata_tone, kata_tone2phone_tone, text_normalize, InvalidToneError, ModelHolder

        self.use_pget_and_download_weights()

        # Import modules after downloading weights
        from infer import InvalidToneError
        from common.tts_model import ModelHolder
        from text.japanese import g2kata_tone, kata_tone2phone_tone, text_normalize

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_holder = ModelHolder(assets_root, self.device)

        self.languages = ["EN", "JP", "ZH"]
        self.langnames = ["English", "Japanese"]

        model_names = model_holder.model_names
        if len(model_names) == 0:
            print(f"[!] No models found. Please place the model in {assets_root}.")
            sys.exit(1)
        initial_id = 0
        initial_pth_files = model_holder.model_files_dict[model_names[initial_id]]

        self.voicedata, self.styledict = load_voicedata()

        # Load the initial model
        model_holder.load_model_gr(model_names[initial_id], initial_pth_files[0])
        print(f"[!] Loaded initial model: {model_names[initial_id]}")

        # Verify that the model is loaded
        if model_holder.current_model is None:
            raise RuntimeError("Failed to load the initial model.")
        else:
            print(
                f"[!] Model loaded successfully: {model_holder.current_model.model_path}"
            )

    def use_pget_and_download_weights(self):
        MODEL_CACHE = "model_assets"
        BERT_CACHE = "bert"

        # Create directories if they don't exist
        os.makedirs(MODEL_CACHE, exist_ok=True)
        os.makedirs(BERT_CACHE, exist_ok=True)

        # Model files and base URLs
        model_files = [
            "SBV2_HoloAus.tar",
            "SBV2_HoloESL.tar",
            "SBV2_HoloHi.tar",
            "SBV2_HoloIDFlu.tar",
            "SBV2_HoloJPTest.tar",
            "SBV2_HoloJPTest2.5.tar",
            "SBV2_HoloJPTest2.tar",
            "SBV2_HoloLow.tar",
            "SBV2_KosekiBijou.tar",
            "SBV2_TakanashiKiara.tar",
        ]

        bert_model_files = [
            "deberta-v2-large-japanese-char-wwm.tar",
            "chinese-roberta-wwm-ext-large.tar",
            "deberta-v3-large.tar",
            "bert_models.json",
        ]

        base_url = f"https://weights.replicate.delivery/default/hololive-style-bert-vits2/{MODEL_CACHE}/"
        bert_base_url = f"https://weights.replicate.delivery/default/hololive-style-bert-vits2/{BERT_CACHE}/"

        # Download model files
        for model_file in model_files:
            url = base_url + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        # Download BERT model files
        for model_file in bert_model_files:
            url = bert_base_url + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(BERT_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

    def predict(
        self,
        speaker: str = Input(
            default="EN_MoriCalliope",
            choices=[
                "EN_MoriCalliope",
                "EN_TakanashiKiara",
                "EN_NinomaeInanis",
                "EN_GawrGura",
                "EN_AmeliaWatson",
                "EN_IRyS",
                "EN_TsukumoSana",
                "EN_CeresFauna",
                "EN_OuroKronii",
                "EN_NanashiMumei",
                "EN_HakosBaelz",
                "EN_ShioriNovella",
                "EN_KosekiBijou",
                "EN_NerissaRavencroft",
                "EN_AiraniIofifteen",
                "EN_KureijiOllie",
                "EN_AnyaMelfissa",
                "EN_VestiaZeta",
                "JP_TokinoSora",
                "JP_HoshimachiSuisei",
                "JP_AZKi",
                "JP_YozoraMel",
                "JP_NatsuiroMatsuri",
                "JP_AkiRosenthal",
                "JP_AkaiHaato",
                "JP_MinatoAqua",
                "JP_NakiriAyame",
                "JP_NekomataOkayu",
                "JP_ShiranuiFlare",
                "JP_ShiroganeNoel",
                "JP_HoushouMarine",
                "JP_TokoyamiTowa",
                "JP_YukihanaLamy",
                "JP_LaplusDarknesss",
                "JP_TakaneLui",
                "JP_HakuiKoyori",
                "JP_SakamataChloe",
                "JP_IchijouRirika",
            ],
            description="Default speaker",
        ),
        text_input: str = Input(
            default="Hello there! This is test audio of a new Hololive text to speech tool running on Replicate!",
            description="Text to convert to speech (text-to-voice)",
        ),
        reference_audio_path: Path = Input(
            default=None, description="Path to a reference audio file (voice-to-voice)"
        ),
        line_split: bool = Input(
            default=True,
            description="Whether to split the text into lines for processing",
        ),
        split_interval: float = Input(
            default=0.5, description="Interval between splits when line_split is True"
        ),
        style: str = Input(
            default="Neutral",
            choices=[
                "Neutral",
                "Normal",
                "Excited",
                "Sana",
                "Baelz1",
                "Baelz2",
                "BaelzShouting",
                "Anya",
                "IofiLoud",
                "Iofi",
                "ZetaSoft",
                "Zeta",
                "ZetaLoud",
                "Ollie",
                "Koyori",
                "Chloe",
                "Lamy",
                "Aqua",
                "Sora",
                "Towa",
                "Suisei",
                "Ayame",
                "Haato",
                "Matsuri",
                "Mel",
                "Aki",
                "Lui",
                "AZKi",
                "Flare",
                "Ririka",
                "Laplus",
                "Noel",
                "Okayu",
                "Marine",
                "Kronii",
                "NerissaLaugh",
                "Calli",
                "Nerissa",
                "Japanese",
                "Happy",
                "Reading",
                "Fauna",
                "Amelia",
                "MumeiLaugh",
                "Shiori",
                "IRyS",
                "Ina",
                "Gura",
                "Mumei",
                "ShioriLaugh",
                "Scared",
                "Angry",
            ],
            description="Style of speech to use (choices may be limited based on the selected speaker)",
        ),
        style_weight: float = Input(
            default=5.0, description="Weight of the style effect"
        ),
        use_tone: bool = Input(
            default=False,
            description="Whether to use tone information in the synthesis (Japanese only)",
        ),
        sdp_ratio: float = Input(
            default=0.2, description="Ratio for speaker-dependent processing"
        ),
        noise_scale: float = Input(
            default=0.6, description="Scale of noise to add to the synthesis"
        ),
        noise_scale_w: float = Input(
            default=0.8, description="Scale of noise for the waveform"
        ),
        length_scale: float = Input(
            default=1.0, description="Scale of the length of the synthesized speech"
        ),
        style_text_weight: float = Input(
            default=0.7, description="Weight of the style text effect"
        ),
        use_style_text: bool = Input(
            default=False,
            description="Whether to use additional style text in the synthesis",
        ),
        style_text: str = Input(
            default="",
            description="Additional text to guide the style of the synthesis",
        ),
    ) -> Path:
        # Infer language and remove the prefix from the speaker
        language, speaker = speaker.split("_", 1)

        # Find the speaker in voicedata
        for voice_group in self.voicedata:
            for voice in voice_group:
                if voice[3] == speaker:
                    (
                        name,
                        model_path,
                        voice_name,
                        speakerid,
                        datasetauthor,
                        image,
                    ) = voice
                    model_name = model_path  # Correctly infer model_name
                    model_path = f"model_assets/{model_path}/{model_path}.safetensors"

                    # Validate the selected style against the available styles for the model
                    available_styles = self.styledict[model_name]
                    if style not in available_styles:
                        raise ValueError(
                            f"The selected style '{style}' is not available for the speaker '{speaker}'. Available styles: {', '.join(available_styles)}"
                        )

                    break
            else:
                continue
            break
        else:
            raise ValueError(f"Speaker {speaker} not found in voicedata.")

        # Ensure the types of the parameters match the expected types in tts_fn
        text_output, (sr, audio), kata_toneas_json_str = tts_fn(
            model_name,
            model_path,
            text_input,
            language,
            reference_audio_path if reference_audio_path else None,
            float(sdp_ratio),
            float(noise_scale),
            float(noise_scale_w),
            float(length_scale),
            bool(line_split),
            float(split_interval),
            style_text,
            float(style_text_weight),
            bool(use_style_text),
            style,
            float(style_weight),
            "",  # kata_tone_json_str
            bool(use_tone),
            speakerid,
        )

        # Check if audio data is valid
        if audio is None or len(audio) == 0:
            raise ValueError("Invalid audio data received from tts_fn")

        # Save the audio output to a file using soundfile
        output_path = "output.wav"
        sf.write(output_path, audio, sr)

        return Path(output_path)
