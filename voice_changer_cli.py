import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from encoder import encoder_handler as encoder
from encoder.model_train_params import model_embedding_size as speaker_embedding_size
from text_synthesizer.synthesizer_handler import Synthesizer
from utils.argutils import print_args
from utils.default_models import check_base_models
from vocoder import vocoder_handler as vocoder

import speech_recognition as sr
import whisper
from datetime import datetime, timedelta
from queue import Queue
from time import sleep

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="saved_models/default/encoder.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_fpath", type=Path,
                        default="saved_models/default/synthesizer.pt",
                        help="Path to a saved synthesizer")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="saved_models/default/vocoder.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, processing is done on CPU, even when a GPU is available.")
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, audio won't be played.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    args = parser.parse_args()
    arg_dict = vars(args)
    print_args(args, parser)

    # Hide GPUs from Pytorch to force CPU processing
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Running a test of your configuration...\n")

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n" %
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))
    # elif torch.backends.mps.is_available():
    #     device_id = torch.device("mps")
    #     print(f"Using GPU {device_id}")
    else:
        print("Using CPU for inference.\n")

    
    ########################################################################################
    ################################ LOAD & TEST THE MODELS ################################
    ########################################################################################
    print("Preparing the encoder, the synthesizer and the vocoder...")
    check_base_models(Path("saved_models"))

    ################################### TESTING ENCODER ####################################
    # Testing the encoder on a false wav form of 1 second, with sample rate of 16000.
    print("Testing your configuration with small inputs.")
    print("\tTesting the encoder...")
    encoder.load_model(args.enc_model_fpath)
    encoder.embed_utterance(np.zeros(encoder.sampling_rate))
    print("\tencoder validation passed...")

    ################################# TESTING TEXT-SYNTHE ##################################
    # Creating a dummy l2 normalized embedding to test the text_synthesizer
    embed = np.random.rand(speaker_embedding_size)
    embed = embed / np.linalg.norm(embed)
    embeds = [embed, np.zeros(speaker_embedding_size)]
    texts = ["test 1", "test 2"]
    
    print("\tTesting the text_synthesizer... (loading the model will output a lot of text)")
    synthesizer = Synthesizer(args.syn_model_fpath)
    mels = synthesizer.synthesize_spectrograms(texts, embeds)
    print("\text_synthesizer validation passed...")

    #################################### TESTING VOCODER ####################################
    mel = np.concatenate(mels, axis=1)
    no_action = lambda *args: None # Hiding callback function to display the generation
    print("\tTesting the vocoder...")
    vocoder.load_model(args.voc_model_fpath)
    vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)
    print("\tvocoder validation passed...")

    print("All test passed! You can now synthesize speech.\n\n")


    ## Interactive speech generation
    print("This is a GUI-less example of interface to SV2TTS. The purpose of this script is to "
          "show how you can interface this project easily with your own. See the source code for "
          "an explanation of what is happening.\n")

    num_generated = 0


    ################################## INITIALIZING WHISPER #############################
    passer_whisper = argparse.ArgumentParser()
    passer_whisper.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    passer_whisper.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    passer_whisper.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    passer_whisper.add_argument("--record_timeout", default=1,
                        help="How real time the recording is in seconds.", type=float)
    passer_whisper.add_argument("--phrase_timeout", default=2,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    args_whisper = passer_whisper.parse_args()

    phrase_time = None
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args_whisper.energy_threshold
    recorder.dynamic_energy_threshold = False

    source = sr.Microphone(sample_rate=16000)

    model = args_whisper.model
    if args_whisper.model != "large" and not args_whisper.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args_whisper.record_timeout
    phrase_timeout = args_whisper.phrase_timeout

    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("Model loaded.\n")
    
    
    ################################## Computing the embedding #############################
    # Get the reference audio filepath
    message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, " \
                "wav, m4a, flac, ...):\n"
    in_fpath = Path(input(message).replace("\"", "").replace("\'", ""))

    preprocessed_wav = encoder.preprocess_aud(in_fpath)
    original_wav, sampling_rate = librosa.load(str(in_fpath))
    preprocessed_wav = encoder.preprocess_aud(original_wav, sampling_rate)
    print("Loaded file succesfully")

    embed = encoder.embed_utterance(preprocessed_wav)
    print("Created the embedding")
    
    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # Clear the current working audio buffer if time elapsed and start with new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now
                
                # Combine audio data from queue
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()

                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Read the transcription.
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                # If detected a pause between recordings, add a new item to our transcription.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                # Clear the console to reprint the updated transcription.
                os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription:
                    print(line)
                # Flush stdout.
                print('', end='', flush=True)
                transcription = ['']
                
                sleep(0.1)
            
                ## Generating the spectrogram
                text = line

                # If seed is specified, reset torch seed and force synthesizer reload
                if args.seed is not None:
                    torch.manual_seed(args.seed)
                    synthesizer = Synthesizer(args.syn_model_fpath)

                # The synthesizer works in batch, so you need to put your data in a list or numpy array
                texts = [text]
                embeds = [embed]
                specs = synthesizer.synthesize_spectrograms(texts, embeds)
                spec = specs[0]
                print("Created the mel spectrogram")


                ## Generating the waveform
                print("Synthesizing the waveform:")

                # If seed is specified, reset torch seed and reload vocoder
                if args.seed is not None:
                    torch.manual_seed(args.seed)
                    vocoder.load_model(args.voc_model_fpath)

                generated_wav = vocoder.infer_waveform(spec)


                ## Post-generation
                generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

                # Trim excess silences to compensate for gaps in spectrograms (issue #53)
                generated_wav = encoder.preprocess_aud(generated_wav)

                # Play the audio (non-blocking)
                if not args.no_sound:
                    import sounddevice as sd
                    try:
                        sd.stop()
                        sd.play(generated_wav, synthesizer.sample_rate)
                    except sd.PortAudioError as e:
                        print("\nCaught exception: %s" % repr(e))
                        print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
                    except:
                        raise

        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
