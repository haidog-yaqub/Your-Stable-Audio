from codec import EncodecModel
from codec.utils import convert_audio
import soundfile as sf
import torchaudio
import torch


if __name__ == '__main__':
    # Instantiate a pretrained EnCodec model
    model = EncodecModel.encodec_model_24khz()
    # The number of codebooks used will be determined bythe bandwidth selected.
    # E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
    # Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
    # For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
    # of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
    model.set_target_bandwidth(24.0)

    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load("test_24k.wav")
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = model.encode(wav)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
        emb = model.encoder(wav)
        # wav_r = model.decode(encoded_frames)
        wav_r = model.decoder(emb)
        sf.write('test_24kr.wav', wav_r.squeeze().cpu().numpy().T, samplerate=sr)