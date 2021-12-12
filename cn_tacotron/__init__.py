from dataclasses import dataclass


@dataclass
class Config:
    preprocess_type = 'BasicProcessor'
    acoustic_type = 'Mel'  # acoustic feature type: Mel, Mellpc, Bark, etc.
    label_dir = '/cn_tacotron/train_data/labels'
    wav_dir = '/Users/tristan/gitproject/tacotron_pytorch/cn_tacotron/demo/wavs'
    out_feature_dir = '/Users/tristan/gitproject/tacotron_pytorch/cn_tacotron/train_data'
    n_jobs = 2
    # synthesizer_type = 'TacotronSynthesizer'
    # supported_preprocess_type = ["BasicProcessor", "AcousticProcessor", "CustomProcessor"]
    # supported_acoustic_type = ["Mel"]
    # batch_size = 48
    # start_decay = 25000
    # decay_steps = 25000
    # decay_rate = 0.5
    # start_lr = 2e-4
    # end_lr: 2e-6
    # adam_beta1: 0.9
    # adam_beta2: 0.999
    # adam_epsilon: 1e-6
    # cross_entropy_pos_weight: 20
    # gradclip_value: 1.0
    # total_training_steps: 400000
    # save_summary_steps: 1000
    # save_checkpoints_steps: 5000
    # keep_checkpoint_max: 3
    # stop_threshold: 0.5
    # random_seed: 20200813
    min_level_db = -115
    ref_level_db = 20
    fmin = 0
    fmax = 8000
    sample_rate = 16000
    num_silent_frames = 4
    max_acoustic_length = 2000
    max_abs_value = 4.
    trim_fft_size = 512
    trim_hop_size = 128
    trim_top_db = 30
    min_db = -115
    outputs_per_step = 5
    acoustic_dim = 80
    num_freq = 1025
    fft_size = 2048
    n_fft = 2048
    hop_size = 200
    win_size = 800
    preemphasis = 0.97
    frame_shift_ms = 12.5
    frame_length_ms = 50
    num_mels = 80
    padding_idx = None
    use_memory_mask = False

    initial_learning_rate = 0.002

    adam_beta1 = 0.9
    adam_beta2 = 0.999
    weight_decay = 0.0


config = Config()
