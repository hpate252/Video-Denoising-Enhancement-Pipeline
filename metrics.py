from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_metrics(clean_frame, output_frame):
    psnr = peak_signal_noise_ratio(
        clean_frame,
        output_frame,
        data_range=255
    )

    ssim = structural_similarity(
        clean_frame,
        output_frame,
        channel_axis=2,
        data_range=255
    )

    return psnr, ssim