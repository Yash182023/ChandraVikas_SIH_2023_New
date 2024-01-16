crop_dim24(srgan_config.stp1_sr_dir, srgan_config.dim24_dir, 960)

    # ## second forward pass
    # for dirs in natsorted(os.listdir(srgan_config.dim24_dir)):
    #     # Get a list of test image file names.
    #     file_names = natsorted(os.listdir(os.path.join(srgan_config.dim24_dir, dirs)))
    #     total_files = len(file_names)

    #     print(f"Processing `{os.path.abspath(os.path.join(srgan_config.dim24_dir, dirs))}` : second itr...")

    #     for index in range(total_files):
    #         lr_image_path = os.path.join(srgan_config.dim24_dir, dirs, file_names[index])
    #         make_directory(os.path.join(srgan_config.dim96_dir, dirs))
    #         sr_image_path = os.path.join(srgan_config.dim96_dir, dirs, file_names[index])

    #         lr_tensor = imgproc.preprocess_one_image(lr_image_path, srgan_config.device)

    #         # Only reconstruct the Y channel image data.
    #         with torch.no_grad():
    #             sr_tensor = g_model(lr_tensor)

    #         # Save image
    #         sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
    #         sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2GRAY)
    #         cv2.imwrite(sr_image_path, sr_image)

    # ## second merge
    # # make_directory(srgan_config.stp2_sr_dir)
    # merge_dim96(srgan_config.dim96_dir, srgan_config.stp2_sr_dir, 3840)

    # ## second improve
    # improve(srgan_config.stp2_sr_dir)

    # ## clean intermediate files
    # clean_dir(srgan_config.dim24_dir)
    # clean_dir(srgan_config.dim96_dir)