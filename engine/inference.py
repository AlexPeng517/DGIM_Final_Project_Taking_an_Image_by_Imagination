import torch

def embed_camera_settings(focal_length=None, aperture=None, iso_speed=None, exposure_time=None, prompt_embeds=None, 
                          negative_focal_length=None, negative_aperture=None, 
                          negative_iso_speed=None, negative_exposure_time=None, negative_prompt_embeds=None, 
                          cam_embed=None, device=None):
    # embed camera settings
    if focal_length is not None and isinstance(focal_length, int):
        focal_lengths = torch.tensor([[float(focal_length)]]).to(device=device, dtype=prompt_embeds.dtype)
    elif focal_length is not None and isinstance(focal_length, float):
        focal_lengths = torch.tensor([[focal_length]]).to(device=device, dtype=prompt_embeds.dtype)
    else:
        focal_lengths = None

    if aperture is not None and isinstance(aperture, int):
        apertures = torch.tensor([[float(aperture)]]).to(device=device, dtype=prompt_embeds.dtype)
    elif aperture is not None and isinstance(aperture, float):
        apertures = torch.tensor([[aperture]]).to(device=device, dtype=prompt_embeds.dtype)
    else:
        apertures = None

    if iso_speed is not None and isinstance(iso_speed, int):
        iso_speeds = torch.tensor([[float(iso_speed)]]).to(device=device, dtype=prompt_embeds.dtype)
    elif iso_speed is not None and isinstance(iso_speed, float):
        iso_speeds = torch.tensor([[iso_speed]]).to(device=device, dtype=prompt_embeds.dtype)
    else:
        iso_speeds = None

    if exposure_time is not None and isinstance(exposure_time, int):
        exposure_times = torch.tensor([[float(exposure_time)]]).to(device=device, dtype=prompt_embeds.dtype)
    elif exposure_time is not None and isinstance(exposure_time, float):
        exposure_times = torch.tensor([[exposure_time]]).to(device=device, dtype=prompt_embeds.dtype)
    else:
        exposure_times = None

    if negative_focal_length is not None and isinstance(negative_focal_length, int):
        negative_focal_length = torch.tensor([[float(negative_focal_length)]]).to(device=device, dtype=prompt_embeds.dtype)
    elif negative_focal_length is not None and isinstance(negative_focal_length, float):
        negative_focal_length = torch.tensor([[negative_focal_length]]).to(device=device, dtype=prompt_embeds.dtype)
    else:
        negative_focal_length = None

    if negative_aperture is not None and isinstance(negative_aperture, int):
        negative_aperture = torch.tensor([[float(negative_aperture)]]).to(device=device, dtype=prompt_embeds.dtype)
    elif negative_aperture is not None and isinstance(negative_aperture, float):
        negative_aperture = torch.tensor([[negative_aperture]]).to(device=device, dtype=prompt_embeds.dtype)
    else:
        negative_aperture = None
    
    if negative_iso_speed is not None and isinstance(negative_iso_speed, int):
        negative_iso_speed = torch.tensor([[float(negative_iso_speed)]]).to(device=device, dtype=prompt_embeds.dtype)
    elif negative_iso_speed is not None and isinstance(negative_iso_speed, float):
        negative_iso_speed = torch.tensor([[negative_iso_speed]]).to(device=device, dtype=prompt_embeds.dtype)
    else:
        negative_iso_speed = None

    if negative_exposure_time is not None and isinstance(negative_exposure_time, int):
        negative_exposure_time = torch.tensor([[float(negative_exposure_time)]]).to(device=device, dtype=prompt_embeds.dtype)
    elif negative_exposure_time is not None and isinstance(negative_exposure_time, float):
        negative_exposure_time = torch.tensor([[negative_exposure_time]]).to(device=device, dtype=prompt_embeds.dtype)
    else:
        negative_exposure_time = None
    

    if focal_lengths is not None and apertures is not None \
        and iso_speeds is not None and exposure_times is not None:
        camera_settings_embeds = cam_embed(
            focal_lengths, apertures, iso_speeds, exposure_times)
        if negative_focal_length is not None and negative_aperture is not None \
            and negative_iso_speed is not None and negative_exposure_time is not None:
            negative_camera_settings_embeds = cam_embed(
                negative_focal_length, negative_aperture, negative_iso_speed, negative_exposure_time)
        elif negative_focal_length is None and negative_aperture is None \
            and negative_iso_speed is None and negative_exposure_time is None:
            negative_camera_settings_embeds = cam_embed(
                torch.tensor([[0.0]]).to(device=device, dtype=prompt_embeds.dtype),
                torch.tensor([[0.0]]).to(device=device, dtype=prompt_embeds.dtype),
                torch.tensor([[0.0]]).to(device=device, dtype=prompt_embeds.dtype),
                torch.tensor([[0.0]]).to(device=device, dtype=prompt_embeds.dtype)
            )
        else:
            if negative_focal_length is None:
                negative_focal_length = torch.tensor([[0.0]]).to(device=device, dtype=prompt_embeds.dtype)
            if negative_aperture is None:
                negative_aperture = torch.tensor([[0.0]]).to(device=device, dtype=prompt_embeds.dtype)
            if negative_iso_speed is None:
                negative_iso_speed = torch.tensor([[0.0]]).to(device=device, dtype=prompt_embeds.dtype)
            if negative_exposure_time is None:
                negative_exposure_time = torch.tensor([[0.0]]).to(device=device, dtype=prompt_embeds.dtype)
            negative_camera_settings_embeds = cam_embed(
                negative_focal_length, negative_aperture, negative_iso_speed, negative_exposure_time)      
    else:
        camera_settings_embeds = []
        negative_camera_settings_embeds = []
        if focal_lengths is not None:
            camera_settings_embeds.append(cam_embed.focal_length_forward(focal_lengths))
            if negative_focal_length is not None:
                negative_camera_settings_embeds.append(cam_embed.focal_length_forward(negative_focal_length))
            else:
                negative_camera_settings_embeds.append(cam_embed.focal_length_forward(torch.tensor([[0.0]]).to(device=device, dtype=prompt_embeds.dtype)))
        if apertures is not None:
            camera_settings_embeds.append(cam_embed.aperture_forward(apertures))
            if negative_aperture is not None:
                negative_camera_settings_embeds.append(cam_embed.aperture_forward(negative_aperture))
            else:
                negative_camera_settings_embeds.append(cam_embed.aperture_forward(torch.tensor([[0.0]]).to(device=device, dtype=prompt_embeds.dtype)))
        if iso_speeds is not None:
            camera_settings_embeds.append(cam_embed.iso_speed_forward(iso_speeds))
            if negative_iso_speed is not None:
                negative_camera_settings_embeds.append(cam_embed.iso_speed_forward(negative_iso_speed))
            else:
                negative_camera_settings_embeds.append(cam_embed.iso_speed_forward(torch.tensor([[0.0]]).to(device=device, dtype=prompt_embeds.dtype)))
        if exposure_times is not None:
            camera_settings_embeds.append(cam_embed.exposure_time_forward(exposure_times))
            if negative_exposure_time is not None:
                negative_camera_settings_embeds.append(cam_embed.exposure_time_forward(negative_exposure_time))
            else:
                negative_camera_settings_embeds.append(cam_embed.exposure_time_forward(torch.tensor([[0.0]]).to(device=device, dtype=prompt_embeds.dtype)))
        camera_settings_embeds = torch.cat(camera_settings_embeds, dim=1)
        negative_camera_settings_embeds = torch.cat(negative_camera_settings_embeds, dim=1)
        
    prompt_embeds = torch.cat([prompt_embeds, camera_settings_embeds], dim=1)
    negative_prompt_embeds = torch.cat([negative_prompt_embeds, negative_camera_settings_embeds], dim=1)

    return prompt_embeds, negative_prompt_embeds