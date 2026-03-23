INPUT_SCHEMA = {
    'mode': {
        'type': str,
        'required': False,
        'default': 'txt2img',
        'constraints': lambda m: m in ('txt2img', 'img2img', 'inpainting')
    },
    'prompt': {
        'type': str,
        'required': False,
    },
    'image': {
        'type': str,
        'required': False,
        'default': None
    },
    'mask_image': {
        'type': str,
        'required': False,
        'default': None
    },
    'strength': {
        'type': float,
        'required': False,
        'default': 0.75
    },
    'height': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'width': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 4
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 0.0
    },
    'num_images': {
        'type': int,
        'required': False,
        'default': 1,
        'constraints': lambda img_count: 3 > img_count > 0
    },
}
