from torchvision import transforms


def get_transform(type='clip', keep_ratio=True, image_size=224):
    if type == 'clip':
        transform = []
        if keep_ratio:
            transform.extend([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ])
        else:
            transform.append(transforms.Resize((image_size, image_size)))
        transform.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        return transforms.Compose(transform)
    elif type == 'clipa':
        transform = []
        if keep_ratio:
            transform.extend([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ])
        else:
            transform.append(transforms.Resize((image_size, image_size)))
        transform.extend(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        return transforms.Compose(transform)
    elif type == 'sd':
        transform = []
        if keep_ratio:
            transform.extend([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
            ])
        else:
            transform.append(
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC))
        transform.extend([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        return transforms.Compose(transform)
    else:
        raise NotImplementedError
