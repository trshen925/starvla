def process_vision_info(messages):
    images, videos = [], []
    for message in messages:
        for item in message.get('content', []):
            if item.get('type') == 'image': images.append(item.get('image'))
            elif item.get('type') == 'video': videos.append(item.get('video'))
    return images, videos
