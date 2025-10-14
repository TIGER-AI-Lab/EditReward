import requests
import numpy as np
import av
import regex as re
import shutil
from pathlib import Path
video_cache_dir = Path(__file__).parent / "video_cache"
if not video_cache_dir.exists():
    video_cache_dir.mkdir(exist_ok=True)
from huggingface_hub import hf_hub_download

def load_template(task, template):
    with open(Path(__file__).parent / "templates" / task / f"{template}.txt") as f:
        return f.read()
    
def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def download_video(url, file_path):
    if "huggingface.co/datasets" in url:
        # https://huggingface.co/datasets/tianleliphoebe/genai-arena-video-mp4/blob/main/dir/79e4e57959b4411e8d7532d655cbd6c6.mp4
        # repo_id = tianleliphoebe/genai-arena-video-mp4
        # revision = main
        # file_path = 79e4e57959b4411e8d7532d655cbd6c6.mp4
        # hf_hub_download(repo_id="tianleliphoebe/genai-arena-video-mp4", filename="79e4e57959b4411e8d7532d655cbd6c6.mp4", repo_type="dataset", local_dir="./video")
        _url = url.split("datasets/")[1]
        repo_id = _url.split("/blob")[0]
        revision_filename = _url.split("/blob/")[1]
        revision = revision_filename[:revision_filename.find("/")]
        filename = revision_filename[revision_filename.find("/")+1:]
        hub_file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision)
        shutil.copy(hub_file_path, file_path)
        return file_path
    else:
        r = requests.get(url, stream=True)
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return file_path


def process_video_into_frames(video_path, max_num_frames, return_path=False):
    if video_path.startswith("http"):
        video_path = download_video(video_path, video_cache_dir / Path(video_path).name)
    else:
        video_path = Path(video_path)
    container = av.open(video_path)
    num_frames = container.streams.video[0].frames
    if num_frames > max_num_frames:
        indices = np.arange(0, num_frames, num_frames / max_num_frames).astype(int)
    else:
        indices = np.arange(num_frames)
        
    frames = read_video_pyav(container, indices)
    if return_path:
        return frames, video_path
    else:
        return frames

def merge_video_into_frames(video_paths, max_num_frames):
    assert isinstance(video_paths, list), "video_paths must be a list of video paths."
    frames = []
    for video_path in video_paths:
        frames.append(process_video_into_frames(video_path, max_num_frames))
    # sample the frames to the same length
    frames = np.concatenate(frames, axis=0)
    sampled_indices = np.arange(0, frames.shape[0], frames.shape[0] / max_num_frames).astype(int)
    return frames[sampled_indices]