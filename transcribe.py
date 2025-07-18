import os
import re
import json
import unicodedata
from yt_dlp import YoutubeDL
from dotenv import load_dotenv

try:
    import webvtt
except ImportError:
    print("webvtt-py not found. Installing...")
    os.system("pip install webvtt-py")
    import webvtt

def slugify(text):
    """Convert text to a safe filename slug."""
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    text = re.sub(r'[\s]+', '_', text)
    return text

def fetch_playlist_videos(playlist_url):
    """Fetch video metadata from a YouTube playlist."""
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'skip_download': True,
        'force_generic_extractor': True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        entries = info.get('entries', [])
        return [
            {
                'id': entry['id'].strip(),
                'title': entry['title'].strip(),
            }
            for entry in entries if entry.get('id') and entry.get('title')
        ]

def download_vtt(video_id, outdir):
    """Download VTT subtitle file for a given video ID."""
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'subtitlesformat': 'vtt',
        'outtmpl': os.path.join(outdir, f"{video_id}.%(ext)s"),
    }
    with YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
        subs = result.get('subtitles', {}) or result.get('automatic_captions', {})
        # Check if 'en' exists
        if 'en' not in subs:
            return False
        # Actually download
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        return True

def convert_vtt_to_json(vtt_path, video_meta):
    """
    Convert VTT subtitle file to structured JSON with timestamps.
    
    Args:
        vtt_path (str): Path to the .vtt file
        video_meta (dict): Video metadata containing 'id', 'title', etc.
    
    Returns:
        dict: JSON payload following the schema in prompt.md
    """
    try:
        # Parse VTT using webvtt-py
        vtt_captions = webvtt.read(vtt_path)
        
        # Extract video duration if available (optional)
        duration = None
        # Note: Duration extraction would require additional yt-dlp call if needed
        
        # Build captions array
        captions = []
        for index, caption in enumerate(vtt_captions):
            # Convert timestamp to seconds (webvtt-py handles this)
            start_seconds = caption.start_in_seconds
            end_seconds = caption.end_in_seconds
            
            # Clean text (remove formatting tags)
            clean_text = re.sub(r'<[^>]+>', '', caption.text)
            clean_text = clean_text.replace('\n', ' ').strip()
            
            captions.append({
                "index": index,
                "start": start_seconds,
                "end": end_seconds,
                "text": clean_text
            })
        
        # Build final payload
        payload = {
            "video_url": f"https://youtube.com/watch?v={video_meta['id']}",
            "title": video_meta['title'],
            "captions": captions
        }
        
        # Add duration if available
        if duration is not None:
            payload["duration"] = duration
            
        return payload
        
    except Exception as e:
        print(f"Error parsing VTT file {vtt_path}: {e}")
        return None

def json_to_markdown(json_payload):
    """
    Convert JSON payload to markdown format (for optional export).
    
    Args:
        json_payload (dict): JSON data with video and caption information
    
    Returns:
        str: Markdown formatted content
    """
    md_lines = [f"# Transcript for {json_payload['video_url']}\n"]
    
    for caption in json_payload['captions']:
        md_lines.append(caption['text'])
    
    return "\n\n".join(md_lines)

def should_process_file(vtt_path, json_path):
    """
    Check if we need to process the VTT file based on modification times.
    Returns True if JSON doesn't exist or VTT is newer than JSON.
    """
    if not os.path.exists(json_path):
        return True
    
    vtt_mtime = os.path.getmtime(vtt_path)
    json_mtime = os.path.getmtime(json_path)
    
    return vtt_mtime > json_mtime

def main():
    """Main function to process YouTube playlist and extract transcripts."""
    load_dotenv()
    playlist_url = os.getenv("YOUTUBE_PLAYLIST_URL")
    save_md = os.getenv("SAVE_MD", "false").lower() == "true"
    
    if not playlist_url:
        print("No YOUTUBE_PLAYLIST_URL found in .env file.")
        return

    # Create directory structure as specified in prompt.md
    transcripts_dir = "transcripts"
    json_dir = os.path.join(transcripts_dir, "json")
    md_dir = os.path.join(transcripts_dir, "md")
    vtt_dir = "vtt"
    
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(vtt_dir, exist_ok=True)
    if save_md:
        os.makedirs(md_dir, exist_ok=True)

    videos = fetch_playlist_videos(playlist_url)
    print(f"Found {len(videos)} videos in playlist.")

    for idx, video in enumerate(videos, 1):
        safe_title = slugify(video['title'])
        vtt_path = os.path.join(vtt_dir, f"{video['id']}.en.vtt")
        json_path = os.path.join(json_dir, f"{safe_title}.json")
        video_url = f"https://www.youtube.com/watch?v={video['id']}"
        
        print(f"[{idx}/{len(videos)}] Processing: {video['title']}")
        
        # Download subtitle if not already downloaded
        if not os.path.exists(vtt_path):
            success = download_vtt(video['id'], vtt_dir)
            if not success:
                print(f"  Skipped: No English subtitle for {video['title']}")
                continue
        
        # Check if we need to process this file (idempotent behavior)
        if not should_process_file(vtt_path, json_path):
            print(f"  Skipped: JSON already up to date for {safe_title}")
            continue
        
        # Convert VTT to JSON
        json_payload = convert_vtt_to_json(vtt_path, video)
        if json_payload is None:
            print(f"  Error: Failed to convert VTT to JSON for {video['title']}")
            continue
        
        # Validate JSON payload
        if not json_payload.get('captions'):
            print(f"  Warning: Empty caption list for {video['title']}")
            continue
        
        # Save JSON file
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_payload, f, indent=2, ensure_ascii=False)
            print(f"  Saved JSON: {json_path}")
        except Exception as e:
            print(f"  Error saving JSON: {e}")
            continue
        
        # Optionally save markdown file
        if save_md:
            md_path = os.path.join(md_dir, f"{safe_title}.md")
            try:
                markdown = json_to_markdown(json_payload)
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(markdown)
                print(f"  Saved markdown: {md_path}")
            except Exception as e:
                print(f"  Error saving markdown: {e}")

def get_available_videos():
    """
    Get list of available videos that have been processed (have JSON files).
    
    Returns:
        list: List of video dictionaries with 'id' and 'title' keys
    """
    json_dir = os.path.join("transcripts", "json")
    if not os.path.exists(json_dir):
        return []
    
    videos = []
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(json_dir, filename)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract video ID from URL
                video_url = data.get('video_url', '')
                if 'v=' in video_url:
                    video_id = video_url.split('v=')[1].split('&')[0]
                else:
                    # Use filename as fallback
                    video_id = filename.replace('.json', '')
                
                videos.append({
                    'id': video_id,
                    'title': data.get('title', filename.replace('.json', ''))
                })
            except Exception as e:
                print(f"Warning: Could not read {filename}: {e}")
                continue
    
    return videos


if __name__ == "__main__":
    main()
